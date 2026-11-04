"""Shared rules for skera's S-reads: false cuts, adapters left inside an S-read, molecule coverage, and mirror junctions."""
import polars as pl

from .molecules import USABLE
from .patterns import MAS_ORDER

COVER = 0.8
TOUCH = 0.2
TOL = 5

_CHUNKS = 16


def false_cuts(sreads: pl.DataFrame, features: pl.DataFrame, tol: int = TOL) -> pl.DataFrame:
    """Find the gaps between consecutive S-reads of a ZMW that hold none of our MAS adapters within tol bp of either edge."""
    s = sreads.sort("zm", "qs").with_columns(pl.col("qs", "qe").cast(pl.Int64))
    mas = features.filter(pl.col("kind") == "MAS").with_columns(pl.col("start", "end").cast(pl.Int64))
    parts = []
    for c in range(_CHUNKS):
        sc = s.filter(pl.col("zm") % _CHUNKS == c)
        nxt = sc.select(pl.all().shift(-1).over("zm").name.suffix("_after"))
        g = (pl.concat([sc, nxt], how="horizontal")
             .filter(pl.col("di_after").is_not_null())
             .rename({"di": "di_before", "qe": "cut_start", "qs_after": "cut_end", "dr": "dr_before"})
             .with_row_index("gid"))
        mc = mas.filter(pl.col("zm") % _CHUNKS == c)
        has_adapter = (g.select("gid", "zm", "cut_start", "cut_end").join(mc, on="zm")
                       .filter((pl.col("start") < pl.col("cut_end") + tol) & (pl.col("end") > pl.col("cut_start") - tol))["gid"].unique())
        parts.append(g.filter(~pl.col("gid").is_in(has_adapter)))
    fc = pl.concat(parts)
    return fc.select("zm", "di_before", "di_after", "cut_start", "cut_end", "dr_before").sort("zm", "cut_start")


def adapters_inside_sreads(sreads: pl.DataFrame, features: pl.DataFrame, tol: int = TOL, strict: bool = False) -> pl.DataFrame:
    """Find our MAS adapters that lie more than tol bp inside an S-read: inclusive of the edge by default, strict when asked."""
    s = sreads.with_columns(pl.col("qs", "qe").cast(pl.Int64))
    mas = features.filter(pl.col("kind") == "MAS").with_columns(pl.col("start", "end").cast(pl.Int64))
    parts = []
    for c in range(_CHUNKS):
        sc = s.filter(pl.col("zm") % _CHUNKS == c)
        mc = mas.filter(pl.col("zm") % _CHUNKS == c)
        j = sc.join(mc, on="zm")
        if strict:
            cond = (pl.col("qs") + tol < pl.col("start")) & (pl.col("end") < pl.col("qe") - tol)
        else:
            cond = (pl.col("start") >= pl.col("qs") + tol) & (pl.col("end") <= pl.col("qe") - tol)
        parts.append(j.filter(cond))
    out = pl.concat(parts)
    return out.select("zm", "di", "name", "strand", "start", "end", "edits").sort("zm", "di", "start")


def _overlaps(mols: pl.DataFrame, sreads: pl.DataFrame) -> pl.DataFrame:
    """mols joined with the S-reads of the same ZMW, with ov: the fraction of the molecule the S-read covers."""
    return (mols.join(sreads, on="zm")
            .with_columns(((pl.min_horizontal("end", "qe") - pl.max_horizontal("start", "qs")).clip(0)
                           / (pl.col("end") - pl.col("start"))).alias("ov")))


def sread_overlaps(molecules: pl.DataFrame, sreads: pl.DataFrame) -> pl.DataFrame:
    """Every molecule paired with every S-read of its ZMW: the molecule's columns, the S-read's, and ov, the fraction of
    the molecule the S-read covers (the overlap molecule_coverage classifies by). Sorted by zm, start and qs."""
    mols = molecules.with_columns(pl.col("start", "end").cast(pl.Int64))
    sr = sreads.with_columns(pl.col("qs", "qe").cast(pl.Int64))
    parts = [_overlaps(mols.filter(pl.col("zm") % _CHUNKS == c), sr.filter(pl.col("zm") % _CHUNKS == c)) for c in range(_CHUNKS)]
    return pl.concat(parts).sort("zm", "start", "qs")


def molecule_coverage(molecules: pl.DataFrame, sreads: pl.DataFrame, cover: float = COVER, touch: float = TOUCH) -> pl.DataFrame:
    """Classify every molecule by how the S-reads of its ZMW cover it.

    CLEAN when one S-read covers at least cover of it and at least touch of no other molecule; FUSED when that S-read
    also covers another molecule by at least touch; PARTIAL when the best S-read covers at least touch but no S-read
    reaches cover alone; NOT_WRITTEN otherwise, including when the ZMW has no S-read at all.
    """
    mols = molecules.with_row_index("mid").with_columns(pl.col("start", "end").cast(pl.Int64))
    sr = sreads.with_columns(pl.col("qs", "qe").cast(pl.Int64))
    parts = []
    for c in range(_CHUNKS):
        mc = mols.filter(pl.col("zm") % _CHUNKS == c)
        sc = sr.filter(pl.col("zm") % _CHUNKS == c)
        pairs = _overlaps(mc, sc)
        n_touch = pairs.filter(pl.col("ov") >= touch).group_by("zm", "di").agg(pl.len().alias("n_touch"))
        pairs = pairs.join(n_touch, on=["zm", "di"], how="left").with_columns(pl.col("n_touch").fill_null(0))
        per = pairs.group_by("mid").agg(
            pl.col("ov").max().alias("maxov"),
            ((pl.col("ov") >= cover) & (pl.col("n_touch") == 1)).any().alias("clean"),
            ((pl.col("ov") >= cover) & (pl.col("n_touch") > 1)).any().alias("fused"),
            pl.col("di").filter((pl.col("ov") >= cover) & (pl.col("n_touch") == 1)).min().alias("clean_di"))
        parts.append(mc.join(per, on="mid", how="left"))
    out = pl.concat(parts)
    out = out.with_columns(
        pl.when(pl.col("clean")).then(pl.lit("CLEAN")).when(pl.col("fused")).then(pl.lit("FUSED"))
        .when(pl.col("maxov") >= touch).then(pl.lit("PARTIAL")).otherwise(pl.lit("NOT_WRITTEN")).alias("cls"))
    return out.sort("mid").drop("clean", "fused")


def best_cover(molecules: pl.DataFrame, sreads: pl.DataFrame) -> pl.DataFrame:
    """Find, for every molecule, the largest fraction of it that any single S-read of its ZMW covers."""
    mols = molecules.with_columns(pl.col("start", "end").cast(pl.Int64))
    sr = sreads.with_columns(pl.col("qs", "qe").cast(pl.Int64))
    parts = []
    for c in range(_CHUNKS):
        mc = mols.filter(pl.col("zm") % _CHUNKS == c)
        sc = sr.filter(pl.col("zm") % _CHUNKS == c)
        cov = (mc.join(sc, on="zm", how="left")
               .with_columns((pl.min_horizontal("end", "qe") - pl.max_horizontal("start", "qs")).clip(0).alias("ov"))
               .group_by("zm", "start", "end").agg((pl.col("ov").max().fill_null(0) / (pl.col("end") - pl.col("start")).first()).alias("cover")))
        parts.append(cov)
    out = pl.concat(parts)
    return out.select("zm", "start", "end", "cover").sort("zm", "start")


def mirror_junctions(molecules: pl.DataFrame) -> pl.DataFrame:
    """Find molecules whose two flanking tokens are the same letter in opposite orientation, a mirror junction."""
    mols = molecules.with_columns(pl.col("start", "end").cast(pl.Int64))
    return (mols.filter(
        pl.col("left_token").is_not_null() & pl.col("right_token").is_not_null()
        & (pl.col("left_token").str.strip_chars("'") == pl.col("right_token").str.strip_chars("'"))
        & (pl.col("left_token").str.ends_with("'") != pl.col("right_token").str.ends_with("'")))
        .with_columns(pl.col("left_token").str.strip_chars("'").alias("letter")))


def single_tso_tso_mirrors(molecules: pl.DataFrame) -> pl.DataFrame:
    """Find ZMWs with exactly one mirror junction, when that junction's molecule is a TSO-TSO molecule."""
    j = mirror_junctions(molecules)
    return (j.group_by("zm").agg(pl.len().alias("nj"), pl.col("start").first().alias("js"), pl.col("end").first().alias("je"),
                                  pl.col("end_class").first().alias("jclass"), pl.col("letter").first())
            .filter((pl.col("nj") == 1) & (pl.col("jclass") == "tso_tso"))
            .select("zm", "js", "je", "letter").sort("zm"))


def passing_from_labels(sreads: pl.DataFrame, order_len: int = 9) -> pl.DataFrame:
    """Derive skera's pass/fail verdict from the S-read adapter labels alone, for the reader command's
    case (it gets only segmented.bam, not segmented.non_passing.bam): a ZMW is passing when, in read
    order (S-reads sorted by qs), its labels (dl, dr) chain from the first adapter to the last, either
    forwards (dl[0] == 0, dr[-1] == order_len - 1, dr == dl + 1 for every S-read) or backwards when the
    first S-read has dl > dr (dl[0] == order_len - 1, dr[-1] == 0, dr == dl - 1), with every S-read's dr
    equal to the next one's dl either way. Coordinates play no part, so a complete read with a false cut
    still passes this check: observed in BD144 kin, the letter skera calls at a false cut equals the
    adapter it skipped in 99.0% of false cuts (missed_same_letter_as_skera_call 0.9904 in
    tests/golden/v10_inputs/false_cut_anatomy/BD144_kin_false_cuts.json), so the label chain stays
    intact at almost every false cut.
    ZMWs without S-reads are not in the result (the caller treats them as failing).

    Measured against skera's own verdict (segmented.non_passing.bam) on BD144: agreement on complete
    reads 1.000000 in ND (1,615,591 reads), HM (2,040,035) and kin (1,960,640), zero disagreements; on
    all reads 0.999990 / 0.999975 / 0.999975 (20, 51, 50 reads skera passes that this rule calls
    failing). Both above the task's 0.999 threshold, so the reader command has no --non-passing option.
    """
    s = (sreads.sort("zm", "qs")
         .with_columns(pl.col("dl", "dr").cast(pl.Int64))
         .with_columns(pl.col("dl").first().over("zm").alias("dl0"), pl.col("dr").first().over("zm").alias("dr0"),
                       pl.col("dl").shift(-1).over("zm").alias("dl_next")))
    reversed_ = pl.col("dl0") > pl.col("dr0")
    step_ok = pl.when(reversed_).then(pl.col("dr") == pl.col("dl") - 1).otherwise(pl.col("dr") == pl.col("dl") + 1)
    link_ok = pl.col("dl_next").is_null() | (pl.col("dr") == pl.col("dl_next"))
    out = (s.group_by("zm", maintain_order=True).agg(
        pl.col("dl0").first(), pl.col("dr0").first(), pl.col("dr").last().alias("dr_last"),
        step_ok.all().alias("steps_ok"), link_ok.all().alias("links_ok"))
        .with_columns(
            pl.when(pl.col("dl0") > pl.col("dr0"))
            .then((pl.col("dl0") == order_len - 1) & (pl.col("dr_last") == 0))
            .otherwise((pl.col("dl0") == 0) & (pl.col("dr_last") == order_len - 1))
            .alias("endpoints_ok")))
    return out.with_columns((pl.col("steps_ok") & pl.col("links_ok") & pl.col("endpoints_ok")).alias("passing")).select("zm", "passing")


def skipped_adapters(fc: pl.DataFrame, sreads: pl.DataFrame, features: pl.DataFrame, tol: int = TOL) -> pl.DataFrame:
    """For every false cut (skera.false_cuts' output), the MAS hit of ours nearest the cut that lies
    inside the S-read before or after it, with the inclusive rule of skera.adapters_inside_sreads
    (strict=False): cut = (cut_start + cut_end) // 2, nearest by |start - cut| with start as the
    tie-break. Columns zm, cut_start, missed, in_sread (1 before, 2 after), dist; null missed when
    neither S-read holds one. The rule: the adapter skera's call skipped at a false cut is the one of
    ours, among those still inside a flanking S-read, closest to the cut. study/analyses/false_cut_anatomy.py
    will use this same rule in this repo's layout."""
    base = fc.with_row_index("gid").with_columns(((pl.col("cut_start") + pl.col("cut_end")) // 2).alias("cut"))
    inside = adapters_inside_sreads(sreads, features, tol=tol, strict=False).select("zm", "di", "name", "start")
    before = base.select("gid", "zm", pl.col("di_before").alias("di"), "cut").join(inside, on=["zm", "di"]) \
        .with_columns(pl.lit(1).alias("in_sread"))
    after = base.select("gid", "zm", pl.col("di_after").alias("di"), "cut").join(inside, on=["zm", "di"]) \
        .with_columns(pl.lit(2).alias("in_sread"))
    cand = (pl.concat([before, after]).select("gid", "name", "start", "cut", "in_sread")
            .with_columns((pl.col("start") - pl.col("cut")).abs().alias("dist")))
    best = (cand.sort("dist", "start").group_by("gid", maintain_order=True)
            .agg(pl.col("name").first().alias("missed"), pl.col("in_sread").first(), pl.col("dist").first()))
    out = base.join(best, on="gid", how="left").select("zm", "cut_start", "missed", "in_sread", "dist")
    return out.sort("zm", "cut_start")


def false_cut_stats(read_summary: pl.DataFrame, features: pl.DataFrame, sreads: pl.DataFrame, *,
                     passing: pl.DataFrame, fc: pl.DataFrame, order=MAS_ORDER) -> dict:
    """summary.json["skera"]'s false-cut numbers: complete_reads (our complete pattern),
    complete_passing_reads (derived via passing_from_labels), false_cuts, false_cut_reads_share
    (reads with at least one false cut / complete_passing_reads), false_cuts_per_read,
    false_cut_letters (letter skera called -> count) and skipped_adapter_letters (letter of ours
    skera skipped -> count, "none" for false cuts without one).

    passing and fc are passing_from_labels' and false_cuts' output over the full sreads/features
    (every ZMW, not just this function's target), taken as arguments so the caller computes each
    once: false_cuts groups entirely by zm (.over("zm")), so filtering its output down to this
    function's target ZMWs gives the same rows as running it on sreads/features pre-filtered to
    that target."""
    complete = read_summary.filter(pl.col("primary_pattern") == "complete").select("zm")
    pass_zm = passing.filter(pl.col("passing")).select("zm")
    target = complete.join(pass_zm, on="zm")["zm"]
    s = sreads.filter(pl.col("zm").is_in(target))
    f = features.filter(pl.col("zm").is_in(target))
    fc_t = fc.filter(pl.col("zm").is_in(target))
    sk = skipped_adapters(fc_t, s, f)
    n = target.len()

    letter_of = dict(enumerate(order))
    letters = (fc_t.with_columns(pl.col("dr_before").cast(pl.Int64).replace_strict(letter_of, return_dtype=pl.Utf8).alias("letter"))
               .group_by("letter").agg(pl.len().alias("n")).sort("letter"))
    false_cut_letters = dict(zip(letters["letter"].to_list(), letters["n"].to_list()))
    missed = (sk.with_columns(pl.col("missed").fill_null("none"))
              .group_by("missed").agg(pl.len().alias("n")).sort("missed"))
    skipped_adapter_letters = dict(zip(missed["missed"].to_list(), missed["n"].to_list()))

    return {
        "complete_reads": complete.height,
        "complete_passing_reads": n,
        "false_cuts": fc_t.height,
        "false_cut_reads_share": (fc_t["zm"].n_unique() / n) if n else float("nan"),
        "false_cuts_per_read": (fc_t.height / n) if n else float("nan"),
        "false_cut_letters": false_cut_letters,
        "skipped_adapter_letters": skipped_adapter_letters,
    }


def mirror_arm_classes(molecules: pl.DataFrame, sreads: pl.DataFrame) -> pl.DataFrame:
    """Every proper, non-internal molecule of a single TSO-TSO mirror read (skera.single_tso_tso_mirrors),
    with its region before (end <= js) or after (start >= je) the junction, and its class under
    molecule_coverage computed on all molecules of that read. The region rule: a molecule lies before
    the junction when it ends at or before the junction's start, after when it starts at or after the
    junction's end. study/analyses/mirror_arms.py will use this same rule in this repo's layout."""
    st = single_tso_tso_mirrors(molecules)
    mols = molecules.join(st.select("zm", "js", "je"), on="zm")
    sr = sreads.filter(pl.col("zm").is_in(st["zm"]))
    cov = molecule_coverage(mols, sr)
    return (cov.filter(USABLE)
            .with_columns(pl.when(pl.col("end") <= pl.col("js")).then(pl.lit("before"))
                          .when(pl.col("start") >= pl.col("je")).then(pl.lit("after"))
                          .otherwise(pl.lit("other")).alias("region"))
            .select("zm", "region", "cls"))


def mirror_arm_stats(molecules: pl.DataFrame, sreads: pl.DataFrame) -> dict:
    """summary.json["skera"]'s mirror_arms: CLEAN shares (and counts by class) on the arm after and
    before a single TSO-TSO mirror junction (mirror_arm_classes' before/after region rule), as the
    share of each arm's molecules that molecule_coverage classifies CLEAN, plus the raw counts per
    class. study/analyses/mirror_arms.py will use this same rule in this repo's layout."""
    arm = mirror_arm_classes(molecules, sreads)
    out = {}
    for region in ("after", "before"):
        r = arm.filter(pl.col("region") == region)
        n = r.height
        counts = {cls: r.filter(pl.col("cls") == cls).height for cls in ("CLEAN", "FUSED", "PARTIAL", "NOT_WRITTEN")}
        out[f"{region}_clean_share"] = (counts["CLEAN"] / n) if n else float("nan")
        out[f"{region}_counts"] = counts
    return out


def budget_classes(molecules: pl.DataFrame, sreads: pl.DataFrame) -> pl.DataFrame:
    """Every molecule's slot-budget role: usable (proper, not internal) and, from molecule_coverage
    computed on all molecules of the read, its class (CLEAN/FUSED/PARTIAL/NOT_WRITTEN). The usable
    rule: a molecule is usable when its end_class is proper and it is not internal (fused).
    study/analyses/molecule_budget.py will use this same rule, without the lima, refine or mapping
    columns (no lima, refine or mapping run here)."""
    return molecule_coverage(molecules, sreads).with_columns(USABLE.alias("usable"))


def budget_per_read(cls: pl.DataFrame) -> pl.DataFrame:
    """Per ZMW with at least one molecule in `cls` (skera.budget_classes' output): molecule count, the
    four artifact kinds, usable molecules and their clean/not-clean split, one row per ZMW.
    study/analyses/molecule_budget.py will use this same per-read aggregation."""
    return (cls.group_by("zm", maintain_order=True).agg(
        pl.len().alias("molecules"),
        (~pl.col("usable") & (pl.col("end_class") == "tso_tso")).sum().alias("artifact_tso_tso"),
        (~pl.col("usable") & (pl.col("end_class") == "r1_r1")).sum().alias("artifact_r1_r1"),
        (~pl.col("usable") & pl.col("end_class").is_in(["tso_only", "r1_only", "none"])).sum().alias("artifact_one_primer_or_none"),
        ((pl.col("end_class") == "proper") & pl.col("internal")).sum().alias("artifact_fused"),
        pl.col("usable").sum().alias("usable"),
        (pl.col("usable") & (pl.col("cls") == "CLEAN")).sum().alias("usable_clean"),
        (pl.col("usable") & (pl.col("cls") != "CLEAN")).sum().alias("usable_not_clean"))
        .with_columns(pl.exclude("zm").cast(pl.Int64)))


def budget_stats(read_summary: pl.DataFrame, cls: pl.DataFrame, order_len: int = 9) -> dict:
    """summary.json["skera"]'s budget: means over every HiFi read in read_summary (reads without
    molecules count as 0) of budget_per_read's counts, plus unfilled_slots (order_len - 1 - molecules)
    and artifact (the sum of the four artifact_* counts, per read, then meaned)."""
    per_read = budget_per_read(cls)
    reads = (read_summary.select("zm").join(per_read, on="zm", how="left")
             .with_columns(pl.exclude("zm").fill_null(0))
             .with_columns((order_len - 1 - pl.col("molecules")).alias("unfilled_slots"),
                           (pl.col("artifact_tso_tso") + pl.col("artifact_r1_r1") + pl.col("artifact_one_primer_or_none")
                            + pl.col("artifact_fused")).alias("artifact")))
    cols = ["molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1", "artifact_one_primer_or_none",
            "artifact_fused", "artifact", "usable", "usable_clean", "usable_not_clean"]
    return {c: (float(reads[c].mean()) if reads.height else float("nan")) for c in cols}


def double_mirrors(molecules: pl.DataFrame) -> pl.Series:
    """Find the ZMWs of reads with exactly two mirror junctions, one TSO-TSO and one R1-R1, both the same letter."""
    j = mirror_junctions(molecules)
    per = j.group_by("zm").agg(
        pl.len().alias("nj"),
        (pl.col("end_class") == "tso_tso").sum().alias("tt"),
        (pl.col("end_class") == "r1_r1").sum().alias("rr"),
        pl.col("letter").filter(pl.col("end_class") == "tso_tso").first().alias("tt_letter"),
        pl.col("letter").filter(pl.col("end_class") == "r1_r1").first().alias("rr_letter"))
    return (per.filter((pl.col("nj") == 2) & (pl.col("tt") == 1) & (pl.col("rr") == 1) & (pl.col("tt_letter") == pl.col("rr_letter")))
            .sort("zm")["zm"])
