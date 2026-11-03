#!/usr/bin/env python3
"""Stage 07: what does TSO-TSO depletion change? (on our segmentation)

Per sample (--sample): array completeness, artifacts, fate and yield, written to <sample>/q2/tables/*.tsv and
<sample>/q2/q2_summary.json. Cross-sample (--all): every sample's q2_summary.json as one table, and ND vs kin (cost
of no depletion) and HM vs kin (home-made vs kit), written to all/q2/tables/.
Usage, from the repository root:
  study/pipeline/s07_depletion.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--processes N]
  study/pipeline/s07_depletion.py --all [--samples PATH] [--out-root DIR] [--fraction F]
Defaults come from study/study.toml.
"""
import argparse
import json

import polars as pl

from config import SAMPLE_ORDER, load_config, load_samples, out_dir, skera_report_reads
from kinnex_qc import metrics
from kinnex_qc.patterns import POLYT_REPORT_CUTOFFS
from kinnex_qc.skera import mirror_junctions as junction_molecules

GROUPS = ["complete", "skera passing, not complete", "skera rejected"]
MATCH_TOL = 30          # bp: our segment and a skera S-read are the same molecule if both ends agree within this
POLYT_BINS = [0, 1, 5, 8, 12, 20, 30, 50, 10_000]


def write_table(q, name: str, df: pl.DataFrame) -> None:
    df.write_csv(q / "tables" / f"{name}.tsv", separator="\t")


def load(out) -> tuple[pl.DataFrame, pl.DataFrame]:
    summary = pl.read_parquet(out / "read_summary.parquet").filter(pl.col("source") == "hifi")
    summary = summary.with_columns(
        pl.when((pl.col("primary_pattern") == "complete") & pl.col("skera_passing")).then(pl.lit(GROUPS[0]))
        .when(pl.col("skera_passing")).then(pl.lit(GROUPS[1])).otherwise(pl.lit(GROUPS[2])).alias("group"))
    segs = (pl.scan_parquet(out / "annot_segments.parquet").filter(pl.col("source") == "hifi")
            .select("zm", "start", "end", "left_token", "right_token", "pos", "sread_len", "end_class", "tso_toward_lower",
                    "internal", "polyT_len", "polyT_anchored", "insert_len").collect()
            .join(summary.select("zm", "group", "primary_pattern"), on="zm"))
    return summary, segs


# ---------- (a) array completeness ----------

def array_sizes(summary: pl.DataFrame, reads: pl.DataFrame) -> pl.DataFrame:
    """Array size distribution per sample, ours and skera's, each split by skera's verdict (a skera-rejected read can
    still hold skera S-reads: observed in the non-passing BAMs). The last class (9) is 9 or more segments."""
    ours = summary.with_columns((pl.col("n_mas").cast(pl.Int64) - 1).clip(0, 9).alias("size"))
    op = ours.filter(pl.col("skera_passing")).group_by("size").len().rename({"len": "ours_skera_passing"})
    orj = ours.filter(~pl.col("skera_passing")).group_by("size").len().rename({"len": "ours_skera_rejected"})
    sk = reads.select(pl.col("n_segments").cast(pl.Int64).clip(0, 9).alias("size"), "passing")
    skp = sk.filter(pl.col("passing")).group_by("size").len().rename({"len": "skera_passing"})
    skr = sk.filter(~pl.col("passing")).group_by("size").len().rename({"len": "skera_rejected"})
    return (op.join(orj, on="size", how="full", coalesce=True).join(skp, on="size", how="full", coalesce=True)
            .join(skr, on="size", how="full", coalesce=True).fill_null(0).sort("size"))


def pattern_table(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.group_by("primary_pattern", "skera_passing").len().sort("primary_pattern", "skera_passing")


def mirror_junctions(segs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Our segments between X and X' (a mirror junction, kinnex_qc.skera.mirror_junctions): their class, and the
    adapter letter. Also the class of segments at array ends (one side without an adapter) and of all segments."""
    junction = junction_molecules(segs)
    ends = segs.filter(pl.col("left_token").is_null() | pl.col("right_token").is_null())
    cls = pl.concat([
        junction.group_by("end_class", "internal").len().with_columns(pl.lit("mirror junction").alias("where")),
        ends.group_by("end_class", "internal").len().with_columns(pl.lit("array end").alias("where")),
        segs.group_by("end_class", "internal").len().with_columns(pl.lit("all segments").alias("where"))])
    letters = junction.group_by("letter").len().rename({"letter": "adapter", "len": "junctions"}).sort("adapter")
    return cls.sort("where", "end_class", "internal"), letters


def orientation(segs: pl.DataFrame) -> dict:
    """Among proper segments of complete arrays, share whose TSO end faces the lower-index adapter."""
    p = segs.filter((pl.col("group") == GROUPS[0]) & (pl.col("end_class") == "proper") & pl.col("tso_toward_lower").is_not_null())
    frac = float(p["tso_toward_lower"].mean()) if p.height else float("nan")
    side = "lower" if frac > 0.5 else "higher"
    return {"n": p.height, "tso_toward_lower": frac,
            "prediction": f"a TSO-TSO insert amplified in position PCR i carries the {side}-index adapter of that position on both ends"}


def skera_recovery(segs: pl.DataFrame, skera: pl.DataFrame) -> pl.DataFrame:
    """For our proper, non-internal molecules: does skera have an S-read with the same two ends (within MATCH_TOL)?
    Molecules skera did not recover have no matching S-read: split to different boundaries, or absent
    altogether. A skera-rejected read can still hold S-reads (observed), so "rejected" does not mean unsegmented."""
    ours = segs.filter((pl.col("end_class") == "proper") & ~pl.col("internal")).select("zm", "start", "end", "sread_len", "group")
    sk = skera.select("zm", pl.col("qs").alias("start"), pl.col("qe").alias("sk_end")).with_columns(pl.col("start").alias("sk_start"))
    j = ours.sort("zm", "start").join_asof(sk.sort("zm", "start"), on="start", by="zm", strategy="nearest")
    j = j.with_columns((((pl.col("sk_start") - pl.col("start")).abs() <= MATCH_TOL) &
                        ((pl.col("sk_end") - pl.col("end")).abs() <= MATCH_TOL)).fill_null(False).alias("recovered"))
    return (j.group_by("group").agg(pl.len().alias("our_molecules"), pl.col("recovered").sum().alias("skera_recovered"),
                                    pl.col("sread_len").filter(~pl.col("recovered")).cast(pl.Int64).sum().alias("bases_not_recovered"))
            .with_columns((pl.col("our_molecules") - pl.col("skera_recovered")).alias("not_recovered")).sort("group"))


def unresolved_arrays(summary: pl.DataFrame, reads: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Two separate signals of loss, not a sum. unsegmented_bases is the unsegmented
    part of non-passing reads; proper_not_recovered_bases is the proper molecules skera did not recover
    anywhere (rec). A not-recovered molecule that lies outside every skera S-read is counted in
    unsegmented_bases too, so adding the two columns counts its bases twice. A skera-rejected read can still
    hold skera S-reads (observed), so unsegmented_bases is not every base in those reads, only the unsegmented
    part. One row per skera verdict, grouping rec's "skera rejected" group as rejected and its other two
    groups as passing."""
    r = reads.join(summary.select("zm", "group"), on="zm").with_columns(
        pl.when(pl.col("group") == GROUPS[2]).then(pl.lit("skera rejected")).otherwise(pl.lit("skera passing")).alias("verdict"),
        pl.col("read_len").cast(pl.Int64), pl.col("unsegmented_bp").cast(pl.Int64))
    total_bases = r["read_len"].sum()
    base = (r.group_by("verdict").agg(pl.len().alias("reads"), pl.col("read_len").sum().alias("read_bases"),
                                      pl.col("unsegmented_bp").sum().alias("unsegmented_bases"),
                                      (pl.col("n_segments") == 0).sum().alias("reads_without_skera_sread"))
            .with_columns((pl.col("unsegmented_bases") / max(total_bases, 1)).alias("unsegmented_base_share")))
    rc = (rec.with_columns(pl.when(pl.col("group") == GROUPS[2]).then(pl.lit("skera rejected"))
                            .otherwise(pl.lit("skera passing")).alias("verdict"))
          .group_by("verdict").agg(pl.col("not_recovered").sum().alias("proper_not_recovered"),
                                   pl.col("bases_not_recovered").sum().alias("proper_not_recovered_bases")))
    return base.join(rc, on="verdict", how="left").fill_null(0).sort("verdict")


# ---------- (b) artifacts ----------

def class_composition(segs: pl.DataFrame) -> pl.DataFrame:
    return (segs.group_by("group", "end_class", "internal").agg(pl.len().alias("segments"),
                                                                 pl.col("sread_len").cast(pl.Int64).sum().alias("bases"))
            .sort("group", "end_class", "internal"))


def artifact_rates(segs: pl.DataFrame) -> pl.DataFrame:
    """Per 1,000 of our segments (metrics.artifact_rates), overall and by read group; every end class counts
    non-internal segments only, with internal segments counted separately."""
    groups = [("all", segs)] + [(g, segs.filter(pl.col("group") == g)) for g in GROUPS]
    return pl.DataFrame([{"group": name, "segments": d.height, **metrics.artifact_rates(d)} for name, d in groups])


def class_lengths(segs: pl.DataFrame) -> pl.DataFrame:
    """Median and 10th/90th percentile of segment length by class, non-internal segments only."""
    return (segs.filter(~pl.col("internal")).group_by("end_class")
            .agg(pl.len().alias("n"), pl.col("sread_len").median().alias("median"),
                pl.col("sread_len").quantile(0.1).alias("p10"), pl.col("sread_len").quantile(0.9).alias("p90"))
            .sort("end_class"))


def class_positions(segs: pl.DataFrame) -> pl.DataFrame:
    return segs.filter(pl.col("pos").is_not_null()).group_by("pos", "end_class").len().sort("pos", "end_class")


def lima_vs_class(fate: pl.DataFrame) -> pl.DataFrame:
    return fate.group_by("end_class", "lima_pair").len().sort("end_class", "len", descending=[False, True])


# ---------- (c) fate and yield ----------

def yield_table(segs: pl.DataFrame, fate: pl.DataFrame, n_hifi: int, n_ccs: int, n_productive: float, hash_sample: bool) -> pl.DataFrame:
    """Usable molecules per HiFi ZMW, per ZMW with any CCS read (HiFi or fail), and per productive ZMW. A hash sample
    indexes part of the HiFi ZMWs but the whole fail BAM (stage 02), so its per-CCS-ZMW denominator is not comparable
    with a full run; per_ccs_zmw is null in that case."""
    usable = segs.filter((pl.col("end_class") == "proper") & ~pl.col("internal"))
    rows = [("our proper molecules, all HiFi reads", usable.height, usable["sread_len"].cast(pl.Int64).sum()),
            ("our proper molecules, complete arrays", usable.filter(pl.col("group") == GROUPS[0]).height,
             usable.filter(pl.col("group") == GROUPS[0])["sread_len"].cast(pl.Int64).sum()),
            ("mapped fltnc (existing pipeline)", int(fate["mapped_fltnc"].sum()),
             fate.filter(pl.col("mapped_fltnc"))["sread_len"].cast(pl.Int64).sum()),
            ("refine as run", int(fate["refine_asrun"].sum()), fate.filter(pl.col("refine_asrun"))["sread_len"].cast(pl.Int64).sum())]
    return pl.DataFrame([{"what": w, "molecules": int(n), "bases": int(b or 0), "per_hifi_zmw": n / max(n_hifi, 1),
                          "per_ccs_zmw": None if hash_sample else n / max(n_ccs, 1), "per_productive_zmw": n / max(n_productive, 1),
                          "bases_per_hifi_zmw": (b or 0) / max(n_hifi, 1),
                          "bases_per_productive_zmw": (b or 0) / max(n_productive, 1)} for w, n, b in rows])


def polyt_retention(segs: pl.DataFrame, fate: pl.DataFrame) -> pl.DataFrame:
    """Share of proper molecules kept at each minimum polyT length: our anchored measurement and refine --require-polya."""
    p = segs.filter((pl.col("group") == GROUPS[0]) & (pl.col("end_class") == "proper"))
    anchored = p.filter(pl.col("polyT_anchored"))
    lp = fate.filter(pl.col("lima_pass"))
    rows = []
    for c in POLYT_REPORT_CUTOFFS:
        rows.append({"min_polyT": c, "ours_all": float((p["polyT_len"] >= c).mean()) if p.height else float("nan"),
                     "ours_anchored": float((anchored["polyT_len"] >= c).mean()) if anchored.height else float("nan"),
                     "refine_require_polyA": float(lp[f"refine_polyA_{c}"].mean()) if lp.height else float("nan")})
    return pl.DataFrame(rows)


def polyt_vs_refine(skera_segments: pl.DataFrame, fate: pl.DataFrame) -> pl.DataFrame:
    """Refine's polyA call against our anchored polyT length, on the same skera S-reads (proper, lima pass)."""
    j = (skera_segments.filter((pl.col("end_class") == "proper") & pl.col("polyT_anchored"))
         .select("zm", "di", "polyT_len").join(fate.filter(pl.col("lima_pass")), on=["zm", "di"]))
    j = j.with_columns(pl.col("polyT_len").cut(POLYT_BINS[1:-1], left_closed=True).alias("polyT_bin"))
    return (j.group_by("polyT_bin").agg(pl.len().alias("n"), pl.col("refine_polyA_5").mean().alias("kept_min5"),
                                        pl.col("refine_polyA_20").mean().alias("kept_min20"), pl.col("polyT_len").min().alias("lo"))
            .sort("lo").drop("lo"))


def per_sample(args) -> None:
    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    q = out / "q2"
    (q / "tables").mkdir(parents=True, exist_ok=True)
    summary, segs = load(out)
    reads = pl.read_parquet(out / "reads.parquet")
    zmw = pl.read_parquet(out / "zmw.parquet", columns=["zm", "read_length", "hifi"])
    skera = pl.read_parquet(out / "segments.parquet", columns=["zm", "di", "qs", "qe", "end_class", "polyT_len", "polyT_anchored"])
    fate = pl.read_parquet(out / "fate.parquet")
    hifi_idx = pl.scan_parquet(out / "hifi_index.parquet").select("zm").collect()
    n_hifi = hifi_idx.height
    n_ccs = n_hifi + pl.scan_parquet(out / "fail_index.parquet").select(pl.len()).collect().item()
    # A productive ZMW is a zmw_metrics row with read_length > 0, counted over the whole cell, not just
    # the zm range this run's HiFi index happens to cover (observed: the HiFi BAM is not in ZMW order); a hash sample
    # only holds a fraction f of the cell's HiFi reads, so the productive-ZMW denominator is scaled by f to match.
    # The zmw_metrics hifi flag is not used as a HiFi count: it disagrees with the HiFi BAM (see
    # zmw_metrics_hifi_flag and hifi_per_productive_zmw).
    productive_zmws = zmw.filter(pl.col("read_length") > 0).height
    zmw_metrics_hifi_flag = int(zmw["hifi"].sum())
    hifi_reads_total = skera_report_reads(sample)
    f = n_hifi / hifi_reads_total
    n_productive = productive_zmws * f
    key = {"sample": sample.sample, "hifi_reads": n_hifi, "productive_zmws": productive_zmws,
           "hifi_reads_total": hifi_reads_total, "zmw_metrics_hifi_flag": zmw_metrics_hifi_flag,
           "hifi_per_productive_zmw": hifi_reads_total / productive_zmws if productive_zmws else None}

    # (a)
    write_table(q, "array_sizes", array_sizes(summary, reads))
    write_table(q, "chain_patterns", pattern_table(summary))
    key.update(metrics.read_fractions(summary))
    key["skera_rejected_fraction"] = float((~summary["skera_passing"]).mean())
    jc, letters = mirror_junctions(segs)
    write_table(q, "mirror_junction_classes", jc)
    write_table(q, "mirror_junction_letters", letters)
    key.update(metrics.mirror_junction_shares(segs))
    o = orientation(segs)
    key["tso_toward_lower"] = o["tso_toward_lower"]
    write_table(q, "orientation", pl.DataFrame([o]))
    rec = skera_recovery(segs, skera)
    write_table(q, "skera_recovery", rec)
    key["not_recovered_by_skera_per_hifi_zmw"] = float(rec["not_recovered"].sum() / max(n_hifi, 1))
    gate = out / "gates" / "gate_skera.json"
    if gate.exists():
        key["skera_agreement"] = json.loads(gate.read_text())["metrics"]["agreement"]
    ua = unresolved_arrays(summary, reads, rec)
    write_table(q, "unresolved_arrays", ua)
    rej = ua.filter(pl.col("verdict") == "skera rejected")
    rejected_row = rej.row(0, named=True) if rej.height else {"reads": 0, "read_bases": 0, "proper_not_recovered": 0}
    key["rejected_read_share"] = float(rejected_row["reads"] / max(n_hifi, 1))
    key["rejected_read_base_share"] = float(rejected_row["read_bases"] / max(ua["read_bases"].sum(), 1))
    key["proper_not_recovered_in_rejected_per_hifi_zmw"] = float(rejected_row["proper_not_recovered"] / max(n_hifi, 1))

    # (b)
    write_table(q, "class_composition", class_composition(segs))
    rates = artifact_rates(segs)
    write_table(q, "artifact_rates", rates)
    key.update({k: float(v) for k, v in rates.filter(pl.col("group") == "all").row(0, named=True).items() if k.endswith("_per_1000")})
    write_table(q, "class_lengths", class_lengths(segs))
    write_table(q, "class_positions", class_positions(segs))
    write_table(q, "lima_vs_class", lima_vs_class(fate))

    # (c)
    key["rejected_sreads"] = int((~fate["skera_passing"]).sum())
    rj = fate.filter(~pl.col("skera_passing"))
    key["rejected_sreads_lima_pass_share"] = float(rj["lima_pass"].mean()) if rj.height else float("nan")
    key["rejected_sreads_mapped_share"] = float(rj["mapped_fltnc"].mean()) if rj.height else float("nan")
    yt = yield_table(segs, fate, n_hifi, n_ccs, n_productive, args.fraction < 1)
    write_table(q, "yield", yt)
    key["usable_per_hifi_zmw"] = float(yt["per_hifi_zmw"][0])
    key["mapped_fltnc_per_hifi_zmw"] = float(yt["per_hifi_zmw"][2])
    key["usable_per_productive_zmw"] = float(yt["per_productive_zmw"][0])
    pr = polyt_retention(segs, fate)
    write_table(q, "polyt_retention", pr)
    key["polyT_ge_20"] = float(pr.filter(pl.col("min_polyT") == 20)["ours_all"].item())
    pv = polyt_vs_refine(pl.read_parquet(out / "segments.parquet", columns=["zm", "di", "end_class", "polyT_len", "polyT_anchored"]), fate)
    write_table(q, "polyt_vs_refine", pv)
    (q / "q2_summary.json").write_text(json.dumps(key, indent=1))


RAW_COUNT_KEYS = {"hifi_reads", "productive_zmws", "hifi_reads_total", "zmw_metrics_hifi_flag", "mirror_junctions",
                  "rejected_sreads"}


def cross_sample(args) -> None:
    """summary: one row per sample in SAMPLE_ORDER, its q2_summary.json. comparisons: ND vs kin (cost of skipping
    depletion) and HM vs kin (home-made vs kit) on every rate and share (raw counts left out); one library per arm, so
    differences include replicate variation. Written only when there is something to compare."""
    samples = [s for s in SAMPLE_ORDER if s in load_samples(args.samples)]
    rows = [json.loads((out_dir(args.out_root, s, args.fraction) / "q2" / "q2_summary.json").read_text()) for s in samples]
    t = pl.DataFrame(rows)
    q = out_dir(args.out_root, "all", args.fraction) / "q2"
    (q / "tables").mkdir(parents=True, exist_ok=True)
    write_table(q, "summary", t)
    comps = []
    ref = "BD144_kin"
    if ref in samples:
        r = t.filter(pl.col("sample") == ref).row(0, named=True)
        for other, what in (("BD144_ND", "cost of no depletion"), ("BD144_HM", "home-made vs kit")):
            if other in samples:
                o = t.filter(pl.col("sample") == other).row(0, named=True)
                for k, v in o.items():
                    if isinstance(v, (int, float)) and k not in RAW_COUNT_KEYS and isinstance(r.get(k), (int, float)):
                        comps.append({"comparison": f"{other} vs {ref}", "meaning": what, "metric": k, "value": v, "BD144_kin": r[k],
                                      "difference": v - r[k], "ratio": v / r[k] if r[k] else None})
        if comps:
            write_table(q, "comparisons", pl.DataFrame(comps, infer_schema_length=None))


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample")
    g.add_argument("--all", action="store_true", help="cross-sample tables from every sample's q2_summary.json")
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="names the output directory of a hash sample run")
    ap.add_argument("--processes", type=int, default=config["processes"], help="unused; accepted for a uniform CLI")
    args = ap.parse_args()
    cross_sample(args) if args.all else per_sample(args)


if __name__ == "__main__":
    main()
