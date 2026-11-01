"""Shared rules for skera's S-reads: false cuts, adapters left inside an S-read, molecule coverage, and mirror junctions."""
import polars as pl

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
        pairs = (mc.join(sc, on="zm")
                 .with_columns(((pl.min_horizontal("end", "qe") - pl.max_horizontal("start", "qs")).clip(0)
                                / (pl.col("end") - pl.col("start"))).alias("ov")))
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
