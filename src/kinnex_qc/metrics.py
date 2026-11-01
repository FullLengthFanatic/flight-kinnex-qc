"""Numbers shared by the reader command and the study: pattern shares, artifact rates, mirror junctions."""
import polars as pl

from .classify import END_CLASSES
from .patterns import MAS_ORDER
from .skera import mirror_junctions

_OTHER_PATTERNS = {"skipped_slot": "other", "repeated": "other", "unresolved": "other"}


def pattern_shares(read_summary: pl.DataFrame) -> pl.DataFrame:
    """Share of reads by primary chain pattern, with skipped_slot, repeated and unresolved folded into "other"."""
    n = read_summary.height
    mapped = read_summary.with_columns(pl.col("primary_pattern").replace(_OTHER_PATTERNS).alias("pattern"))
    counts = mapped.group_by("pattern").agg(pl.len().alias("reads")).sort("pattern")
    # Plain Python division, not a polars column / scalar expression: polars' divide kernel can be
    # one ULP off a plain float division for some values (e.g. 1615591 / 1969703), and v10's own
    # script divided in plain Python (g.height / n).
    return counts.with_columns(pl.Series("share", [r / n for r in counts["reads"].to_list()]))


def artifact_rates(molecules: pl.DataFrame) -> dict:
    """Per 1,000 molecules: every end class's share (non-internal molecules only), then the internal share."""
    n = molecules.height
    rates = {f"{c}_per_1000": 1000 * molecules.filter((pl.col("end_class") == c) & ~pl.col("internal")).height / max(n, 1)
             for c in END_CLASSES}
    rates["internal_per_1000"] = 1000 * molecules["internal"].sum() / max(n, 1)
    return rates


def read_fractions(read_summary: pl.DataFrame) -> dict:
    """Share of reads whose primary chain pattern is complete, and whose pattern is mirror.
    NaN (no reads) when read_summary is empty: polars' mean() of an empty Series is None, not NaN,
    and float(None) raises, so the empty case is handled before the float() call."""
    pattern = read_summary["primary_pattern"]
    if pattern.len() == 0:
        return {"complete_fraction": float("nan"), "mirror_fraction": float("nan")}
    return {"complete_fraction": float((pattern == "complete").mean()), "mirror_fraction": float((pattern == "mirror").mean())}


def mirror_junction_shares(molecules: pl.DataFrame) -> dict:
    """Count of mirror junction molecules (internal included), and the share that are TSO-TSO or R1-R1."""
    j = mirror_junctions(molecules)
    n = j.height
    return {"mirror_junctions": n,
            "mirror_junction_tso_tso_share": j.filter(pl.col("end_class") == "tso_tso").height / max(n, 1),
            "mirror_junction_r1_r1_share": j.filter(pl.col("end_class") == "r1_r1").height / max(n, 1)}


def junction_table(molecules: pl.DataFrame) -> pl.DataFrame:
    """Mirror junction molecules by adapter letter and molecule class (TSO-TSO, R1-R1 or other)."""
    j = mirror_junctions(molecules).with_columns(
        pl.when(pl.col("end_class") == "tso_tso").then(pl.lit("TSO-TSO"))
        .when(pl.col("end_class") == "r1_r1").then(pl.lit("R1-R1")).otherwise(pl.lit("other")).alias("molecule"))
    return j.group_by("letter", "molecule").agg(pl.len().alias("junctions")).sort("letter", "molecule")


def junction_composition(molecules: pl.DataFrame, read_summary: pl.DataFrame) -> pl.DataFrame:
    """Per read with at least one mirror junction: what the junctions are made of (TSO-TSO, R1-R1, both or
    neither), joined to read_summary for array size and skera's verdict."""
    j = mirror_junctions(molecules)
    comp = j.group_by("zm").agg((pl.col("end_class") == "tso_tso").sum().alias("tt"), (pl.col("end_class") == "r1_r1").sum().alias("rr"))
    comp = comp.join(read_summary.select("zm", "n_mas", "skera_passing").with_columns(pl.col("n_mas").cast(pl.Int64)), on="zm")
    return (comp.with_columns(
        pl.when((pl.col("tt") > 0) & (pl.col("rr") > 0)).then(pl.lit("TSO-TSO and R1-R1"))
        .when(pl.col("tt") > 0).then(pl.lit("TSO-TSO only")).when(pl.col("rr") > 0).then(pl.lit("R1-R1 only"))
        .otherwise(pl.lit("other molecule")).alias("junction_molecules"))
        .group_by("junction_molecules").agg(pl.len().alias("reads"), (pl.col("n_mas") - 1).median().alias("median_molecules"),
                                            pl.col("skera_passing").mean().alias("skera_pass_share"))
        .sort("reads", descending=True))


def mirror_size_rule(molecules: pl.DataFrame, read_summary: pl.DataFrame, order: list = MAS_ORDER) -> pl.DataFrame:
    """For reads with exactly one mirror junction, a TSO-TSO one: array size (n_mas - 1) against the size that
    junction's adapter letter predicts for a clean two-arm mirror."""
    predicted = {letter: 2 * (len(order) - 1 - i) + 1 for i, letter in enumerate(order)}
    j = mirror_junctions(molecules)
    comp = j.group_by("zm").agg(pl.len().alias("junctions"), (pl.col("end_class") == "tso_tso").sum().alias("tt"),
                                pl.col("letter").first().alias("letter"))
    comp = comp.join(read_summary.select("zm", "n_mas").with_columns(pl.col("n_mas").cast(pl.Int64)), on="zm")
    single = (comp.filter((pl.col("junctions") == 1) & (pl.col("tt") == 1))
              .with_columns((pl.col("n_mas") - 1).alias("molecules"),
                            pl.col("letter").replace_strict(predicted, return_dtype=pl.Int64).alias("predicted")))
    return (single.group_by("letter").agg(pl.len().alias("reads"), pl.col("predicted").first(),
                                          pl.col("molecules").median().alias("median_molecules"),
                                          (pl.col("molecules") == pl.col("predicted")).mean().alias("share_matching"))
            .sort("letter"))
