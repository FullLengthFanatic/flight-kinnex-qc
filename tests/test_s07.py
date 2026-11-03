"""Tests for study/pipeline/s07_depletion.py: aggregation functions on toy frames, and end to end on the mini dataset."""
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from kinnex_qc import metrics
from pipeline_helpers import key_structure, mini_samples, run_cross_sample

import s07_depletion as q2

GOLDEN = Path(__file__).parent / "golden" / "v10_inputs" / "hash_0.18"
SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")
HEADER = "sample\tcell\tdepletion\thifi_bam\tfail_bam\tsegmented_bam\tnon_passing_bam\tzmw_metrics\tskera_report\tfltnc_bam\n"


def seg(zm, start, end, left, right, cls, group="complete", internal=False, pos=None, ttl=None):
    return {"zm": zm, "start": start, "end": end, "left_token": left, "right_token": right, "pos": pos,
            "sread_len": end - start, "end_class": cls, "tso_toward_lower": ttl, "internal": internal,
            "polyT_len": 25, "polyT_anchored": True, "insert_len": 100, "group": group, "primary_pattern": "complete"}


def test_mirror_junctions():
    s = pl.DataFrame([seg(1, 0, 100, "C", "D", "proper", pos=2), seg(1, 100, 200, "D", "D'", "tso_tso"),
                      seg(1, 200, 300, "D'", "C'", "proper", pos=2), seg(2, 0, 90, None, "A", "r1_only"),
                      seg(3, 0, 80, "G'", "G", "r1_r1")])
    cls, letters = q2.mirror_junctions(s)
    j = cls.filter(pl.col("where") == "mirror junction")
    assert dict(zip(j["end_class"], j["len"])) == {"tso_tso": 1, "r1_r1": 1}
    assert letters.to_dicts() == [{"adapter": "D", "junctions": 1}, {"adapter": "G", "junctions": 1}]
    assert cls.filter(pl.col("where") == "array end")["end_class"].to_list() == ["r1_only"]


def test_mirror_junctions_row_order():
    """Rows also come sorted by internal, whatever order group_by returns them in."""
    rows = [seg(1, 0, 100, "C", "D", "proper", internal=True), seg(2, 0, 100, "C", "D", "proper", internal=False)] * 3
    for _ in range(5):
        cls, _ = q2.mirror_junctions(pl.DataFrame(rows).sample(fraction=1.0, shuffle=True))
        j = cls.filter(pl.col("where") == "all segments")
        assert j.select("internal", "len").rows() == [(False, 3), (True, 3)]


def test_array_sizes():
    summary = pl.DataFrame({"n_mas": [9, 9, 5, 1], "skera_passing": [True, True, True, False]})
    # zm 4's read is skera-rejected but still holds 2 skera segments.
    reads = pl.DataFrame({"n_segments": [8, 8, 3, 2], "passing": [True, True, True, False]})
    t = q2.array_sizes(summary, reads)
    d = {r["size"]: r for r in t.to_dicts()}
    assert d[8] == {"size": 8, "ours_skera_passing": 2, "ours_skera_rejected": 0, "skera_passing": 2, "skera_rejected": 0}
    assert d[4] == {"size": 4, "ours_skera_passing": 1, "ours_skera_rejected": 0, "skera_passing": 0, "skera_rejected": 0}
    assert d[3] == {"size": 3, "ours_skera_passing": 0, "ours_skera_rejected": 0, "skera_passing": 1, "skera_rejected": 0}
    assert d[2] == {"size": 2, "ours_skera_passing": 0, "ours_skera_rejected": 0, "skera_passing": 0, "skera_rejected": 1}
    assert d[0] == {"size": 0, "ours_skera_passing": 0, "ours_skera_rejected": 1, "skera_passing": 0, "skera_rejected": 0}


def test_orientation_and_rates():
    s = pl.DataFrame([seg(1, 0, 100, "C", "D", "proper", pos=2, ttl=False), seg(1, 100, 200, "D", "E", "proper", pos=3, ttl=False),
                      seg(2, 0, 100, "C", "D", "tso_tso", group="skera rejected"), seg(2, 100, 200, "D", "E", "tso_tso", internal=True)])
    o = q2.orientation(s)
    assert o["n"] == 2 and o["tso_toward_lower"] == 0.0 and "higher-index" in o["prediction"]
    r = q2.artifact_rates(s).filter(pl.col("group") == "all").row(0, named=True)
    assert r["tso_tso_per_1000"] == 250.0          # the internal one is not counted as TSO-TSO
    assert r["internal_per_1000"] == 250.0


def test_artifact_rates_by_group_come_from_the_metric():
    """One row for all segments, then one per read group, each with the segment count and metrics.artifact_rates."""
    s = pl.DataFrame([seg(1, 0, 100, "C", "D", "proper"), seg(2, 0, 100, "C", "D", "tso_tso", group="skera rejected"),
                      seg(3, 0, 100, "C", "D", "r1_only", group="skera rejected", internal=True)])
    t = q2.artifact_rates(s)
    assert t["group"].to_list() == ["all", *q2.GROUPS]
    assert t.columns == ["group", "segments", *metrics.artifact_rates(s)]
    for row, frame in zip(t.iter_rows(named=True), [s, *(s.filter(pl.col("group") == g) for g in q2.GROUPS)]):
        assert row == {"group": row["group"], "segments": frame.height, **metrics.artifact_rates(frame)}


def test_skera_recovery():
    ours = pl.DataFrame([seg(1, 18, 1000, "A", "B", "proper"), seg(1, 1017, 3000, "B", "C", "proper"),
                         seg(1, 3017, 5000, "C", "D", "proper")])
    skera = pl.DataFrame({"zm": [1, 1], "qs": [20, 1017], "qe": [1002, 4200]})
    r = q2.skera_recovery(ours, skera).row(0, named=True)
    assert (r["our_molecules"], r["skera_recovered"], r["not_recovered"]) == (3, 1, 2)


def test_skera_recovery_bases_past_int32():
    """sread_len (int32 on disk) summed past 2**31 must not wrap negative."""
    big = 1_000_000_000
    ours = pl.DataFrame([seg(z, 0, big, "A", "B", "proper") for z in (1, 2, 3)]).with_columns(pl.col("sread_len").cast(pl.Int32))
    skera = pl.DataFrame({"zm": [], "qs": [], "qe": []}, schema={"zm": pl.Int64, "qs": pl.Int64, "qe": pl.Int64})
    r = q2.skera_recovery(ours, skera).row(0, named=True)
    assert r["bases_not_recovered"] == 3 * big


def test_unresolved_arrays():
    # zm 4 is skera-rejected but still holds 2 skera segments: not counted in reads_without_skera_sread.
    summary = pl.DataFrame({"zm": [1, 2, 3, 4],
                            "group": ["complete", "skera passing, not complete", "skera rejected", "skera rejected"]})
    reads = pl.DataFrame({"zm": [1, 2, 3, 4], "read_len": [120, 80, 60, 50], "unsegmented_bp": [20, 20, 60, 10],
                          "n_segments": [8, 3, 0, 2]})
    rec = pl.DataFrame({"group": ["complete", "skera passing, not complete", "skera rejected"],
                        "not_recovered": [1, 2, 3], "bases_not_recovered": [10, 20, 30]})
    t = q2.unresolved_arrays(summary, reads, rec)
    p = t.filter(pl.col("verdict") == "skera passing").row(0, named=True)
    assert (p["reads"], p["read_bases"], p["unsegmented_bases"], p["reads_without_skera_sread"]) == (2, 200, 40, 0)
    assert abs(p["unsegmented_base_share"] - 40 / 310) < 1e-9
    assert (p["proper_not_recovered"], p["proper_not_recovered_bases"]) == (3, 30)
    j = t.filter(pl.col("verdict") == "skera rejected").row(0, named=True)
    assert (j["reads"], j["read_bases"], j["unsegmented_bases"], j["reads_without_skera_sread"]) == (2, 110, 70, 1)
    assert abs(j["unsegmented_base_share"] - 70 / 310) < 1e-9
    assert (j["proper_not_recovered"], j["proper_not_recovered_bases"]) == (3, 30)


def test_unresolved_arrays_bases_past_int32():
    """read_len/unsegmented_bp (int32 on disk) summed past 2**31 must not wrap negative."""
    big = 1_000_000_000
    summary = pl.DataFrame({"zm": [1, 2, 3], "group": ["complete"] * 3})
    reads = pl.DataFrame({"zm": [1, 2, 3], "read_len": [big] * 3, "unsegmented_bp": [big] * 3, "n_segments": [8] * 3},
                         schema_overrides={"read_len": pl.Int32, "unsegmented_bp": pl.Int32})
    rec = pl.DataFrame({"group": ["complete"], "not_recovered": [0], "bases_not_recovered": [0]})
    p = q2.unresolved_arrays(summary, reads, rec).filter(pl.col("verdict") == "skera passing").row(0, named=True)
    assert (p["read_bases"], p["unsegmented_bases"]) == (3 * big, 3 * big)
    assert p["unsegmented_base_share"] == pytest.approx(1.0)


def test_class_composition_row_order():
    """Rows come sorted by group, end class and internal flag, whatever order group_by returns them in."""
    rows = [seg(z, 0, 100, "A", "B", c, group=g, internal=i) for z, (g, c, i) in enumerate(
        [("skera rejected", "tso_tso", False), ("complete", "proper", True), ("complete", "proper", False),
         ("skera passing, not complete", "r1_only", False), ("complete", "none", False)] * 3)]
    for _ in range(5):
        t = q2.class_composition(pl.DataFrame(rows).sample(fraction=1.0, shuffle=True))
        assert t.select("group", "end_class", "internal").rows() == [
            ("complete", "none", False), ("complete", "proper", False), ("complete", "proper", True),
            ("skera passing, not complete", "r1_only", False), ("skera rejected", "tso_tso", False)]


def test_class_composition_bases_past_int32():
    """sread_len (int32 on disk) summed past 2**31 must not wrap negative."""
    big = 1_000_000_000
    segs = pl.DataFrame([seg(z, 0, big, "A", "B", "proper") for z in (1, 2, 3)]).with_columns(pl.col("sread_len").cast(pl.Int32))
    row = q2.class_composition(segs).row(0, named=True)
    assert row["bases"] == 3 * big


def test_yield_and_polyt():
    s = pl.DataFrame([seg(1, 0, 1000, "A", "B", "proper"), seg(1, 1000, 2000, "B", "C", "proper", internal=True),
                      seg(2, 0, 500, "A", "B", "tso_tso")])
    fate = pl.DataFrame({"zm": [1, 1], "di": [0, 1], "sread_len": [1000, 1000], "mapped_fltnc": [True, False],
                         "refine_asrun": [True, True], "lima_pass": [True, True],
                         **{f"refine_polyA_{c}": [False, False] for c in (5, 8, 12, 20)}})
    y = q2.yield_table(s, fate, n_hifi=2, n_ccs=4, n_productive=5, hash_sample=False)
    assert y["molecules"].to_list() == [1, 1, 1, 2]
    assert y["per_hifi_zmw"][0] == 0.5
    assert y["per_productive_zmw"][0] == 0.2
    assert y["per_ccs_zmw"][0] == 0.25
    pr = q2.polyt_retention(s, fate)
    assert pr["ours_all"].to_list() == [1.0, 1.0, 1.0, 1.0] and pr["refine_require_polyA"].to_list() == [0.0] * 4


def test_polyt_retention_empty_groups_give_nan_not_a_crash():
    """No complete proper segment at all, and one where none is polyT-anchored: the mean of an empty column
    is None, and float(None) would raise."""
    s = pl.DataFrame([seg(1, 0, 1000, "A", "B", "tso_tso")])   # no proper segment in a complete array
    fate = pl.DataFrame({"zm": [], "di": [], "sread_len": [], "mapped_fltnc": [], "refine_asrun": [], "lima_pass": [],
                         **{f"refine_polyA_{c}": [] for c in (5, 8, 12, 20)}}, schema={"zm": pl.Int64, "di": pl.Int64,
                         "sread_len": pl.Int64, "mapped_fltnc": pl.Boolean, "refine_asrun": pl.Boolean,
                         "lima_pass": pl.Boolean, **{f"refine_polyA_{c}": pl.Boolean for c in (5, 8, 12, 20)}})
    pr = q2.polyt_retention(s, fate)
    assert all(v != v for v in pr["ours_all"].to_list())              # NaN, not a crash
    assert all(v != v for v in pr["refine_require_polyA"].to_list())

    s2 = pl.DataFrame([{**seg(2, 0, 1000, "A", "B", "proper"), "polyT_anchored": False}])
    pr2 = q2.polyt_retention(s2, fate)
    assert all(v != v for v in pr2["ours_anchored"].to_list())        # no anchored polyT at all: NaN, not a crash


def test_yield_table_bases_past_int32():
    """sread_len (int32 on disk) summed past 2**31 must not wrap negative, in every row's bases total."""
    big = 1_000_000_000
    s = pl.DataFrame([seg(z, 0, big, "A", "B", "proper") for z in (1, 2, 3)]).with_columns(pl.col("sread_len").cast(pl.Int32))
    fate = pl.DataFrame({"zm": [1, 2, 3], "di": [0, 0, 0], "sread_len": [big] * 3, "mapped_fltnc": [True] * 3,
                         "refine_asrun": [True] * 3, "lima_pass": [True] * 3,
                         **{f"refine_polyA_{c}": [False] * 3 for c in (5, 8, 12, 20)}},
                        schema_overrides={"sread_len": pl.Int32})
    y = q2.yield_table(s, fate, n_hifi=3, n_ccs=3, n_productive=3, hash_sample=False)
    assert y["bases"].to_list() == [3 * big] * 4


def test_yield_table_per_ccs_zmw_null_in_a_hash_sample():
    """A hash sample indexes part of the HiFi ZMWs but every fail read, so per_ccs_zmw is not comparable with a full run."""
    s = pl.DataFrame([seg(1, 0, 1000, "A", "B", "proper")])
    fate = pl.DataFrame({"zm": [1], "di": [0], "sread_len": [1000], "mapped_fltnc": [True],
                         "refine_asrun": [True], "lima_pass": [True],
                         **{f"refine_polyA_{c}": [False] for c in (5, 8, 12, 20)}})
    y = q2.yield_table(s, fate, n_hifi=1, n_ccs=2, n_productive=1, hash_sample=True)
    assert y["per_ccs_zmw"].to_list() == [None] * 4
    assert q2.yield_table(s, fate, n_hifi=1, n_ccs=2, n_productive=1, hash_sample=False)["per_ccs_zmw"][0] == 0.5


def write_samples(path, samples):
    path.write_text(HEADER + "".join("\t".join([s, "c", "none"] + ["x"] * 7) + "\n" for s in samples))


def test_cross_sample_comparisons(tmp_path):
    """ND vs kin and HM vs kin only, on rates and shares: raw counts are left out of the comparisons."""
    samples = ("BD144_ND", "BD144_HM", "BD144_kin")
    write_samples(tmp_path / "samples.tsv", samples)
    for i, s in enumerate(samples):
        d = tmp_path / "out" / "hash_0.18" / s / "q2"
        d.mkdir(parents=True)
        d.joinpath("q2_summary.json").write_text(json.dumps({
            "sample": s, "hifi_reads": 1000 + i, "productive_zmws": 2000 + i, "hifi_reads_total": 10_000 + i,
            "zmw_metrics_hifi_flag": 1500 + i, "mirror_junctions": 30 + i, "rejected_sreads": 40 + i,
            "mirror_fraction": 0.1 + i / 100, "tso_tso_per_1000": 5.0 + i, "usable_per_hifi_zmw": 6.0 + i}))
    q2.cross_sample(SimpleNamespace(samples=tmp_path / "samples.tsv", out_root=tmp_path / "out", fraction=0.18))
    tables = tmp_path / "out" / "hash_0.18" / "all" / "q2" / "tables"
    t = pl.read_csv(tables / "comparisons.tsv", separator="\t")
    assert set(t["comparison"]) == {"BD144_ND vs BD144_kin", "BD144_HM vs BD144_kin"}
    assert set(t["metric"]) == {"mirror_fraction", "tso_tso_per_1000", "usable_per_hifi_zmw"}
    assert pl.read_csv(tables / "summary.tsv", separator="\t")["sample"].to_list() == list(samples)
    assert not (tables.parent / "manifest.json").exists() and not (tables.parent / "figures").exists()


def test_cross_sample_no_comparisons_table_when_only_kin_is_present(tmp_path):
    """BD144_kin with neither ND nor HM: comps is empty, so no empty comparisons.tsv is written."""
    write_samples(tmp_path / "samples.tsv", ("BD144_kin",))
    d = tmp_path / "out" / "BD144_kin" / "q2"
    d.mkdir(parents=True)
    d.joinpath("q2_summary.json").write_text(json.dumps({"sample": "BD144_kin", "hifi_reads": 1000,
                                                          "mirror_fraction": 0.1, "usable_per_hifi_zmw": 6.0}))
    q2.cross_sample(SimpleNamespace(samples=tmp_path / "samples.tsv", out_root=tmp_path / "out", fraction=1))
    tables = tmp_path / "out" / "all" / "q2" / "tables"
    assert (tables / "summary.tsv").exists()
    assert not (tables / "comparisons.tsv").exists()


@pytest.mark.real
def test_q2_mini_nd(mini_study, mini):
    q = mini_study / "BD144_ND" / "q2"
    tables = {p.stem for p in (q / "tables").glob("*.tsv")}
    assert tables == {"array_sizes", "chain_patterns", "mirror_junction_classes", "mirror_junction_letters", "orientation",
                      "skera_recovery", "unresolved_arrays", "class_composition", "artifact_rates", "class_lengths",
                      "class_positions", "lima_vs_class", "yield", "polyt_retention", "polyt_vs_refine"}
    assert not (q / "figures").exists() and not (q / "manifest.json").exists()
    key = json.loads((q / "q2_summary.json").read_text())
    assert key["mirror_fraction"] > 0.05                      # ND: mirror arrays from TSO-TSO inserts
    assert key["mirror_junction_tso_tso_share"] > 0.8
    assert key["tso_tso_per_1000"] > 5
    letters = pl.read_csv(q / "tables" / "mirror_junction_letters.tsv", separator="\t")
    assert "A" not in letters["adapter"].to_list()           # TSO faces the higher-index adapter
    # A productive ZMW is a zmw_metrics row with read_length > 0, over the whole cell (not the hifi_index zm range);
    # this run's HiFi reads are a fraction f of the cell's (the skera report's read_segmentation.reads), and the
    # productive-ZMW denominator is scaled by f to match.
    assert key["hifi_reads_total"] == 10_945_011              # the whole-cell report, copied into the mini as it is
    assert key["productive_zmws"] > 0
    assert key["zmw_metrics_hifi_flag"] >= 0
    f = key["hifi_reads"] / key["hifi_reads_total"]
    assert key["hifi_per_productive_zmw"] == pytest.approx(key["hifi_reads_total"] / key["productive_zmws"])
    assert key["usable_per_productive_zmw"] == pytest.approx(key["usable_per_hifi_zmw"] * key["hifi_reads"] / (key["productive_zmws"] * f))
    assert "rejected_read_share" in key and "rejected_read_base_share" in key
    assert "proper_not_recovered_in_rejected_per_hifi_zmw" in key
    assert key["skera_agreement"] == json.loads((GOLDEN / "BD144_ND" / "gates" / "gate_skera.json").read_text())["metrics"]["agreement"]
    assert pl.read_csv(q / "tables" / "yield.tsv", separator="\t")["per_ccs_zmw"].null_count() == 0      # fraction 1


@pytest.mark.real
def test_skera_report_reads_of_the_mini_is_the_whole_cell(mini):
    """The mini carries the whole cell's skera report: its read count is v10's hifi_reads_total."""
    for s in SAMPLES:
        v10 = json.loads((GOLDEN / s / "q2" / "q2_summary.json").read_text())["hifi_reads_total"]
        assert q2.skera_report_reads(mini_samples(mini)[s]) == v10


@pytest.mark.real
@pytest.mark.parametrize("sample", SAMPLES)
def test_summary_keys_match_v10(mini_study, sample):
    """q2_summary.json has exactly the keys, in the same order, of the frozen v10 summary."""
    new = json.loads((mini_study / sample / "q2" / "q2_summary.json").read_text())
    v10 = json.loads((GOLDEN / sample / "q2" / "q2_summary.json").read_text())
    assert key_structure(new) == key_structure(v10)


@pytest.mark.real
def test_cross_sample_mini(mini_study, mini):
    run_cross_sample("s07_depletion.py", mini_study, mini / "samples_mini.tsv", "--fraction", "1")
    tables = mini_study / "all" / "q2" / "tables"
    t = pl.read_csv(tables / "summary.tsv", separator="\t")
    assert t["sample"].to_list() == list(SAMPLES)
    comps = pl.read_csv(tables / "comparisons.tsv", separator="\t")
    assert set(comps["comparison"]) == {"BD144_ND vs BD144_kin", "BD144_HM vs BD144_kin"}
    assert not set(comps["metric"]) & q2.RAW_COUNT_KEYS
