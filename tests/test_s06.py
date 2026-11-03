"""Tests for study/pipeline/s06_concatenation_bias.py: null worker count and fork safety, and end to end on the mini dataset."""
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from pipeline_helpers import key_structure, run_cross_sample

import s06_concatenation_bias as q1

GOLDEN = Path(__file__).parent / "golden" / "v10_inputs" / "hash_0.18"
SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")
GB = 2**30


def test_null_workers_cap():
    # full run: 9.4M arrays at 700 bytes x 1.25 is 8.2 GB per worker
    assert q1.null_workers(7, 9_400_000, 20 * GB) == 2
    assert q1.null_workers(7, 9_400_000, 3 * GB) == 1           # never below 1
    assert q1.null_workers(7, 171_000, 80 * GB) == 7            # hash sample: the process count is the limit
    assert q1.null_workers(7, 1_000_000, 4 * 875_000_000) == 4  # exactly 4 workers fit
    assert q1.null_workers(7, 1_000_000, 4 * 875_000_000 - 1) == 3


def test_band_all_nan_column_is_quiet():
    """A column NaN in every replicate (no index segment landed in that bin) stays NaN, with no
    RuntimeWarning; a column with some data is unaffected."""
    reps = [{"v": np.array([1.0, np.nan, 10.0])}, {"v": np.array([3.0, np.nan, 20.0])},
            {"v": np.array([2.0, np.nan, 30.0])}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mean, lo, hi = q1.band(reps, "v")
    assert mean[0] == pytest.approx(2.0) and lo[0] <= mean[0] <= hi[0]
    assert np.isnan(mean[1]) and np.isnan(lo[1]) and np.isnan(hi[1])
    assert mean[2] == pytest.approx(20.0)


def test_mem_budget():
    assert q1.mem_budget_bytes(20) == 20 * GB
    with open("/proc/meminfo") as f:
        total = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemTotal:"))
    assert 0 < q1.mem_budget_bytes(None) <= 0.8 * total


class NoPolars:
    def __getattr__(self, name):
        raise AssertionError(f"a null worker called polars ({name})")


def test_null_replicates_do_not_call_polars(monkeypatch):
    """Null workers are forked after the parent used polars, so they must not call it."""
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(np.log(1500), 0.6, (400, 8)))
    eligible = rng.random((400, 8)) < 0.9
    shared = {"values": values, "eligible": eligible, "elig_frac": eligible.mean(axis=0),
              "model": q1.st.fit_mt(values, rng, n_sim=20_000)}
    monkeypatch.setitem(q1._G, "test", shared)
    monkeypatch.setattr(q1.st, "pl", NoPolars())
    monkeypatch.setattr(q1, "pl", NoPolars())
    for kind in ("P", "MT", "shuffle", "boot"):
        out = q1.one_rep((kind, "test", 1))
        if kind in ("shuffle", "boot"):
            assert out["lag"].shape == (7,)
        else:
            assert out["cut"].shape == (len(q1.CUTOFFS),)


def test_complete_arrays_pair_inserts_with_their_sread_lengths():
    """The observed IPW weights of insert_len arrays come from the S-read lengths of the same arrays, also
    when some arrays drop out of the insert matrix (a null insert_len)."""
    rows = [{"zm": zm, "pos": pos, "end_class": "proper", "internal": False, "sread_len": 1000 * zm + pos,
             "insert_len": None if (zm == 2 and pos == 4) else 100 * zm + pos} for zm in (1, 2, 3) for pos in range(8)]
    seg = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=3)
    values, _, zms, t = q1.complete_arrays(seg, "insert_len")
    assert zms.tolist() == [1, 3]
    assert t[:, 0].tolist() == [1000, 3000] and values[:, 0].tolist() == [100, 300]
    v, _, _, t = q1.complete_arrays(seg, "sread_len")
    assert t is v


def test_sampling_probabilities(tmp_path, monkeypatch):
    """HiFi reads have probability a = hifi_index rows / skera's cell total in a hash sample (1 at fraction 1, where the
    report is not read); fail_index reads fail_index rows / fail BAM records; annotated fail reads that over
    fail_sub_modulo."""
    pl.DataFrame({"zm": np.arange(200, dtype=np.uint32)}).write_parquet(tmp_path / "hifi_index.parquet")
    pl.DataFrame({"zm": np.arange(50, dtype=np.uint32)}).write_parquet(tmp_path / "fail_index.parquet")
    (tmp_path / "annotate.done.json").write_text(json.dumps({"fail": 50, "fail_bam_records": 50, "fail_sub_modulo": 10}))
    sample = SimpleNamespace(sample="toy")
    monkeypatch.setattr(q1, "skera_report_reads", lambda s: 1000)
    p = q1.sampling(tmp_path, sample, 0.2)
    assert (p["hifi"], p["fail"], p["fail_sub"]) == (0.2, 1.0, 0.1)
    assert (p["hifi_reads"], p["hifi_reads_total"], p["fail_reads"], p["fail_sub_modulo"]) == (200, 1000, 50, 10)
    assert p["hifi_note"] == "0.2000 (200 of skera's 1,000 HiFi reads of the cell, a hash sample, fraction 0.2)"
    monkeypatch.setattr(q1, "skera_report_reads", lambda s: pytest.fail("the report is read at fraction 1"))
    p = q1.sampling(tmp_path, sample, 1)
    assert p["hifi"] == 1.0 and p["hifi_reads_total"] is None
    assert p["hifi_note"] == "1 (a full run holds every HiFi read of the cell)"


@pytest.mark.real
def test_per_sample_tables(mini_study):
    q = mini_study / "BD144_ND" / "q1"
    tables = {p.stem for p in (q / "tables").glob("*.tsv")}
    for tag in ("sread", "insert"):
        for tid in ("partner_curve", "long_long", "lag", "positions", "positions_pairwise", "mt_fit", "partner_cutoffs", "ipw",
                    "ipw_long_long", "partial_curve", "truncation"):
            assert f"{tid}_{tag}" in tables
    assert "selection_T" in tables
    assert not (q / "figures").exists() and not (q / "manifest.json").exists()
    curve = pl.read_csv(q / "tables" / "partner_curve_sread.tsv", separator="\t")
    assert {"partner_mean", "P_mean_lo", "P_mean_hi", "MT_mean_lo", "MT_mean_hi"} <= set(curve.columns)
    summ = json.loads((q / "q1_summary.json").read_text())
    v = summ["values"][0]
    assert v["n_arrays"] > 1000 and v["D"]["MT_lo"] <= v["D"]["MT_hi"]
    assert v["longest_index_bp"] < q1.EDGES[-2]                  # the mini has no index segment of 10 kb or more
    # the sampling probabilities behind the w(T) weights: fraction 1, 1 in 10 fail reads annotated
    assert summ["sampling"]["hifi"] == 1.0 and summ["sampling"]["hifi_reads_total"] is None
    assert summ["sampling"]["fail"] == 1.0 and summ["sampling"]["fail_sub"] == pytest.approx(0.1)
    # every scaled curve has maximum 1; s(T) is empty exactly where no complete array has that length
    sel = pl.read_csv(q / "tables" / "selection_T.tsv", separator="\t")
    for col in ("w(T) empirical", "w(T) model", "s(T) from M_T"):
        if col in sel.columns:
            assert np.nanmax(sel[col].to_numpy()) == pytest.approx(1.0)
    s_t, n_arr = sel["s(T) from M_T"].to_numpy(), sel["complete_arrays"].to_numpy()
    assert np.isnan(s_t[n_arr == 0]).all() and not np.isnan(s_t[n_arr > 0]).any()
    share = sel["w_empirical_hifi_share"].to_numpy()
    assert np.allclose(sel["w(T) empirical"].to_numpy(), share / np.nanmax(share), equal_nan=True)
    # the weights reach w(T): with a = 1 and fail_sub = 0.1, each annotated fail read counts ten times
    n_h, n_f = sel["n_hifi"].to_numpy().astype(float), sel["n_fail_sub"].to_numpy().astype(float)
    has = (n_h + n_f) > 0
    assert (n_f[has] > 0).any()
    assert np.allclose(share[has], n_h[has] / (n_h[has] + 10 * n_f[has]))
    # bootstrap CI of D next to the M_T band
    assert v["D"]["boot_lo"] <= v["D"]["boot_hi"]
    assert v["D"]["residual_lo"] <= v["D"]["residual"] <= v["D"]["residual_hi"]
    assert v["D"]["residual"] == pytest.approx(v["D"]["observed"] - v["D"]["MT_mean"])
    assert pl.read_csv(q / "tables" / "positions_pairwise_sread.tsv", separator="\t").height == 28
    ll = pl.read_csv(q / "tables" / "ipw_long_long_sread.tsv", separator="\t")
    assert ll["null"].str.starts_with("independence, IPW").all() and set(ll["cutoff"]) == set(q1.CUTOFFS)
    ipw = pl.read_csv(q / "tables" / "ipw_sread.tsv", separator="\t")
    assert {"null_mean", "null_lo", "null_hi", "diff_vs_null_bp", "diff_lo", "diff_hi", "outside_null"} <= set(ipw.columns)
    cuts = pl.read_csv(q / "tables" / "partner_cutoffs_sread.tsv", separator="\t")
    assert {"ratio_vs_P", "ratio_vs_MT", "outside_P", "outside_MT"} <= set(cuts.columns)
    assert cuts["outside_P"].dtype == pl.Boolean and cuts["outside_MT"].dtype == pl.Boolean
    assert "outside" in ll.columns and ll["outside"].dtype == pl.Boolean


@pytest.mark.real
@pytest.mark.parametrize("sample", SAMPLES)
def test_summary_keys_match_v10(mini_study, sample):
    """q1_summary.json has exactly the keys, in the same order and at every depth, of the frozen v10 summary."""
    new = json.loads((mini_study / sample / "q1" / "q1_summary.json").read_text())
    v10 = json.loads((GOLDEN / sample / "q1" / "q1_summary.json").read_text())
    assert key_structure(new) == key_structure(v10)


@pytest.mark.real
def test_cross_sample(mini_study, mini, tmp_path):
    # cross-sample mode reads every sample of the samples table: restrict it to two
    two = ("BD144_ND", "BD144_kin")
    tsv = pl.read_csv(mini / "samples_mini.tsv", separator="\t").filter(pl.col("sample").is_in(pl.Series(two)))
    path = tmp_path / "samples_two.tsv"
    tsv.write_csv(path, separator="\t")
    run_cross_sample("s06_concatenation_bias.py", mini_study, path, "--fraction", "1")
    q = mini_study / "all" / "q1"
    assert {p.name for p in (q / "tables").iterdir()} == {f"{t}_{v}.tsv" for t in ("d_statistic", "oe_2plus") for v in ("sread", "insert")}
    assert not (q / "figures").exists() and not (q / "manifest.json").exists()
    d_table = pl.read_csv(q / "tables" / "d_statistic_sread.tsv", separator="\t")
    assert d_table["sample"].to_list() == list(two) and {"boot_lo", "boot_hi", "MT_lo", "MT_hi"} <= set(d_table.columns)
    d_nd = json.loads((mini_study / "BD144_ND" / "q1" / "q1_summary.json").read_text())["values"][0]["D"]
    assert d_table.row(0, named=True)["observed"] == d_nd["observed"]
    oe = pl.read_csv(q / "tables" / "oe_2plus_insert.tsv", separator="\t")
    assert oe.select("sample", "null").rows() == [(s, n) for s in two for n in ("null P", "M_T")]
