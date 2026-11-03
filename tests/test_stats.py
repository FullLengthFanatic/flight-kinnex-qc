"""Tests for stats.py, including three scenarios with known truth."""
import numpy as np
import polars as pl
import pytest

from pipeline_helpers import PIPELINE  # noqa: F401  (puts study/pipeline on sys.path)

import stats as st

EDGES = np.array([0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 20000], float)
MU = np.log(1500) + np.linspace(-0.05, 0.05, 8)


def independent(rng, n):
    return np.exp(rng.normal(MU, 0.6, (n, 8)))


def capped(rng, n, t50=15000):
    """Independent positions, then kept with a logistic probability in total length (not multiplicative)."""
    v = independent(rng, 3 * n)
    keep = rng.random(len(v)) < 1 / (1 + np.exp((v.sum(axis=1) - t50) / 800))
    return v[keep][:n]


def adjacent(rng, n):
    v = independent(rng, n)
    for p in range(7):
        both = (v[:, p] > 2500) & (v[:, p + 1] > 2500)
        v[both, p + 1] *= 0.4
    return v


def bands(stats):
    return np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)


def scenario(values, rng, reps=20):
    e = np.ones_like(values, bool)
    obs = st.partner_curve(values, e, EDGES)
    ok = obs["n_index"].to_numpy() > 200
    om = obs["partner_mean"].to_numpy()
    perm = [st.partner_curve(*st.permute_within_position(values, e, rng), EDGES)["partner_mean"].to_numpy() for _ in range(reps)]
    model = st.fit_mt(values, rng, n_sim=600_000)
    sims = [st.simulate_mt(model, len(values), rng) for _ in range(reps)]
    mt = [st.partner_curve(x, np.ones_like(x, bool), EDGES)["partner_mean"].to_numpy() for x in sims]
    plo, phi = bands(perm)
    mlo, mhi = bands(mt)
    d = st.d_statistic(st.lag_profile(values))
    dn = [st.d_statistic(st.lag_profile(x)) for x in sims]
    return {"om": om[ok], "out_perm": np.mean(((om < plo) | (om > phi))[ok]),
            "out_mt": np.mean(((om < mlo) | (om > mhi))[ok]), "d": d,
            "d_band": (np.percentile(dn, 2.5), np.percentile(dn, 97.5)), "model": model}


class TestScenarios:
    def test_independent_is_flat_and_inside_both_nulls(self):
        r = scenario(independent(np.random.default_rng(1), 60_000), np.random.default_rng(2))
        assert r["om"].max() - r["om"].min() < 60
        assert r["out_mt"] <= 0.3
        assert r["d_band"][0] - 0.002 <= r["d"] <= r["d_band"][1] + 0.002
        assert r["model"].converged

    def test_total_length_selection_explained_by_mt(self):
        r = scenario(capped(np.random.default_rng(3), 60_000), np.random.default_rng(4))
        assert r["om"][0] - r["om"][-1] > 200          # long index, shorter partners
        assert r["out_perm"] >= 0.7                    # the permutation null rejects it
        assert r["out_mt"] <= 0.3                      # total-length selection explains it
        assert r["d_band"][0] - 0.002 <= r["d"] <= r["d_band"][1] + 0.002

    def test_adjacency_effect_detected(self):
        r = scenario(adjacent(np.random.default_rng(5), 60_000), np.random.default_rng(6))
        assert r["d"] < r["d_band"][0] - 0.02


class TestUnits:
    def test_partner_curve_by_hand(self):
        v = np.array([[1000, 5000, 200, 300, 400, 500, 600, 700], [4500, 100, 100, 100, 100, 100, 100, 100]], float)
        e = np.zeros_like(v, bool)
        e[0, 1] = e[1, 0] = True
        c = st.partner_curve(v, e, np.array([0, 4000, 6000], float))
        row = c.row(1, named=True)
        assert row["n_index"] == 2
        partners = [1000, 200, 300, 400, 500, 600, 700] + [100] * 7
        assert row["partner_mean"] == pytest.approx(np.mean(partners) + 5, abs=1e-6)   # bin mids are +5 bp
        assert row["partner_frac_ge_4kb"] == 0
        assert c.row(0, named=True)["n_index"] == 0

    def test_weights_equal_duplication(self):
        rng = np.random.default_rng(0)
        v = independent(rng, 500)
        e = np.ones_like(v, bool)
        w = rng.integers(1, 4, 500).astype(float)
        a = st.partner_curve(v, e, EDGES, weights=w)
        b = st.partner_curve(np.repeat(v, w.astype(int), axis=0), np.ones((int(w.sum()), 8), bool), EDGES)
        assert np.allclose(a["partner_mean"].to_numpy(), b["partner_mean"].to_numpy(), equal_nan=True)

    def test_permutations_keep_margins(self):
        rng = np.random.default_rng(0)
        v = independent(rng, 1000)
        e = rng.random((1000, 8)) < 0.9
        pv, pe = st.permute_within_position(v, e, rng)
        assert np.allclose(np.sort(pv, axis=0), np.sort(v, axis=0))
        assert (pe.sum(axis=0) == e.sum(axis=0)).all()
        sv = st.within_array_shuffle(v, rng)
        assert np.allclose(np.sort(sv, axis=1), np.sort(v, axis=1))

    def test_ipw_independence_null_has_no_artificial_gap(self):
        """External review (2026-10-05): eight independent positions, 1 kb or 5 kb (long with p = 0.1), arrays with 2+
        long segments kept with probability 0.5. Reweighted by the true 1/r the observed partner mean of long index
        segments is 1,400 bp, and the independence null must be 1,400 too; the old weighted null P gave 1,450 bp."""
        rng = np.random.default_rng(3)
        pre = np.where(rng.random((400_000, 8)) < 0.1, 5000.0, 1000.0)
        r = np.where((pre >= 4000).sum(axis=1) >= 2, 0.5, 1.0)
        obs = pre[rng.random(len(pre)) < r]
        w = 1 / np.where((obs >= 4000).sum(axis=1) >= 2, 0.5, 1.0)
        edges = np.array([0, 4000, 1e9])
        ind = st.ipw_independence_null(obs, np.ones(obs.shape, bool), w, edges, [4000], rng, n_boot=20)
        assert ind["obs_cut"][0] == pytest.approx(1400, abs=12)
        assert ind["null_cut"][0] == pytest.approx(1400, abs=12)
        assert ind["diff_cut_lo"][0] < 0 < ind["diff_cut_hi"][0]
        # expected arrays by long count: the reweighted population is binomial(8, 0.1)
        from math import comb
        binom = [comb(8, n) * 0.1 ** n * 0.9 ** (8 - n) for n in range(9)]
        exp = ind["long_expected"][0] / w.sum()
        assert exp == pytest.approx([binom[0], binom[1], binom[2], sum(binom[3:])], abs=0.004)

    def test_ipw_independence_null_without_weights_matches_null_p(self):
        """With weights 1 the independence null is the expectation of the permutation null P."""
        rng = np.random.default_rng(4)
        v = independent(rng, 3000) * np.linspace(0.8, 1.2, 8)
        e = rng.random(v.shape) < 0.9
        edges = np.array([0, 1500, 3000, 1e9])
        ind = st.ipw_independence_null(v, e, np.ones(len(v)), edges, [1500, 3000], rng, n_boot=10)
        perm = []
        for _ in range(60):
            pv, pe = st.permute_within_position(v, e, rng)
            num = den = 0.0
            for k in range(8):
                m = pe[:, k] & (pv[:, k] >= 1500)
                num += (pv[m].sum(axis=1) - pv[m, k]).sum()
                den += 7 * m.sum()
            perm.append(num / den)
        assert ind["null_cut"][0] == pytest.approx(np.mean(perm), rel=0.003)

    def test_long_counts_and_oe(self):
        v = np.array([[5000] * 3 + [100] * 5, [5000] + [100] * 7, [100] * 8], float)
        obs = st.long_counts(v, 4000)
        assert obs.tolist() == [1, 1, 0, 1]
        assert st.long_counts(v, 4000, np.array([2.0, 0.5, 3.0])).tolist() == [3.0, 0.5, 0.0, 2.0]
        t = st.oe_table(obs, [np.array([1, 2, 0, 0]), np.array([1, 0, 2, 0])], "P")
        assert t["expected"].to_list() == [1, 1, 1, 0]
        # observed outside its own null band is flagged, not left for a reader to work out from rounded numbers
        assert t["outside"].to_list() == [(o < lo or o > hi) for o, lo, hi in zip(t["observed"], t["lo"], t["hi"])]

    def test_cutoff_partner_mean_matches_cutoff_summary(self):
        rng = np.random.default_rng(3)
        v = independent(rng, 2000)
        hist, n_index = st.partner_hist(v, np.ones_like(v, bool), EDGES)
        hist[EDGES[:-1] >= 6000] = 0          # no partners at the top cut-off: NaN mean
        cutoffs = (3000, 4000, 6000)
        expected = st.cutoff_summary(hist, n_index, EDGES, cutoffs)["partner_mean"].to_numpy()
        got = st.cutoff_partner_mean(hist, EDGES, cutoffs)
        assert np.isnan(got[-1])
        np.testing.assert_array_equal(got, expected)

    def test_count_edges_merge_sparse_tail(self):
        x = np.concatenate([np.full(1000, 25.0), [5000.0, 9000.0]])
        e = st.count_edges(x, 50, 100)
        counts = np.histogram(x, e)[0]
        assert (counts >= 100).all() or counts[-1] == 2 and len(counts) == 1

    def test_position_pairs(self):
        v = np.tile(np.array([1000, 2000, 5000, 1000, 1000, 1000, 1000, 1000], float), (10, 1))
        t = st.position_pairs(st.position_table(v))
        assert t.height == 28 and t.select("pos_a", "pos_b").row(0) == (0, 1)
        r = t.filter((pl.col("pos_a") == 1) & (pl.col("pos_b") == 2)).row(0, named=True)
        assert r["median_diff"] == -3000 and r["frac_ge_4kb_diff"] == -1.0

    def test_lag_profile_and_d(self):
        rng = np.random.default_rng(0)
        assert abs(st.d_statistic(st.lag_profile(independent(rng, 20_000)))) < 0.03
        prof = np.array([-0.2, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05])
        assert st.d_statistic(prof) == pytest.approx(-0.15)


class TestHiFiSelection:
    def test_w_hifi_empirical(self):
        t = st.w_hifi_empirical(np.array([100, 100, 5000]), np.array([5000, 5000, 5000]), np.array([0, 1000, 10000]))
        assert t["w"].to_list() == [1.0, 0.25]

    def test_weighted_w_recovers_cell_share(self):
        """The finding's scenario (BD144_HM, 2,000,000 pilot): the run holds a = 17.35% of the cell's HiFi reads but
        its whole fail BAM, of which 1 in 10 reads is annotated. Weighting each read by 1 / its sampling probability
        gives back the cell's HiFi share; the unweighted counts give the share of the two samples."""
        edges = np.array([0, 1000, 2000, 3000], float)
        cell_hifi = np.array([300_000, 870_000, 120_000])
        cell_fail = np.array([700_000, 130_000, 880_000])
        truth = cell_hifi / (cell_hifi + cell_fail)                 # 0.30, 0.87, 0.12
        a, p_sub = 0.1735, 0.1
        t_hifi = np.repeat([500.0, 1500.0, 2500.0], np.round(a * cell_hifi).astype(int))
        t_fail = np.repeat([500.0, 1500.0, 2500.0], np.round(p_sub * cell_fail).astype(int))
        weighted = st.w_hifi_empirical(t_hifi, t_fail, edges, 1 / a, 1 / p_sub)["w"].to_numpy()
        unweighted = st.w_hifi_empirical(t_hifi, t_fail, edges)["w"].to_numpy()
        assert np.allclose(weighted, truth, atol=1e-4)
        assert unweighted[1] > truth[1] + 0.05                       # 1 in 10 fail reads against 17% of HiFi reads
        # the pass table: HiFi index at a, the whole fail index at 1
        reads = pl.DataFrame({"np": [3] * 1000, "rq": [0.995] * 174 + [0.9] * 826,
                              "weight": [1 / a] * 174 + [1.0] * 826})
        assert st.pass_rq_table(reads)["frac_rq_ge_099"].item() == pytest.approx(0.548, abs=0.002)   # cell: 1003 of 1829
        assert st.pass_rq_table(reads.drop("weight"))["frac_rq_ge_099"].item() == pytest.approx(0.174)

    def test_model_lookup_gap_and_fail_only_tail(self):
        """A pass count missing from the table takes the next lower value present; a sparse fail-only tail above the
        HiFi maximum is not used, and pass counts above that cap take the value at the cap."""
        pr = pl.DataFrame({"np": [1, 2, 3, 4, 6, 60, 61, 90], "frac_rq_ge_099": [0.1, 0.5, 0.7, 0.8, 0.95, 0.99, 0.0, 0.0],
                           "n": [100, 100, 100, 100, 100, 100, 1, 1]})
        L = np.array([100_000.0])
        t_gap = 100_000 / 6.5 - 45                    # floor(L / (T + 45)) - 1 = 5: missing from the table
        t_top = 100_000 / 80.5 - 45                   # 79 passes: above the HiFi maximum of 60
        w = st.w_hifi_model(L, pr, np.array([t_gap, t_top]), max_np=60)
        assert w.tolist() == [0.8, 0.99]
        old_style = st.w_hifi_model(L, pr, np.array([t_top]))       # no cap: the fail-only tail sets it
        assert old_style[0] == 0.0

    def test_pass_rq_and_model(self):
        reads = pl.DataFrame({"np": [1, 1, 2, 3, 3, 10], "rq": [0.9, 0.95, 0.98, 0.99, 0.995, 0.9999]})
        pr = st.pass_rq_table(reads)
        assert pr.filter(pl.col("np") == 3)["frac_rq_ge_099"].item() == 1.0
        w = st.w_hifi_model(np.full(1000, 100_000.0), pr, np.array([2000.0, 20000.0, 60000.0]))
        assert w[0] >= w[1] >= w[2]
        assert w[2] == 0.0          # 100 kb polymerase read, 60 kb array: 0 full passes

    def test_pol_length_usable(self):
        hifi = pl.DataFrame({"zm": [1, 2], "read_len": [1000, 1000], "np": [10, 10]}, schema_overrides={"zm": pl.UInt32})
        good = pl.DataFrame({"zm": [1, 2], "pol_length": [12000, 11000]}, schema_overrides={"zm": pl.UInt32})
        bad = pl.DataFrame({"zm": [1, 2], "pol_length": [1000, 11000]}, schema_overrides={"zm": pl.UInt32})
        assert st.pol_length_usable(good, hifi) == (True, 1.0)
        assert st.pol_length_usable(bad, hifi) == (False, 0.5)

    def test_ipw_weights(self):
        w_at = lambda t: np.where(t < 10_000, 1.0, np.where(t < 20_000, 0.1, 0.001))
        weights, kept = st.ipw_weights(np.array([5000.0] * 98 + [15_000.0, 30_000.0]), w_at, clip_q=0.99)
        assert kept.tolist() == [True] * 99 + [False]
        assert weights[-1] == 0 and weights[0] == 1
        assert weights[98] <= 10


class TestArraysAndTruncation:
    def test_complete_array_matrix(self):
        rows = []
        for zm in (1, 2):
            for pos in range(8):
                rows.append({"zm": zm, "pos": pos, "sread_len": 100 * (pos + 1), "insert_len": None if (zm == 2 and pos == 3) else 90,
                             "end_class": "proper" if pos else "tso_only", "internal": pos == 5})
        rows.append({"zm": 3, "pos": 0, "sread_len": 1, "insert_len": 1, "end_class": "proper", "internal": False})
        seg = pl.DataFrame(rows)
        v, e, zm = st.complete_array_matrix(seg, "sread_len")
        assert v.shape == (2, 8) and zm.tolist() == [1, 2]
        assert v[0].tolist() == [100, 200, 300, 400, 500, 600, 700, 800]
        assert e[0].tolist() == [False, True, True, True, True, False, True, True]
        v, e, zm = st.complete_array_matrix(seg, "insert_len")
        assert zm.tolist() == [1]

    def test_partial_runs_split_mirror_and_restart(self):
        """Every non-complete read is a partial array; a mirror read gives one run per half, a restart one
        per start, a skipped slot stays in its run. Breaks sit at a missing end, a skip and a mirror junction."""
        def read(zm, tokens, lengths):
            rows = []
            for i, (l, r) in enumerate(zip(tokens, tokens[1:])):
                mas = "ABCDEFGHQ"
                same = l.endswith("'") == r.endswith("'")
                d = abs(mas.index(l[0]) - mas.index(r[0]))
                rows.append({"zm": zm, "seg_index": i, "left_token": l, "pos": min(mas.index(l[0]), mas.index(r[0])) if same and d == 1 else None,
                             "end_class": "proper", "internal": False, "sread_len": lengths[i], "insert_len": lengths[i] - 90})
            return rows
        seg = pl.DataFrame(read(1, ["A", "B", "C", "D", "D'", "C'", "B'", "A'"], [1000, 1100, 1200, 50, 2200, 2100, 2000])  # mirror
                           + read(2, ["C", "D", "E", "F", "G", "H", "Q"], [3000] * 6)                                         # truncated
                           + read(3, ["Q'", "H'", "G'", "F'", "D'", "C'", "B'", "A'"], [500] * 7)                            # skip E, minus
                           + read(4, ["F", "G", "H", "Q", "A", "B"], [800] * 5), schema_overrides={"pos": pl.Int8})           # restart
        runs = st.partial_runs(seg)
        assert runs.group_by("zm").agg(pl.col("run").n_unique()).sort("zm")["run"].to_list() == [2, 1, 1, 2]
        v, e, zm = st.partial_array_matrix(runs, "sread_len")
        assert zm.tolist() == [1, 1, 2, 3, 4, 4]
        assert v[0, :3].tolist() == [1000, 1100, 1200] and v[1, :3].tolist() == [2000, 2100, 2200]
        assert np.isnan(v[0, 3:]).all() and np.isnan(v[3, [3, 4]]).all() and not np.isnan(v[3, [2, 5]]).any()
        brk = st.break_side(runs, "sread_len")
        # mirror halves break at position 2 (twice); truncated read at 2 (A, B missing); skip of E at 2 and 5;
        # restart: F G H Q gives 5 (missing 4), A B gives 0 (missing 1)
        assert sorted(brk["pos"].to_list()) == [0, 2, 2, 2, 2, 5, 5]

    def test_truncation_table(self):
        rng = np.random.default_rng(0)
        full = np.full((200, 8), 1000.0)
        brk = pl.DataFrame({"pos": [0] * 50 + [7] * 50, "value": [4500.0] * 25 + [1000.0] * 25 + [1000.0] * 50})
        t = st.truncation_table(brk, full, rng, n_boot=50)
        r0 = t.filter(pl.col("pos") == 0).row(0, named=True)
        assert r0["frac_ge_4kb_break"] == 0.5 and r0["frac_ge_4kb_complete"] == 0
        assert t.filter(pl.col("pos") == 7)["median_ratio"].item() == 1.0

    def test_truncation_table_is_deterministic_regardless_of_row_order(self):
        """The bootstrap draws from a shared rng across positions, so a run-dependent group_by order
        would change which position consumes which draws. Same input and seed must give the same table,
        even with the input rows shuffled between calls."""
        full = np.full((30, 8), 1000.0)
        brk = pl.DataFrame({"pos": [3] * 20 + [0] * 20 + [7] * 20,
                            "value": [4500.0] * 10 + [1000.0] * 10 + [1000.0] * 20 + [4500.0] * 10 + [1000.0] * 10})
        shuffled = brk.sample(fraction=1.0, shuffle=True, seed=7)
        t1 = st.truncation_table(brk, full, np.random.default_rng(42), n_boot=30)
        t2 = st.truncation_table(shuffled, full, np.random.default_rng(42), n_boot=30)
        assert t1.equals(t2)
