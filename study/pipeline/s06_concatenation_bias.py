#!/usr/bin/env python3
"""Stage 06: does concatenation pair long molecules with short ones, and why?

Runs on our segmentation: complete arrays are reads whose adapter chain is A..Q in order.
Per sample (--sample): Steps 1 to 4 for sread_len and insert_len, written to <sample>/q1/tables/*.tsv and
<sample>/q1/q1_summary.json. Cross-sample (--all): the D statistic and the long-long observed/expected of every
sample in the samples table, written to all/q1/tables/.
Usage, from the repository root:
  study/pipeline/s06_concatenation_bias.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--processes N]
      [--n-null 200] [--n-sim 10000000] [--mem-budget-gb GB]
  study/pipeline/s06_concatenation_bias.py --all [--samples PATH] [--out-root DIR] [--fraction F]
Defaults come from study/study.toml.
"""
import argparse
import json
import sys
import time

import numpy as np
import polars as pl

import stats as st
from config import SAMPLE_ORDER, load_config, load_samples, out_dir, skera_report_reads
from kinnex_qc.annotate import PRIORITY
from kinnex_qc.parallel import process_pool

CUTOFFS = (3000, 4000, 5000, 6000)
EDGES = np.array([*range(0, 10_001, 500), 1e9], dtype=np.float64)       # 0.5 kb bins, then >= 10 kb
ADAPTERS_BP = 9 * 17 - 3                                                  # A..H (17 nt) and Q (14 nt)
T_EDGES = np.arange(0, 60_001, 1000, dtype=np.float64)
VALUES = ("sread_len", "insert_len")
PARTIAL_PATTERNS = tuple(p for p in PRIORITY if p != "complete")          # a partial array is any other read
NULL_BYTES_PER_ARRAY = 700   # peak extra memory of one M_T replicate per complete array (measured: 680)
MEM_MARGIN = 1.25

_G: dict = {}   # arrays shared with forked null workers


def one_rep(job):
    """One null replicate: kind P (permutation), MT (simulation), shuffle, or boot (arrays resampled with
    replacement, for the bootstrap CI of D)."""
    kind, key, seed = job
    rng = np.random.default_rng(seed)
    g = _G[key]
    v, e = g["values"], g["eligible"]
    out = {}
    if kind == "shuffle":
        return {"lag": st.lag_profile(st.within_array_shuffle(v, rng))}
    if kind == "boot":
        return {"lag": st.lag_profile(v[rng.integers(0, len(v), len(v))])}
    if kind == "MT":
        v = st.simulate_mt(g["model"], len(v), rng)
        e = rng.random(v.shape) < g["elig_frac"]
        out["lag"] = st.lag_profile(v)
    else:
        v, e = st.permute_within_position(v, e, rng)
    hist, _ = st.partner_hist(v, e, EDGES)
    out["median"], out["mean"], out["frac"] = st.hist_summary(hist)
    out["cut"] = st.cutoff_partner_mean(hist, EDGES, CUTOFFS)
    out["long"] = np.stack([st.long_counts(v, c) for c in CUTOFFS])
    return out


def mem_budget_bytes(gb: float | None) -> float:
    """Memory the null workers of this process may use: gb GiB if given, else 80% of MemAvailable."""
    if gb is not None:
        return float(gb) * 2**30
    with open("/proc/meminfo") as f:
        kib = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
    return 0.8 * kib * 1024


def null_workers(processes: int, n_arrays: int, budget_bytes: float) -> int:
    """At most processes workers, and no more than the budget holds at n_arrays * 700 bytes * 1.25 each; at least 1."""
    per_worker = n_arrays * NULL_BYTES_PER_ARRAY * MEM_MARGIN
    return max(1, min(processes, int(budget_bytes // max(per_worker, 1))))


def run_nulls(processes: int, mem_gb: float | None, kind: str, key: str, n: int, base_seed: int) -> list[dict]:
    """n replicates in a fresh fork pool, so the workers see the arrays placed in _G just before.

    The workers are forked after this process has used polars, so one_rep must not call polars.
    """
    n_arrays = len(_G[key]["values"])
    budget = mem_budget_bytes(mem_gb)
    workers = null_workers(processes, n_arrays, budget)
    print(f"{key} null {kind}: {n_arrays:,} arrays, {n_arrays * NULL_BYTES_PER_ARRAY * MEM_MARGIN / 2**30:.3f} GB "
          f"per worker (estimate), budget {budget / 2**30:.1f} GB, {workers} of {processes} workers", file=sys.stderr)
    with process_pool(workers) as pool:
        return list(pool.map(one_rep, [(kind, key, base_seed + i) for i in range(n)]))


def band(reps: list[dict], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 2.5/97.5% band of a field across replicates, per column.

    A column (a length bin, mostly) can be NaN in every replicate: no index segment landed there in
    any of them. NaN is then the right answer, but nanmean/nanpercentile warn when the slice they
    reduce is entirely NaN, so those columns are written directly instead of reduced.
    """
    z = np.array([r[field] for r in reps], dtype=np.float64)
    have = ~np.all(np.isnan(z), axis=0)
    mean, lo, hi = (np.full(z.shape[1], np.nan) for _ in range(3))
    if have.any():
        mean[have] = np.nanmean(z[:, have], axis=0)
        lo[have] = np.nanpercentile(z[:, have], 2.5, axis=0)
        hi[have] = np.nanpercentile(z[:, have], 97.5, axis=0)
    return mean, lo, hi


def write_table(q, name: str, df: pl.DataFrame) -> None:
    df.write_csv(q / "tables" / f"{name}.tsv", separator="\t")


def load(out, source: str, patterns: tuple[str, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    summ = pl.read_parquet(out / "read_summary.parquet", columns=["zm", "source", "primary_pattern", "read_len"])
    reads = summ.filter((pl.col("source") == source) & pl.col("primary_pattern").is_in(patterns))
    # A semi join, not is_in: is_in against a multi-million Series costs n_chunks x len(series) on a scan read
    # one chunk per parquet row group (polars 1.5.0), and complete arrays run into the millions at full scale.
    seg = (pl.scan_parquet(out / "annot_segments.parquet")
           .filter(pl.col("source") == source)
           .join(reads.lazy().select("zm"), on="zm", how="semi")
           .select("zm", "seg_index", "left_token", "pos", "end_class", "internal", "sread_len", "insert_len").collect())
    return reads, seg


def curve_table(obs: pl.DataFrame, nulls: dict) -> pl.DataFrame:
    t = obs
    for name, reps in nulls.items():
        for f in ("mean", "median"):
            m, lo, hi = band(reps, f)
            t = t.with_columns(pl.Series(f"{name}_{f}", m), pl.Series(f"{name}_{f}_lo", lo), pl.Series(f"{name}_{f}_hi", hi))
    return t


def longest_index_bp(values: np.ndarray, eligible: np.ndarray) -> float:
    """Longest index segment observed (eligible, present). Bins above it are empty by absence, not a fault."""
    idx = values[eligible]
    idx = idx[~np.isnan(idx)]
    return float(idx.max()) if idx.size else 0.0


def complete_arrays(seg: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """st.complete_array_matrix for value, plus t_values: the S-read lengths of the same arrays in the same order.
    IPW looks w(T) up at the array length for both values, and an insert total falls about 0.7 kb short of it."""
    values, eligible, zms = st.complete_array_matrix(seg, value)
    if value == "sread_len":
        return values, eligible, zms, values
    sv, _, szm = st.complete_array_matrix(seg, "sread_len")      # zm order in both; insert arrays are a subset
    return values, eligible, zms, sv[np.searchsorted(szm, zms)]


def analyse_value(q, processes: int, mem_gb: float | None, out, value: str, n_null: int, n_sim: int, rng, w_models: dict) -> dict:
    tag = value.replace("_len", "")
    reads, seg = load(out, "hifi", ("complete",))
    values, eligible, zms, t_values = complete_arrays(seg, value)
    summary = {"value": value, "n_arrays": int(len(values))}
    t0 = time.time()

    # Step 1: observed vs random pairing (null P) ------------------------------------------
    elig_frac = eligible.mean(axis=0)
    _G[value] = {"values": values, "eligible": eligible, "elig_frac": elig_frac}
    obs = st.partner_curve(values, eligible, EDGES)
    perm = run_nulls(processes, mem_gb, "P", value, n_null, 1000)

    # Step 2a: M_T fit and its nulls
    model = st.fit_mt(values, rng, n_sim=n_sim)
    _G[value]["model"] = model
    mt = run_nulls(processes, mem_gb, "MT", value, n_null, 2000)
    shuffle = run_nulls(processes, mem_gb, "shuffle", value, n_null, 3000)
    # M_T fit: iterative proportional fit of independent positions selected on total array length; converged means
    # every total-length bin with at least 100 arrays and every per-position length bin with at least 13 segments
    # (100 / 8) is within 1% of the observed count.
    fit_t = pl.DataFrame({"value": [value], "n_arrays": [len(values)], "n_sim": [n_sim], "iterations": [model.iterations],
                          "converged": [model.converged], "max_rel_err": [model.max_rel_err]})
    write_table(q, f"mt_fit_{tag}", fit_t)

    # partner length by index length: for every proper, non-internal index segment, the other seven segments of its
    # array; null P shuffles segments across arrays within each position, M_T is the fitted total-length selection
    # model; bands are 2.5 to 97.5% of n_null replicates
    ct = curve_table(obs, {"P": perm, "MT": mt})
    write_table(q, f"partner_curve_{tag}", ct)
    summary["longest_index_bp"] = longest_index_bp(values, eligible)

    hist, n_index = st.partner_hist(values, eligible, EDGES)
    cut = st.cutoff_summary(hist, n_index, EDGES, CUTOFFS)
    for name, reps in (("P", perm), ("MT", mt)):
        mu, lo, hi = band(reps, "cut")
        cut = cut.with_columns(pl.Series(f"{name}_mean", mu), pl.Series(f"{name}_lo", lo), pl.Series(f"{name}_hi", hi),
                               (pl.col("partner_mean") - pl.Series(mu)).alias(f"diff_vs_{name}_bp"),
                               (pl.col("partner_mean") / pl.Series(mu)).alias(f"ratio_vs_{name}"),
                               ((pl.col("partner_mean") < pl.Series(lo)) | (pl.col("partner_mean") > pl.Series(hi))).alias(f"outside_{name}"))
    write_table(q, f"partner_cutoffs_{tag}", cut)
    summary["cutoffs"] = cut.to_dicts()

    long_rows, oe_2plus = [], {"null P": [], "M_T": []}
    for j, c in enumerate(CUTOFFS):
        o = st.long_counts(values, c)
        for name, reps in (("null P", perm), ("M_T", mt)):
            t = st.oe_table(o, [r["long"][j] for r in reps], name).with_columns(pl.lit(c).alias("cutoff"))
            long_rows.append(t)
            exp2 = np.array([r["long"][j][2:].sum() for r in reps], float)
            oe_2plus[name].append(o[2:].sum() / max(exp2.mean(), 1e-9))
    write_table(q, f"long_long_{tag}", pl.concat(long_rows))
    summary["oe_2plus"] = {name: dict(zip(map(str, CUTOFFS), map(float, v))) for name, v in oe_2plus.items()}

    # Step 1 sensitivity: partial arrays, every HiFi read that is not complete, one row per run of adapters
    _, pseg = load(out, "hifi", PARTIAL_PATTERNS)
    runs = st.partial_runs(pseg)
    pv, pe, _ = st.partial_array_matrix(runs, value)
    if len(pv) >= 50:
        _G[value + "_partial"] = {"values": pv, "eligible": pe}
        pperm = run_nulls(processes, mem_gb, "P", value + "_partial", n_null, 4000)
        write_table(q, f"partial_curve_{tag}", curve_table(st.partner_curve(pv, pe, EDGES), {"P": pperm}))

    # Step 2c: IPW with each available w(T), against independent positions in the reweighted population. Each array
    # is weighted by 1/w(T), T its array length (the sum of its S-read lengths plus the adapters, for both values).
    ipw_rows, ipw_long, ipw_oe = [], [], {}
    t_total = np.nansum(t_values, axis=1) + ADAPTERS_BP
    for name, w_at in w_models.items():
        weights, kept = st.ipw_weights(t_total, w_at)
        ind = st.ipw_independence_null(values, eligible, weights, EDGES, CUTOFFS, rng, n_boot=n_null)
        whist, wn = st.partner_hist(values, eligible, EDGES, weights)
        wcut = st.cutoff_summary(whist, wn, EDGES, CUTOFFS).with_columns(pl.Series("partner_mean", ind["obs_cut"]))
        ipw_rows.append(wcut.with_columns(pl.lit(name).alias("w"), pl.lit(float(1 - kept.mean())).alias("excluded_fraction"),
                                          pl.Series("null_mean", ind["null_cut"]), pl.Series("null_lo", ind["null_cut_lo"]),
                                          pl.Series("null_hi", ind["null_cut_hi"]),
                                          (pl.col("partner_mean") - pl.Series(ind["null_cut"])).alias("diff_vs_null_bp"),
                                          pl.Series("diff_lo", ind["diff_cut_lo"]), pl.Series("diff_hi", ind["diff_cut_hi"]),
                                          (pl.col("partner_mean") / pl.Series(ind["null_cut"])).alias("ratio_vs_null"),
                                          ((pl.Series(ind["diff_cut_lo"]) > 0) | (pl.Series(ind["diff_cut_hi"]) < 0)).alias("outside_null")))
        ipw_oe[name] = {}
        for j, c in enumerate(CUTOFFS):
            o = st.long_counts(values, c, weights)
            ipw_long.append(st.oe_table(o, list(ind["long_boot"][:, j]), f"independence, IPW ({name})").with_columns(pl.lit(c).alias("cutoff")))
            ipw_oe[name][str(c)] = float(o[2:].sum() / max(ind["long_expected"][j][2:].sum(), 1e-9))
    if ipw_rows:
        it = pl.concat(ipw_rows)
        write_table(q, f"ipw_{tag}", it)
        write_table(q, f"ipw_long_long_{tag}", pl.concat(ipw_long))
        summary["ipw_oe_2plus"] = ipw_oe
        summary["ipw"] = it.to_dicts()

    # Step 2d: fail-read arrays
    freads, fseg = load(out, "fail_sub", ("complete",))
    fv, fe, _ = st.complete_array_matrix(fseg, value)
    if len(fv) >= 50:
        _G[value + "_fail"] = {"values": fv, "eligible": fe}
        fperm = run_nulls(processes, mem_gb, "P", value + "_fail", n_null, 6000)
        write_table(q, f"fail_curve_{tag}", curve_table(st.partner_curve(fv, fe, EDGES), {"P": fperm}))

    # Step 3: adjacency
    prof = st.lag_profile(values)
    _, mlo, mhi = band(mt, "lag")
    _, slo, shi = band(shuffle, "lag")
    d = st.d_statistic(prof)
    dm = np.array([st.d_statistic(r["lag"]) for r in mt])
    ds = np.array([st.d_statistic(r["lag"]) for r in shuffle])
    boot = run_nulls(processes, mem_gb, "boot", value, n_null, 9000)          # bootstrap CIs over arrays
    _, blo, bhi = band(boot, "lag")
    db = np.array([st.d_statistic(r["lag"]) for r in boot])
    d_lo, d_hi = float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))
    mt_lo, mt_hi = float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))
    # residual adjacency effect: D beyond what M_T predicts, with the bootstrap and M_T spreads combined (normal
    # approximation; the uncertainty of the M_T fit itself is not included). Overlapping intervals are not a test.
    resid, resid_se = d - float(dm.mean()), float(np.hypot(db.std(ddof=1), dm.std(ddof=1)))
    lag_t = pl.DataFrame({"lag": np.arange(1, 8), "observed": prof, "boot_lo": blo, "boot_hi": bhi, "MT_lo": mlo, "MT_hi": mhi,
                          "shuffle_lo": slo, "shuffle_hi": shi})
    write_table(q, f"lag_{tag}", lag_t)
    summary["D"] = {"observed": d, "boot_lo": d_lo, "boot_hi": d_hi, "MT_lo": mt_lo, "MT_hi": mt_hi, "MT_mean": float(dm.mean()),
                    "residual": resid, "residual_lo": resid - 1.96 * resid_se, "residual_hi": resid + 1.96 * resid_se,
                    "residual_z": resid / resid_se,
                    "shuffle_lo": float(np.percentile(ds, 2.5)), "shuffle_hi": float(np.percentile(ds, 97.5))}

    # Step 4: positions and truncation
    pos_t = st.position_table(values)
    write_table(q, f"positions_{tag}", pos_t)
    write_table(q, f"positions_pairwise_{tag}", st.position_pairs(pos_t))
    brk = st.break_side(runs, value)
    if brk.height:
        write_table(q, f"truncation_{tag}", st.truncation_table(brk, values, rng))
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def sampling(out, sample, fraction: float) -> dict:
    """The probability that a read of the cell is in this run, per read set. Every read enters
    w(T) and the pass table with weight 1 / that probability.

    hifi: hifi_index rows over the cell's HiFi reads (skera's read_segmentation.reads) in a hash sample; 1 at
    fraction 1. fail: fail_index rows over the records of the fail BAM (stage 02 indexes the whole BAM, so 1).
    fail_sub: the annotated fail reads, 1 in fail_sub_modulo of fail_index by ZMW hash, so fail / fail_sub_modulo.
    """
    done = json.loads((out / "annotate.done.json").read_text())
    n_hifi = pl.scan_parquet(out / "hifi_index.parquet").select(pl.len()).collect().item()
    n_fail = pl.scan_parquet(out / "fail_index.parquet").select(pl.len()).collect().item()
    if fraction < 1:
        total = skera_report_reads(sample)
        hifi = n_hifi / total
        hifi_note = f"{hifi:.4f} ({n_hifi:,} of skera's {total:,} HiFi reads of the cell, a hash sample, fraction {fraction:g})"
    else:
        total, hifi = None, 1.0
        hifi_note = "1 (a full run holds every HiFi read of the cell)"
    fail = n_fail / done["fail_bam_records"]
    return {"hifi": hifi, "fail": fail, "fail_sub": fail / done["fail_sub_modulo"], "fail_sub_modulo": done["fail_sub_modulo"],
            "hifi_reads": n_hifi, "hifi_reads_total": total, "fail_reads": n_fail, "hifi_note": hifi_note}


def selection_curves(q, out, rng, samp: dict) -> dict:
    """Step 2b: w(T) estimates; returns name -> w_at for the IPW step, and writes the selection_T table.

    The empirical w(T) comes from the annotated reads (every HiFi read and the fail-read subsample; T is the read
    length, which our segments and adapters tile), the model's pass table from the HiFi and fail indexes; every
    read is weighted by 1 / its sampling probability (sampling). The table's curves are scaled to 1 at their maximum
    and left empty where they have no data (s(T): no complete arrays of that length); IPW uses the unscaled empirical
    HiFi share (w_empirical_hifi_share), where the 0.01 floor applies. n_hifi and n_fail_sub are unweighted counts.
    The model estimate is used only when the pol_length check passes (usable if >= 0.9 of HiFi ZMWs pass).
    """
    hifi = pl.read_parquet(out / "hifi_index.parquet")
    fail = pl.read_parquet(out / "fail_index.parquet")
    ann = pl.read_parquet(out / "read_summary.parquet", columns=["source", "read_len"])
    emp = st.w_hifi_empirical(ann.filter(pl.col("source") == "hifi")["read_len"].to_numpy(),
                              ann.filter(pl.col("source") == "fail_sub")["read_len"].to_numpy(), T_EDGES,
                              1 / samp["hifi"], 1 / samp["fail_sub"])
    w_share = emp["w"].to_numpy()                     # the cell's HiFi share per read-length bin, NaN without reads
    w_e = np.nan_to_num(w_share, nan=0.0)
    w_models = {"empirical": lambda t, w=w_e: w[np.clip(np.digitize(t, T_EDGES) - 1, 0, len(w) - 1)]}
    usable, frac = st.pol_length_usable(pl.read_parquet(out / "zmw.parquet"), hifi)
    grid = (T_EDGES[:-1] + T_EDGES[1:]) / 2
    series = {"w(T) empirical": w_share / np.nanmax(w_share)}
    if usable:
        zmw = pl.read_parquet(out / "zmw.parquet", columns=["pol_length"])
        pr = st.pass_rq_table(pl.concat([hifi.select("np", "rq", pl.lit(1 / samp["hifi"]).alias("weight")),
                                         fail.select("np", "rq", pl.lit(1 / samp["fail"]).alias("weight"))]))
        w_m = st.w_hifi_model(zmw["pol_length"].to_numpy(), pr, grid, max_np=int(hifi["np"].max()))
        w_m = w_m / max(w_m.max(), 1e-12)
        w_models["model"] = lambda t, w=w_m: np.interp(t, grid, w)
        series["w(T) model"] = w_m
    reads, seg = load(out, "hifi", ("complete",))
    values, _, _ = st.complete_array_matrix(seg, "sread_len")
    counts = np.histogram(values.sum(axis=1) + ADAPTERS_BP, T_EDGES)[0]
    model = st.fit_mt(values, rng, n_sim=min(len(values) * 10, 2_000_000))
    s_on_grid = model.s[np.clip(np.digitize(grid - ADAPTERS_BP, model.t_edges) - 1, 0, len(model.s) - 1)]
    s_on_grid = np.where(counts > 0, s_on_grid, np.nan)     # no complete arrays of that length: no value
    series["s(T) from M_T"] = s_on_grid / np.nanmax(s_on_grid)
    table = pl.DataFrame({"T_bin_start": T_EDGES[:-1], "T_bin_end": T_EDGES[1:], **series,
                          "w_empirical_hifi_share": w_share, "n_hifi": emp["n_hifi"], "n_fail_sub": emp["n_fail"],
                          "complete_arrays": counts})
    write_table(q, "selection_T", table.with_columns(pl.lit(usable).alias("pol_length_usable"), pl.lit(frac).alias("pol_length_check_fraction")))
    return w_models


def per_sample(args) -> None:
    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    q = out / "q1"
    (q / "tables").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260924)
    samp = sampling(out, sample, args.fraction)
    w_models = selection_curves(q, out, rng, samp)
    summaries = [analyse_value(q, args.processes, args.mem_budget_gb, out, value, args.n_null, args.n_sim, rng, w_models)
                 for value in VALUES]
    (q / "q1_summary.json").write_text(json.dumps({"sample": sample.sample, "sampling": samp, "values": summaries},
                                                  indent=1, default=float))


def cross_sample(args) -> None:
    """Per value: d_statistic_<tag> (observed D with its bootstrap 95% CI over arrays, the M_T 95% band and the
    residual beyond M_T) and oe_2plus_<tag> (observed over expected number of arrays with 2+ segments at or above
    each cut-off), one row per sample (and null) in SAMPLE_ORDER."""
    samples = [s for s in SAMPLE_ORDER if s in load_samples(args.samples)]
    q = out_dir(args.out_root, "all", args.fraction) / "q1"
    (q / "tables").mkdir(parents=True, exist_ok=True)
    for value in VALUES:
        tag = value.replace("_len", "")
        oe, drows = [], []
        for s in samples:
            summ = next(v for v in json.loads((out_dir(args.out_root, s, args.fraction) / "q1" / "q1_summary.json").read_text())["values"]
                        if v["value"] == value)
            drows.append({"sample": s, **{k: summ["D"][k] for k in ("observed", "boot_lo", "boot_hi", "MT_lo", "MT_hi", "residual",
                                                                     "residual_lo", "residual_hi", "residual_z")}})
            for name, row in summ["oe_2plus"].items():
                oe.append({"sample": s, "null": name, **{f"oe_ge_{c}": row[str(c)] for c in CUTOFFS}})
        write_table(q, f"d_statistic_{tag}", pl.DataFrame(drows))
        write_table(q, f"oe_2plus_{tag}", pl.DataFrame(oe))


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample")
    g.add_argument("--all", action="store_true", help="cross-sample tables from every sample's q1_summary.json")
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="names the output directory of a hash sample run")
    ap.add_argument("--processes", type=int, default=config["processes"])
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-sim", type=int, default=10_000_000)
    ap.add_argument("--mem-budget-gb", type=float, help="memory the null workers may use (GiB); default 80%% of MemAvailable")
    args = ap.parse_args()
    cross_sample(args) if args.all else per_sample(args)


if __name__ == "__main__":
    main()
