"""Q1 statistics. Arrays are (n_arrays, 8) float arrays of lengths, column = array position."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

LONG_BP = 4000
VBIN = 10                 # bp resolution of partner-length histograms (medians)
VMAX = 30_000             # partner lengths above this share the last histogram bin
HAIRPIN = 45              # SMRTbell hairpin length added to each pass


# ---------- arrays ----------

def complete_array_matrix(segments: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, eligible, zm) for complete arrays.

    segments: our segments (annot_segments) of reads whose chain is complete, i.e. exactly one
    segment per position 0..7. eligible = proper and not internal (an index may only be such a
    segment). Arrays with a null value at any position (insert_len of a non-proper segment) are dropped.
    """
    s = segments.filter(pl.col("pos").is_not_null()).sort("zm", "pos")
    per = s.group_by("zm", maintain_order=True).agg(pl.len().alias("n"), pl.col("pos").n_unique().alias("u"),
                                                     pl.col(value).null_count().alias("nulls"))
    keep = per.filter((pl.col("n") == 8) & (pl.col("u") == 8) & (pl.col("nulls") == 0))["zm"]
    s = s.filter(pl.col("zm").is_in(keep))
    n = s.height // 8
    values = s[value].to_numpy().astype(np.float64).reshape(n, 8)
    eligible = ((s["end_class"] == "proper") & ~s["internal"]).to_numpy().reshape(n, 8)
    return values, eligible, s["zm"].to_numpy()[::8]


def partial_runs(segments: pl.DataFrame) -> pl.DataFrame:
    """Positioned segments of partial arrays with a run id, one run per stretch of an array.

    segments: our segments of reads that are not complete (columns zm, seg_index, left_token, pos and more).
    A run is a stretch of positioned segments of one read, in read order and on one strand, whose positions
    move one way (up on the "+" strand, down on "-"). A mirror read (X X') gives two runs, split at its
    junction, and a read whose chain starts again gives one run per start; a skipped slot stays inside
    its run as a missing position, like the missing end of a truncated read.
    """
    s = segments.filter(pl.col("pos").is_not_null()).sort("zm", "seg_index")
    zm = s["zm"].to_numpy()
    pos = s["pos"].to_numpy().astype(np.int64)
    minus = s["left_token"].str.ends_with("'").to_numpy()
    new = np.ones(s.height, bool)
    new[1:] = ~((zm[1:] == zm[:-1]) & (minus[1:] == minus[:-1]) & (np.sign(np.diff(pos)) == np.where(minus[1:], -1, 1)))
    return s.with_columns(pl.Series("run", np.cumsum(new) - 1))


def partial_array_matrix(runs: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, eligible, zm) for partial arrays: one row per run of partial_runs, NaN where a position
    is missing or its value is null; runs without any value are dropped."""
    n = int(runs["run"].max()) + 1 if runs.height else 0
    values = np.full((n, 8), np.nan)
    eligible = np.zeros((n, 8), bool)
    s = runs.filter(pl.col(value).is_not_null())
    r, p = s["run"].to_numpy(), s["pos"].to_numpy().astype(np.int64)
    values[r, p] = s[value].to_numpy()
    eligible[r, p] = ((s["end_class"] == "proper") & ~s["internal"]).to_numpy()
    zm = np.zeros(n, np.int64)
    zm[runs["run"].to_numpy()] = runs["zm"].to_numpy()
    keep = ~np.isnan(values).all(axis=1)
    return values[keep], eligible[keep], zm[keep]


def break_side(runs: pl.DataFrame, value: str) -> pl.DataFrame:
    """(pos, value) of the segments next to a break, from partial_runs: a positioned
    segment whose neighbouring position (p - 1 or p + 1, inside 0 to 7) has no segment in its run. That
    is the last position before a missing end, a skipped slot or a mirror junction. Presence counts any
    positioned segment; segments with a null value are then left out."""
    if runs.height == 0:
        return pl.DataFrame({"pos": [], "value": []}, schema={"pos": pl.Int8, "value": pl.Float64})
    r, p = runs["run"].to_numpy(), runs["pos"].to_numpy().astype(np.int64)
    have = np.zeros((int(r.max()) + 1, 10), bool)     # column p + 1 is position p; columns 0 and 9 lie beyond A and Q
    have[r, p + 1] = True
    have[:, [0, 9]] = True
    at_break = ~have[r, p] | ~have[r, p + 2]
    return (runs.filter(pl.Series(at_break)).filter(pl.col(value).is_not_null())
            .select("pos", pl.col(value).cast(pl.Float64).alias("value")))


def permute_within_position(values: np.ndarray, eligible: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Null P: shuffle whole segments (length and eligibility) across arrays within each position.

    Missing positions (NaN, partial arrays) stay missing; only present segments are shuffled.
    """
    v, e = values.copy(), eligible.copy()
    for p in range(values.shape[1]):
        rows = np.flatnonzero(~np.isnan(values[:, p]))
        perm = rng.permutation(rows)
        v[rows, p], e[rows, p] = values[perm, p], eligible[perm, p]
    return v, e


def within_array_shuffle(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the values between positions inside each array."""
    order = np.argsort(rng.random(values.shape), axis=1)
    return np.take_along_axis(values, order, axis=1)


# ---------- partner curves ----------

NV = VMAX // VBIN + 1
MIDS = (np.arange(NV) + 0.5) * VBIN


def partner_hist(values: np.ndarray, eligible: np.ndarray, edges: np.ndarray,
                 weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(hist, n_index): for each index-length bin, the histogram (VBIN bp bins) of the lengths of the
    other segments in the same arrays, and the (weighted) number of index segments.
    An index is an eligible segment; NaN (missing position) is neither index nor partner."""
    n, k = values.shape
    w = np.ones(n) if weights is None else weights
    nb = len(edges) - 1
    present = ~np.isnan(values)
    ib = np.digitize(np.nan_to_num(values, nan=-1.0), edges) - 1
    ok = eligible & present & (ib >= 0) & (ib < nb)
    vb = np.clip(np.nan_to_num(values, nan=0.0) // VBIN, 0, NV - 1).astype(np.int64)   # empty molecules (insert <= 0) go to bin 0
    hist = np.zeros(nb * NV)
    n_index = np.zeros(nb)
    for p in range(k):
        rows = np.flatnonzero(ok[:, p])
        b = ib[rows, p]
        n_index += np.bincount(b, weights=w[rows], minlength=nb)
        for q in range(k):
            if q != p:
                r = rows[present[rows, q]]
                hist += np.bincount(ib[r, p] * NV + vb[r, q], weights=w[r], minlength=nb * NV)
    return hist.reshape(nb, NV), n_index


def hist_summary(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(median, mean, fraction >= 4 kb) of each row of a partner histogram; NaN for empty rows."""
    hist = np.atleast_2d(hist)
    total = hist.sum(axis=1)
    safe = np.maximum(total, 1e-300)
    mean = np.where(total > 0, hist @ MIDS / safe, np.nan)
    frac = np.where(total > 0, hist[:, LONG_BP // VBIN:].sum(axis=1) / safe, np.nan)
    cdf = np.cumsum(hist, axis=1)
    median = np.array([MIDS[np.searchsorted(cdf[i], total[i] / 2)] if total[i] > 0 else np.nan
                       for i in range(len(total))])
    return median, mean, frac


def partner_curve(values: np.ndarray, eligible: np.ndarray, edges: np.ndarray,
                  weights: np.ndarray | None = None) -> pl.DataFrame:
    """Per index-length bin [edges[i], edges[i+1]): number of index segments and the median, mean and
    fraction >= 4 kb of their 7 partners. weights: one per array (IPW), default 1."""
    hist, n_index = partner_hist(values, eligible, edges, weights)
    median, mean, frac = hist_summary(hist)
    return pl.DataFrame({"bin_start": edges[:-1].astype(np.int64), "bin_end": edges[1:].astype(np.int64),
                         "n_index": n_index, "partner_median": median, "partner_mean": mean,
                         "partner_frac_ge_4kb": frac})


def cutoff_summary(hist: np.ndarray, n_index: np.ndarray, edges: np.ndarray, cutoffs) -> pl.DataFrame:
    """Partner median/mean/fraction >= 4 kb for index segments >= each cutoff (a cutoff must be an edge)."""
    rows = []
    for c in cutoffs:
        i = int(np.flatnonzero(edges == c)[0])
        median, mean, frac = hist_summary(hist[i:].sum(axis=0))
        rows.append({"cutoff": c, "n_index": float(n_index[i:].sum()), "partner_median": median[0],
                     "partner_mean": mean[0], "partner_frac_ge_4kb": frac[0]})
    return pl.DataFrame(rows)


def cutoff_partner_mean(hist: np.ndarray, edges: np.ndarray, cutoffs) -> np.ndarray:
    """The partner_mean column of cutoff_summary without polars (for forked null workers)."""
    return np.array([hist_summary(hist[int(np.flatnonzero(edges == c)[0]):].sum(axis=0))[1][0] for c in cutoffs])


def long_counts(values: np.ndarray, cutoff: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Number (or summed weight) of arrays with 0, 1, 2 and 3+ segments >= cutoff (missing positions count as not long)."""
    c = (np.nan_to_num(values, nan=0.0) >= cutoff).sum(axis=1)
    return np.bincount(np.minimum(c, 3), weights=weights, minlength=4)


def oe_table(observed: np.ndarray, nulls: list[np.ndarray], label: str) -> pl.DataFrame:
    """Observed vs null mean and 2.5/97.5% band per count class (0, 1, 2, 3+)."""
    z = np.array(nulls)
    lo, hi = np.percentile(z, 2.5, axis=0), np.percentile(z, 97.5, axis=0)
    return pl.DataFrame({"null": label, "n_long": ["0", "1", "2", "3+"], "observed": observed,
                         "expected": z.mean(axis=0), "lo": lo, "hi": hi,
                         "obs_over_exp": observed / np.maximum(z.mean(axis=0), 1e-300),
                         "outside": (observed < lo) | (observed > hi)})


def position_table(values: np.ndarray) -> pl.DataFrame:
    """Per position: n, median, fraction >= 4 kb."""
    return pl.DataFrame({"pos": np.arange(values.shape[1]), "n": np.full(values.shape[1], values.shape[0]),
                         "median": np.median(values, axis=0), "frac_ge_4kb": (values >= LONG_BP).mean(axis=0)})


def position_pairs(pos_table: pl.DataFrame) -> pl.DataFrame:
    """For every pair of positions a < b, the difference (a minus b) in median length and
    in the fraction at or above 4 kb."""
    t = pos_table.select("pos", "median", "frac_ge_4kb")
    return (t.join(t, how="cross", suffix="_b").filter(pl.col("pos") < pl.col("pos_b"))
            .select(pl.col("pos").alias("pos_a"), "pos_b", pl.col("median").alias("median_a"), "median_b",
                    (pl.col("median") - pl.col("median_b")).alias("median_diff"),
                    pl.col("frac_ge_4kb").alias("frac_ge_4kb_a"), "frac_ge_4kb_b",
                    (pl.col("frac_ge_4kb") - pl.col("frac_ge_4kb_b")).alias("frac_ge_4kb_diff"))
            .sort("pos_a", "pos_b"))


# ---------- M_T: independent positions selected on total length ----------

@dataclass
class MTModel:
    edges: np.ndarray          # length bin edges (bin_bp)
    g: np.ndarray              # (8, n_bins) pre-selection per-position probabilities
    t_edges: np.ndarray        # total-length bin edges
    s: np.ndarray              # selection weight per total-length bin, max 1
    iterations: int
    converged: bool
    max_rel_err: float


def _draw(g: np.ndarray, edges: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n draws per position from binned distributions g, uniform within the bin; returns (values, bins)."""
    k = g.shape[0]
    bins = np.empty((n, k), dtype=np.int64)
    for p in range(k):
        cdf = np.cumsum(g[p])
        bins[:, p] = np.minimum(np.searchsorted(cdf / cdf[-1], rng.random(n), side="right"), g.shape[1] - 1)
    width = np.diff(edges)
    vals = edges[:-1][bins] + rng.random((n, k)) * width[bins]
    return vals, bins


def count_edges(x: np.ndarray, width: float, min_count: int) -> np.ndarray:
    """Bin edges of the given width from 0, with neighbouring bins merged until each holds at least
    min_count values (sparse tails otherwise make the fitted ratios pure noise)."""
    fine = np.arange(0, x.max() + 2 * width, width, dtype=np.float64)
    counts = np.histogram(x, fine)[0]
    edges, acc = [0.0], 0
    for right, c in zip(fine[1:], counts):
        acc += c
        if acc >= min_count:
            edges.append(right)
            acc = 0
    if acc:
        edges[-1] = fine[-1]
    return np.array(edges if len(edges) > 1 else [0.0, fine[-1]])


def fit_mt(values: np.ndarray, rng: np.random.Generator, bin_bp: int = 50, n_sim: int = 10_000_000,
           max_iter: int = 30, tol: float = 0.01, min_count: int = 100) -> MTModel:
    """Fit g (per position) and s (total length) so that independent draws from g, kept with
    probability s(T), reproduce the observed per-position and total-length distributions.

    Iterative proportional fitting on one fixed set of n_sim draws from the observed marginals g0,
    reweighted by prod_k g_k/g0_k times s(T) (importance weights), so each iteration is exact for
    those draws. Bins are bin_bp wide, merged in the tails so each holds >= min_count observed values;
    converged when every bin is within tol (relative).
    """
    n, k = values.shape
    edges = count_edges(values.ravel(), bin_bp, min_count)
    t_obs = values.sum(axis=1)
    t_edges = count_edges(t_obs, bin_bp, min_count)
    nb, nt = len(edges) - 1, len(t_edges) - 1
    obs_bins = np.clip(np.digitize(values, edges) - 1, 0, nb - 1)
    obs_g = np.stack([np.bincount(obs_bins[:, p], minlength=nb) for p in range(k)]).astype(np.float64)
    obs_t = np.bincount(np.clip(np.digitize(t_obs, t_edges) - 1, 0, nt - 1), minlength=nt).astype(np.float64)
    g0 = obs_g / obs_g.sum(axis=1, keepdims=True)
    sim, sim_bins = _draw(g0, edges, n_sim, rng)
    sim_t = np.clip(np.digitize(sim.sum(axis=1), t_edges) - 1, 0, nt - 1)
    log_ratio = np.zeros((k, nb))          # log(g / g0)
    s = np.ones(nt)
    err, it = np.inf, 0
    for it in range(1, max_iter + 1):
        w = s[sim_t] * np.exp(sum(log_ratio[p][sim_bins[:, p]] for p in range(k)))
        w_t = np.bincount(sim_t, weights=w, minlength=nt)
        w_t *= n / w_t.sum()
        s = np.where(w_t > 0, s * obs_t / np.where(w_t > 0, w_t, 1), s)
        s /= s.max()
        w = s[sim_t] * np.exp(sum(log_ratio[p][sim_bins[:, p]] for p in range(k)))
        errs = []
        for p in range(k):
            w_p = np.bincount(sim_bins[:, p], weights=w, minlength=nb)
            w_p *= n / w_p.sum()
            good = w_p > 0
            log_ratio[p][good] += np.log(np.where(obs_g[p][good] > 0, obs_g[p][good], 1e-12) / w_p[good])
            m = obs_g[p] >= min_count / 8
            errs.append(np.abs(w_p[m] / obs_g[p][m] - 1).max() if m.any() else 0)
        w_t = np.bincount(sim_t, weights=w, minlength=nt)
        w_t *= n / w_t.sum()
        m = obs_t >= min_count
        errs.append(np.abs(w_t[m] / obs_t[m] - 1).max() if m.any() else 0)
        err = float(max(errs))
        if err < tol:
            break
    g = g0 * np.exp(log_ratio)
    g /= g.sum(axis=1, keepdims=True)
    return MTModel(edges, g, t_edges, s, it, err < tol, err)


def simulate_mt(model: MTModel, n_arrays: int, rng: np.random.Generator) -> np.ndarray:
    """n_arrays arrays drawn from g and kept with probability s(T) (rejection sampling)."""
    out, have = [], 0
    nt = len(model.t_edges) - 1
    while have < n_arrays:
        vals, _ = _draw(model.g, model.edges, max(2 * (n_arrays - have), 1000), rng)
        tb = np.clip(np.digitize(vals.sum(axis=1), model.t_edges) - 1, 0, nt - 1)
        vals = vals[rng.random(len(vals)) < model.s[tb]]
        out.append(vals)
        have += len(vals)
    return np.concatenate(out)[:n_arrays]


# ---------- adjacency ----------

def lag_profile(values: np.ndarray) -> np.ndarray:
    """Mean Spearman correlation between positions at lag 1..7 (within-position quantile normalisation)."""
    ranks = np.argsort(np.argsort(values, axis=0), axis=0).astype(np.float64)
    c = np.corrcoef(ranks.T)
    k = values.shape[1]
    return np.array([np.mean([c[i, i + lag] for i in range(k - lag)]) for lag in range(1, k)])


def d_statistic(profile: np.ndarray) -> float:
    """Lag-1 correlation minus the mean correlation at lags 2..7."""
    return float(profile[0] - profile[1:].mean())


# ---------- HiFi selection ----------

def w_hifi_empirical(t_hifi: np.ndarray, t_fail: np.ndarray, edges: np.ndarray,
                     hifi_weight: float = 1.0, fail_weight: float = 1.0) -> pl.DataFrame:
    """HiFi / (HiFi + fail) per read-length bin; overestimates w(T) where zero-pass ZMWs are common (long T).

    hifi_weight, fail_weight: 1 / the probability that a read of the cell is in each sample. A pilot holds a
    smaller share of the cell's HiFi reads than of its fail reads, and unweighted counts would give the
    share of the two samples, not the cell's HiFi share. n_hifi and n_fail are the unweighted counts.
    """
    h = np.histogram(t_hifi, edges)[0]
    f = np.histogram(t_fail, edges)[0]
    hw, tot = h * hifi_weight, h * hifi_weight + f * fail_weight
    return pl.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "n_hifi": h, "n_fail": f,
                         "w": np.where(tot > 0, hw / np.where(tot > 0, tot, 1), np.nan)})


def pass_rq_table(reads: pl.DataFrame) -> pl.DataFrame:
    """Share of reads with rq >= 0.99 per pass count np (HiFi and fail reads together).

    An optional weight column (1 / the probability that the read is sampled from the cell) weighs each
    read, so HiFi and fail reads sampled at different fractions give the cell's share; n is unweighted.
    """
    if "weight" not in reads.columns:
        reads = reads.with_columns(pl.lit(1.0).alias("weight"))
    hifi = (pl.col("rq") >= 0.99).cast(pl.Float64) * pl.col("weight")
    return (reads.group_by("np").agg((hifi.sum() / pl.col("weight").sum()).alias("frac_rq_ge_099"), pl.len().alias("n"))
            .sort("np"))


def w_hifi_model(pol_length: np.ndarray, pass_rq: pl.DataFrame, t_grid: np.ndarray, max_np: int | None = None) -> np.ndarray:
    """w(T) = mean over ZMWs of P(rq >= 0.99 | np), with np = floor(pol_length / (T + hairpin)) - 1.

    P(rq >= 0.99 | np) comes from pass_rq for pass counts up to max_np (default: the largest np in pass_rq;
    stage 06 passes the largest HiFi np, so a sparse fail-only tail above it is not used). A pass count
    missing from the table takes the value of the next lower one present, a pass count above max_np the
    value at max_np, and one below the smallest np in the table 0.
    """
    q = dict(zip(pass_rq["np"].to_list(), pass_rq["frac_rq_ge_099"].to_list()))
    top = max(q) if max_np is None else int(max_np)
    lut, last = np.zeros(top + 1), 0.0
    for p in range(top + 1):
        last = q.get(p, last)
        lut[p] = last
    L = pol_length[pol_length > 0].astype(np.float64)
    out = []
    for T in t_grid:
        npass = np.clip(np.floor(L / (T + HAIRPIN)) - 1, 0, top).astype(np.int64)
        out.append(lut[npass].mean())
    return np.array(out)


def pol_length_usable(zmw: pl.DataFrame, hifi_index: pl.DataFrame) -> tuple[bool, float]:
    """Usable if >= 90% of HiFi ZMWs have pol_length >= 0.8 * np * read_len."""
    j = hifi_index.join(zmw.select("zm", "pol_length"), on="zm", how="inner")
    if j.height == 0:
        return False, 0.0
    frac = float((j["pol_length"] >= 0.8 * j["np"] * j["read_len"]).mean())
    return frac >= 0.9, frac


def _independence_parts(values: np.ndarray, eligible: np.ndarray, w: np.ndarray, edges: np.ndarray, cutoffs):
    """Weighted observed and independence-null partner sums per index bin, and expected arrays by long count."""
    n, k = values.shape
    nb = len(edges) - 1
    ib = np.digitize(values, edges) - 1
    ok = eligible & (ib >= 0) & (ib < nb)
    rowsum = values.sum(axis=1)
    mu = (w[:, None] * values).sum(axis=0) / max(w.sum(), 1e-300)      # weighted mean of each position
    num_obs, den, num_null = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for p in range(k):
        r = ok[:, p]
        b, wr = ib[r, p], w[r]
        num_obs += np.bincount(b, weights=wr * (rowsum[r] - values[r, p]), minlength=nb)
        cnt = np.bincount(b, weights=wr, minlength=nb)
        den += cnt * (k - 1)
        num_null += cnt * (mu.sum() - mu[p])
    total = w.sum()
    expected = []
    for c in cutoffs:
        dist = np.array([1.0])
        for pk in (w[:, None] * (values >= c)).sum(axis=0) / max(total, 1e-300):
            dist = np.convolve(dist, [1 - pk, pk])                       # number of long positions, independent
        expected.append(total * np.array([dist[0], dist[1], dist[2], dist[3:].sum()]))
    return num_obs, den, num_null, np.array(expected)


def ipw_independence_null(values: np.ndarray, eligible: np.ndarray, weights: np.ndarray, edges: np.ndarray, cutoffs,
                          rng: np.random.Generator, n_boot: int = 200) -> dict:
    """Partner means of index segments against independent positions in the population the IPW weights stand for.

    values: complete arrays (no missing position). Under independence a partner at position j has the weighted mean
    mu_j of position j over all arrays, whatever the index, so the null for index segments of a length bin is the mean
    of mu_j over j != k, averaged over the weighted index segments of that bin at each position k. A null P weighted by
    1/w(T) is not this null: it draws arrays from the selected marginals and then reweights them as if they had been
    selected, which shifts the null by tens of bp in either direction. Observed means are exact here, like the null,
    not taken from the 10 bp partner histograms. The expected weighted number of arrays with 0, 1, 2 and 3+ segments at
    or above each cutoff follows from the weighted per-position shares of long segments. Bootstrap over arrays (weights
    fixed, n_boot resamples) gives 95% intervals for the null and for observed minus null.

    Returns obs_bin, null_bin, null_bin_lo, null_bin_hi, diff_bin_lo, diff_bin_hi (per index bin), obs_cut, null_cut,
    null_cut_lo, null_cut_hi, diff_cut_lo, diff_cut_hi (index at or above each cutoff), long_expected
    (cutoffs x 4) and long_boot (n_boot x cutoffs x 4).
    """
    starts = [int(np.flatnonzero(edges == c)[0]) for c in cutoffs]

    def means(parts):
        num_obs, den, num_null, expected = parts
        with np.errstate(invalid="ignore", divide="ignore"):
            ob, nu = num_obs / den, num_null / den
            oc = np.array([num_obs[i:].sum() / den[i:].sum() for i in starts])
            nc = np.array([num_null[i:].sum() / den[i:].sum() for i in starts])
        return ob, nu, oc, nc, expected

    ob, nu, oc, nc, expected = means(_independence_parts(values, eligible, weights, edges, cutoffs))
    boots = [means(_independence_parts(values[i], eligible[i], weights[i], edges, cutoffs))
             for i in (rng.integers(0, len(values), len(values)) for _ in range(n_boot))]

    def pct(z):
        z = np.array(z)
        have = ~np.all(np.isnan(z), axis=0)
        lo, hi = np.full(z.shape[1:], np.nan), np.full(z.shape[1:], np.nan)
        if have.any():
            lo[have], hi[have] = np.nanpercentile(z[:, have], 2.5, axis=0), np.nanpercentile(z[:, have], 97.5, axis=0)
        return lo, hi

    null_bin_lo, null_bin_hi = pct([b[1] for b in boots])
    diff_bin_lo, diff_bin_hi = pct([b[0] - b[1] for b in boots])
    null_cut_lo, null_cut_hi = pct([b[3] for b in boots])
    diff_cut_lo, diff_cut_hi = pct([b[2] - b[3] for b in boots])
    return {"obs_bin": ob, "null_bin": nu, "null_bin_lo": null_bin_lo, "null_bin_hi": null_bin_hi,
            "diff_bin_lo": diff_bin_lo, "diff_bin_hi": diff_bin_hi, "obs_cut": oc, "null_cut": nc,
            "null_cut_lo": null_cut_lo, "null_cut_hi": null_cut_hi, "diff_cut_lo": diff_cut_lo, "diff_cut_hi": diff_cut_hi,
            "long_expected": expected, "long_boot": np.array([b[4] for b in boots])}


def ipw_weights(total_len: np.ndarray, w_at: Callable[[np.ndarray], np.ndarray],
                clip_q: float = 0.99, min_w: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """(weights, kept): weight 1/w(T) for arrays with w(T) >= min_w, clipped at the clip_q quantile."""
    w = w_at(total_len)
    kept = w >= min_w
    weights = np.where(kept, 1.0 / np.maximum(w, min_w), 0.0)
    if kept.any():
        weights = np.minimum(weights, np.quantile(weights[kept], clip_q))
    return weights, kept


# ---------- truncation ----------

def truncation_table(break_values: pl.DataFrame, full_values: np.ndarray, rng: np.random.Generator,
                     n_boot: int = 200) -> pl.DataFrame:
    """Truncation table. break_values: columns pos, value (the segment next to the missing end of each
    truncated array). full_values: complete arrays. Per position: median ratio and ratio of fractions
    >= 4 kb (break side over complete arrays at the same position), with bootstrap 95% CIs."""
    rows = []
    for (pos,), grp in break_values.sort(["pos", "value"]).group_by("pos", maintain_order=True):
        b = grp["value"].to_numpy().astype(np.float64)
        f = full_values[:, pos]
        med, fr = [], []
        for _ in range(n_boot):
            bb, ff = rng.choice(b, len(b)), rng.choice(f, len(f))
            med.append(np.median(bb) / np.median(ff))
            fr.append((bb >= LONG_BP).mean() / max((ff >= LONG_BP).mean(), 1e-9))
        rows.append({"pos": pos, "n_break": len(b), "n_complete": len(f),
                     "median_ratio": np.median(b) / np.median(f),
                     "median_ratio_lo": np.percentile(med, 2.5), "median_ratio_hi": np.percentile(med, 97.5),
                     "frac_ge_4kb_break": (b >= LONG_BP).mean(), "frac_ge_4kb_complete": (f >= LONG_BP).mean(),
                     "frac_ratio_lo": np.percentile(fr, 2.5), "frac_ratio_hi": np.percentile(fr, 97.5)})
    return pl.DataFrame(rows).sort("pos") if rows else pl.DataFrame()
