#!/usr/bin/env python3
"""Validation gates: each one writes <out>/gates/gate_<name>.json with its metrics, verdict and threshold.

  synthetic      annotator recall and precision on synthetic arrays built from the first 3,000 S-reads of the
                 sample's segmented BAM (hard gate)
  skera          how often skera's S-read boundaries agree with our MAS adapter calls (a measurement, never fails)
  lima           our end classes against lima's primer pairs, and the mapped fltnc S-reads inside lima's pass set
                 (hard gate)
  adapter_index  skera's dl/dr index k is adapter MAS_ORDER[k] at each S-read boundary (hard gate)

skera and adapter_index read 1 in N HiFi ZMWs (sampling.keep_zmw_array, N = --fail-subsample, the modulo stage 02
subsamples fail reads with). synthetic reads only the segmented BAM; the others read stage 02, 03 and 05 outputs.
Exits 1 when a hard gate that ran failed, after running every gate asked for: a verdict failure (a hard gate below
its threshold) still lets the other gates run, while an exception stops the stage. The runner runs the skera gate as
its own step.
Usage, from the repository root:
  study/pipeline/gates.py --sample <name> [--gate synthetic|skera|lima|adapter_index|all] [--samples PATH]
      [--out-root DIR] [--fraction F] [--fail-subsample N] [--processes N]
Defaults come from study/study.toml.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

from config import load_config, load_samples, out_dir
from kinnex_qc.annotate import annotate_read
from kinnex_qc.classify import classify_segment
from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import FAIL_MAX_EDITS, MAS, MAS_ORDER, all_patterns, primer_patterns, revcomp
from kinnex_qc.sampling import keep_zmw_array
from kinnex_qc.scan import Hit, find_all
from kinnex_qc.skera import adapters_inside_sreads, false_cuts

HIFI_RATES = (0.001, 0.005, 0.01)
FAIL_RATES = (0.05, 0.08, 0.12)
LIMA_MAP = {"proper": "5p--3p", "tso_tso": "5p--5p", "r1_r1": "3p--3p"}   # every other class maps to "none"
N_SREADS = 3000
GATES = ("synthetic", "skera", "lima", "adapter_index")
# The threshold texts are v10's, verbatim, except the skera one, which no longer cites an internal reference.
THRESHOLDS = {
    "synthetic": {"hifi_rates": "MAS/TSO/R1 recall >= 0.99 and precision >= 0.999", "fused_internal_recall": ">= 0.99"},
    "skera": {"agreement": "reported only"},
    "lima": {"class_pair_agreement": ">= 0.98", "fltnc_in_lima_pass": ">= 0.995"},
    "adapter_index": {"agreement": ">= 0.99"},
}


def write_gate(out: Path, name: str, passed: bool, metrics: dict, threshold: dict) -> Path:
    path = out / "gates" / f"gate_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gate": name, "passed": bool(passed), "metrics": metrics, "threshold": threshold}, indent=1))
    return path


# ---------- synthetic ----------

def mutate(seq: str, rate: float, rng: random.Random) -> str:
    """Substitutions, deletions and insertions, each at rate/3 per base."""
    out = []
    for c in seq:
        r = rng.random()
        if r < rate / 3:
            out.append(rng.choice("ACGT".replace(c, "")))
        elif r < 2 * rate / 3:
            continue
        elif r < rate:
            out.append(c + rng.choice("ACGT"))
        else:
            out.append(c)
    return "".join(out)


def _clean_sreads(sreads: list[str]) -> list[tuple[str, list[Hit]]]:
    """Proper S-reads whose only TSO/R1 hits anywhere are their two end primers (these are the truth).

    Checked with find_all over the whole S-read, so tandem TSO concatemers inside an end window
    (real sequence the truth could not list) are excluded.
    """
    keep = []
    primers = primer_patterns()
    for s in sreads:
        call = classify_segment(s)
        if call.end_class != "proper" or call.internal_hits:
            continue
        found = [h for p in primers for strand in "+-" for h in find_all(s, p, strand, 0, len(s))]
        if len(found) == 2:
            keep.append((s, found))
    return keep


def build_array(pieces: list[tuple[str, list[Hit]]], rate: float, rng: random.Random) -> tuple[str, list[tuple]]:
    """A..Q with the pieces between them, each part mutated separately; truth = (name, strand, start, end)."""
    seq, truth = "", []
    for i, name in enumerate(MAS_ORDER):
        a = mutate(MAS[name], rate, rng)
        truth.append((name, "+", len(seq), len(seq) + len(a)))
        seq += a
        if i == len(pieces):
            break
        s, ends = pieces[i]
        cuts = sorted({0, len(s), *[x for h in ends for x in (h.start, h.end)]})
        parts = {(a0, b0): mutate(s[a0:b0], rate, rng) for a0, b0 in zip(cuts, cuts[1:])}
        offset = len(seq)
        pos = {}
        for (a0, b0), p in parts.items():
            pos[(a0, b0)] = (offset, offset + len(p))
            offset += len(p)
        for h in ends:
            truth.append((h.name, h.strand, *pos[(h.start, h.end)]))
        seq += "".join(parts.values())
    return seq, truth


def _score(hits: list[Hit], truth: list[tuple], counts: dict) -> None:
    used = set()
    for name, strand, a, b in truth:
        kind = "MAS" if name in MAS else name
        c = counts.setdefault(kind, {"truth": 0, "found": 0, "called": 0, "correct": 0})
        c["truth"] += 1
        for j, h in enumerate(hits):
            if j not in used and h.name == name and h.strand == strand and h.start < b and a < h.end:
                c["found"] += 1
                used.add(j)
                break
    for j, h in enumerate(hits):
        c = counts.setdefault(h.kind, {"truth": 0, "found": 0, "called": 0, "correct": 0})
        c["called"] += 1
        c["correct"] += j in used


def synthetic_benchmark(sreads: list[str], seed: int = 1, n_arrays: int = 200, n_fused: int = 200,
                        fail_max_edits: dict[str, int] = FAIL_MAX_EDITS) -> dict:
    """Recall and precision of annotate_read per kind and error rate; internal recall for fused segments.

    HiFi rates use the HiFi thresholds, fail-read rates the fail_max_edits thresholds that stage 02 applies
    to fail reads. The arrays do not depend on the thresholds, so runs with the same seed score the same arrays.
    """
    rng = random.Random(seed)
    pool = _clean_sreads(sreads)
    fail_patterns = all_patterns(fail_max_edits)
    result = {"n_clean_sreads": len(pool), "fail_max_edits": dict(fail_max_edits), "rates": {}}
    for rate in HIFI_RATES + FAIL_RATES:
        patterns = fail_patterns if rate in FAIL_RATES else all_patterns()
        counts: dict = {}
        for _ in range(n_arrays):
            pieces = rng.sample(pool, 8)
            seq, truth = build_array(pieces, rate, rng)
            if rng.random() < 0.5:
                L = len(seq)
                seq = revcomp(seq)
                truth = [(n, "-" if s == "+" else "+", L - b, L - a) for n, s, a, b in truth]
            _score(annotate_read(seq, patterns), truth, counts)
        result["rates"][str(rate)] = {k: {"recall": v["found"] / v["truth"] if v["truth"] else None,
                                          "precision": v["correct"] / v["called"] if v["called"] else None,
                                          **v} for k, v in counts.items()}
    fused = {}
    for rate in HIFI_RATES:
        flagged = sum(classify_segment(mutate(a + b, rate, rng)).internal
                      for (a, _), (b, _) in (rng.sample(pool, 2) for _ in range(n_fused)))
        fused[str(rate)] = flagged / n_fused
    result["fused_internal_recall"] = fused
    return result


def synthetic_passed(result: dict) -> bool:
    ok = all(result["fused_internal_recall"][str(r)] >= 0.99 for r in HIFI_RATES)
    for r in HIFI_RATES:
        for kind in ("MAS", "TSO", "R1"):
            m = result["rates"][str(r)][kind]
            ok &= m["recall"] >= 0.99 and m["precision"] >= 0.999
    return ok


# ---------- skera, adapter_index, lima ----------

def _mas_hits(features: pl.DataFrame) -> dict[int, list[tuple]]:
    by = {}
    for zm, name, strand, a, b in features.filter(pl.col("kind") == "MAS").select("zm", "name", "strand", "start", "end").iter_rows():
        by.setdefault(zm, []).append((name, strand, a, b))
    return by


def skera_agreement(segments: pl.DataFrame, features: pl.DataFrame, passing: pl.Series, tol: int = 5) -> dict:
    """Share of skera-passing HiFi reads where every gap between consecutive S-reads holds an annotator
    MAS hit (within tol bp) and no annotator MAS hit lies inside an S-read (a measurement
    of skera's split accuracy, reported, not a hard gate).

    Compared: the ZMWs in passing with at least one of our HiFi MAS hits and at least one S-read. A read disagrees
    when it has a false cut (skera.false_cuts) or an adapter strictly more than tol bp inside an S-read
    (skera.adapters_inside_sreads, strict=True). segments needs zm, di, dr, qs and qe."""
    mas = features.filter((pl.col("source") == "hifi") & (pl.col("kind") == "MAS"))
    s = segments.filter(pl.col("zm").is_in(mas["zm"].unique()) & pl.col("zm").is_in(passing))
    compared = s["zm"].unique()
    disagree = pl.concat([false_cuts(s, mas, tol)["zm"], adapters_inside_sreads(s, mas, tol, strict=True)["zm"]]).unique()
    n = compared.len()
    agreed = int((~compared.is_in(disagree)).sum())
    return {"compared_reads": n, "agreeing_reads": agreed, "agreement": agreed / n if n else None}


def adapter_index_agreement(segments: pl.DataFrame, features: pl.DataFrame, tol: int = 5) -> dict:
    """For S-reads of annotated HiFi ZMWs: the MAS hit ending within tol bp of qs is MAS_ORDER[dl] and the
    one starting within tol bp of qe is MAS_ORDER[dr]."""
    mas = _mas_hits(features.filter(pl.col("source") == "hifi"))
    checked = agreed = 0
    for zm, dl, dr, qs, qe in segments.filter(pl.col("zm").is_in(pl.Series(list(mas), dtype=pl.UInt32))) \
            .select("zm", "dl", "dr", "qs", "qe").iter_rows():
        left = [n for n, _, a, b in mas[zm] if abs(b - qs) <= tol]
        right = [n for n, _, a, b in mas[zm] if abs(a - qe) <= tol]
        if left:
            checked += 1
            agreed += left[0] == MAS_ORDER[dl]
        if right:
            checked += 1
            agreed += right[0] == MAS_ORDER[dr]
    return {"checked_boundaries": checked, "agreeing": agreed, "agreement": agreed / checked if checked else None}


def lima_agreement(fate: pl.DataFrame) -> dict:
    """end_class vs plain-lima pair, and share of primary mapped fltnc S-reads inside the lima --isoseq pass set.
    The confusion rows come by count, largest first; rows tied on count by lima_pair, then end_class."""
    expected = fate["end_class"].replace_strict(LIMA_MAP, default="none")
    agree = (expected == fate["lima_pair"]).mean()
    mapped = fate.filter(pl.col("mapped_fltnc"))
    table = (fate.group_by("end_class", "lima_pair").len()
             .sort("len", "lima_pair", "end_class", descending=[True, False, False]))
    return {"class_pair_agreement": float(agree),
            "fltnc_in_lima_pass": float(mapped["lima_pass"].mean()) if mapped.height else None,
            "n_mapped_fltnc": mapped.height,
            "confusion": table.to_dicts()}


# ---------- the gates on a sample's files ----------

def hifi_subsample(out: Path, modulo: int) -> pl.Series:
    """The 1-in-modulo subsample of the ZMWs in hifi_index.parquet."""
    zms = pl.read_parquet(out / "hifi_index.parquet", columns=["zm"])["zm"]
    return zms.filter(pl.Series(keep_zmw_array(zms.to_numpy(), modulo)))


def mas_features(out: Path, sub: pl.Series) -> pl.DataFrame:
    return pl.scan_parquet(out / "read_features.parquet").filter(pl.col("kind") == "MAS", pl.col("zm").is_in(sub)).collect()


def gate_synthetic(sample) -> tuple[bool, dict]:
    sreads = [r.query_sequence for r in iter_bam(sample.segmented_bam, max_records=N_SREADS)]
    result = synthetic_benchmark(sreads)
    return synthetic_passed(result), result


def gate_skera(out: Path, modulo: int) -> tuple[bool, dict]:
    sub = hifi_subsample(out, modulo)
    reads = pl.read_parquet(out / "reads.parquet", columns=["zm", "passing"])
    m = skera_agreement(pl.read_parquet(out / "segments.parquet", columns=["zm", "di", "dr", "qs", "qe"]).filter(pl.col("zm").is_in(sub)),
                        mas_features(out, sub), reads.filter(pl.col("passing"))["zm"])
    return True, m


def gate_lima(out: Path) -> tuple[bool, dict]:
    m = lima_agreement(pl.read_parquet(out / "fate.parquet"))
    return m["class_pair_agreement"] >= 0.98 and m["fltnc_in_lima_pass"] is not None and m["fltnc_in_lima_pass"] >= 0.995, m


def gate_adapter_index(out: Path, modulo: int) -> tuple[bool, dict]:
    sub = hifi_subsample(out, modulo)
    m = adapter_index_agreement(pl.read_parquet(out / "segments.parquet", columns=["zm", "dl", "dr", "qs", "qe"]).filter(pl.col("zm").is_in(sub)),
                                mas_features(out, sub))
    return m["agreement"] is not None and m["agreement"] >= 0.99, m


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--gate", choices=(*GATES, "all"), default="all", help="the gate to run, or all four")
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="names the output directory of a hash sample run")
    ap.add_argument("--fail-subsample", type=int, default=config["fail_subsample"],
                    help="skera and adapter_index read 1 in N HiFi ZMWs")
    ap.add_argument("--processes", type=int, default=config["processes"], help="unused; accepted for a uniform CLI")
    args = ap.parse_args()

    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    runs = {"synthetic": lambda: gate_synthetic(sample),
            "skera": lambda: gate_skera(out, args.fail_subsample),
            "lima": lambda: gate_lima(out),
            "adapter_index": lambda: gate_adapter_index(out, args.fail_subsample)}
    failed = []
    for name in GATES if args.gate == "all" else (args.gate,):
        passed, metrics = runs[name]()
        path = write_gate(out, name, passed, metrics, THRESHOLDS[name])
        if name == "skera":
            print(f"skera agreement {metrics['agreement']}: {path}")
        else:
            print(f"gate {name} {'passed' if passed else 'FAILED'}: {path}")
            if not passed:
                failed.append(name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
