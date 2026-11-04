#!/usr/bin/env python3
"""Synthetic benchmark of adapter-chain classification, in particular mirror calls.

The synthetic gate (gates.py) scores adapter and primer detection on complete arrays only. Here arrays of known
structure are built from clean real S-reads (gates._clean_sreads, mutated with gates.mutate) and classified with the
package's own annotate_read, adapter_chain, chain_flags and primary_pattern:

  complete            A B ... H Q
  tso_tso_mirror_X    Q' H' ... X' [TSO-TSO] X ... H Q        X = B..Q (Q: a lone TSO-TSO molecule between Q' and Q)
  r1_r1_mirror_X      A B ... X [R1-R1] X' ... B' A'          X = A..H
  truncated_no_A      [part of a molecule] B ... Q
  truncated_no_Q      A ... H [part of a molecule]
  adapter_destroyed   complete, one of B..H mutated far beyond the 2-edit threshold (a missed adapter)
  decoy_motif         complete, CTCCTCCTCC and GGAGGAGGAGG (D-like) inserted inside every molecule
  opposite_join       a partial array A..X joined to the reverse complement of a partial array Y..Q, X != Y,
                      so adapters change orientation without a mirror junction (same letter on both sides)

A TSO-TSO molecule is built from two clean S-reads: the half holding the TSO end of the first, reverse complemented,
followed by the half holding the TSO end of the second, so a TSO sits at both ends on opposite strands. R1-R1 the
same with the Read1 ends. Arms are clean S-reads; pieces of the reverse arm are reverse complemented with it.
Writes mirror_benchmark.tsv (calls per error rate and truth) and mirror_benchmark_summary.tsv (mirror recall and
precision per error rate) into --out.
Usage: study/analyses/chain_benchmark.py --sreads <S-read BAM> --out <dir> [--n 200] [--seed 1] [--max-sreads 20000]
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from kinnex_qc.annotate import FAIL_PATTERNS, HIFI_PATTERNS, adapter_chain, annotate_read, chain_flags, primary_pattern
from kinnex_qc.classify import classify_segment
from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import MAS, MAS_ORDER, revcomp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from gates import _clean_sreads, mutate  # noqa: E402

RATES = (0.001, 0.005, 0.01, 0.05)          # 0.05 is scored with the fail-read thresholds, as stage 02 does for fail reads
DECOYS = ("CTCCTCCTCC", "GGAGGAGGAGG")


def adapter(token: str, rate: float, rng: random.Random) -> str:
    seq = MAS[token.rstrip("'")]
    return mutate(revcomp(seq) if token.endswith("'") else seq, rate, rng)


def assemble(tokens: list[str], pieces: list[str], rate: float, rng: random.Random, head: str = "", tail: str = "") -> str:
    """tokens[i] adapter, then pieces[i] (len(pieces) == len(tokens) - 1), each part mutated separately."""
    out = [mutate(head, rate, rng)] if head else []
    for i, t in enumerate(tokens):
        out.append(adapter(t, rate, rng))
        if i < len(pieces):
            out.append(mutate(pieces[i], rate, rng))
    if tail:
        out.append(mutate(tail, rate, rng))
    return "".join(out)


def end_half(seq: str, hits, kind: str) -> str | None:
    """The half of a clean S-read that holds its primer of kind (TSO or R1), oriented with that primer at the end."""
    h = next((x for x in hits if x.kind == kind), None)
    if h is None:
        return None
    mid = len(seq) // 2
    return seq[mid:] if h.start >= mid else revcomp(seq[:mid])


def double_end(a, b, kind: str) -> str | None:
    """A molecule with a primer of kind at both ends, on opposite strands."""
    x, y = end_half(*a, kind), end_half(*b, kind)
    return None if x is None or y is None else revcomp(x) + y


def build(cls: str, pool, rate: float, rng: random.Random) -> str:
    def pick(n):
        return [p[0] for p in rng.sample(pool, n)]

    if cls == "complete":
        return assemble(MAS_ORDER, pick(8), rate, rng)
    if cls.startswith("tso_tso_mirror_") or cls.startswith("r1_r1_mirror_"):
        x = cls[-1]
        kind = "TSO" if cls.startswith("tso") else "R1"
        mol = None
        while mol is None:
            mol = double_end(*rng.sample(pool, 2), kind)
        i = MAS_ORDER.index(x)
        if kind == "TSO":           # Q' ... X' [mol] X ... Q: both arms run from X to Q
            arm = MAS_ORDER[i:]
            left, right = pick(len(arm) - 1), pick(len(arm) - 1)
            tokens = [t + "'" for t in reversed(arm)] + arm
            pieces = [revcomp(p) for p in reversed(left)] + [mol] + right
        else:                       # A ... X [mol] X' ... A': both arms run from A to X
            arm = MAS_ORDER[:i + 1]
            left, right = pick(len(arm) - 1), pick(len(arm) - 1)
            tokens = arm + [t + "'" for t in reversed(arm)]
            pieces = left + [mol] + [revcomp(p) for p in reversed(right)]
        return assemble(tokens, pieces, rate, rng)
    if cls == "truncated_no_A":
        p = pick(8)
        return assemble(MAS_ORDER[1:], p[1:], rate, rng, head=p[0][len(p[0]) // 2:])
    if cls == "truncated_no_Q":
        p = pick(8)
        return assemble(MAS_ORDER[:-1], p[:-1], rate, rng, tail=p[-1][:len(p[-1]) // 2])
    if cls == "adapter_destroyed":
        k = rng.randrange(1, 8)
        seq = assemble(MAS_ORDER[:k], pick(k), rate, rng)
        bad = "".join(rng.choice("ACGT".replace(c, "")) if j % 3 == 0 else c for j, c in enumerate(MAS[MAS_ORDER[k]]))
        rest = pick(8 - k)
        return seq + bad + assemble(MAS_ORDER[k + 1:], rest[1:], rate, rng, head=rest[0])
    if cls == "decoy_motif":
        pieces = []
        for s in pick(8):
            m = len(s) // 2
            pieces.append(s[:m // 2] + DECOYS[0] + s[m // 2:m + m // 2] + DECOYS[1] + s[m + m // 2:])
        return assemble(MAS_ORDER, pieces, rate, rng)
    if cls == "opposite_join":
        x, y = rng.sample(range(2, 7), 2)            # partial arrays A..X and Y..Q with X != Y
        first = assemble(MAS_ORDER[:x + 1], pick(x), rate, rng)
        second = assemble(MAS_ORDER[y:], pick(8 - y), rate, rng)
        return first + mutate(pick(1)[0], rate, rng) + revcomp(second)
    raise ValueError(cls)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sreads", required=True, help="skera S-read BAM, the source of clean molecules")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=200, help="arrays per error rate and truth")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-sreads", type=int, default=20_000, help="S-reads read from the start of the BAM")
    args = ap.parse_args(argv)
    seqs = [r.query_sequence for r in iter_bam(args.sreads, max_records=args.max_sreads)]
    pool = _clean_sreads(seqs)
    print(f"{len(pool):,} clean S-reads of {len(seqs):,}", file=sys.stderr)
    classes = (["complete"] + [f"tso_tso_mirror_{x}" for x in MAS_ORDER[1:]] + [f"r1_r1_mirror_{x}" for x in MAS_ORDER[:-1]]
               + ["truncated_no_A", "truncated_no_Q", "adapter_destroyed", "decoy_motif", "opposite_join"])
    rng = random.Random(args.seed)
    rows = []
    for rate in RATES:
        patterns = FAIL_PATTERNS if rate >= 0.05 else HIFI_PATTERNS
        for cls in classes:
            calls = Counter()
            junction_class = Counter()
            for _ in range(args.n):
                seq = build(cls, pool, rate, rng)
                hits = annotate_read(seq, patterns)
                tokens, _ = adapter_chain(hits)
                calls[primary_pattern(chain_flags(tokens))] += 1
                if "mirror_" in cls:          # class of the molecule between the two junction adapters
                    mas = [h for h in hits if h.kind == "MAS"]
                    for a, b in zip(mas, mas[1:]):
                        if a.name == b.name and a.strand != b.strand:
                            junction_class[classify_segment(seq[a.start:b.end]).end_class] += 1
                            break
            rows.append({"error_rate": rate, "truth": cls, "n": args.n, **{f"called_{k}": v for k, v in sorted(calls.items())},
                         "junction_molecule_classes": json.dumps(dict(junction_class))})
            print(rate, cls, dict(calls), file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pl.DataFrame(rows).fill_null(0)
    t.write_csv(out / "mirror_benchmark.tsv", separator="\t")
    m = t.with_columns(pl.col("truth").str.contains("mirror").alias("is_mirror"))
    summ = []
    for rate in RATES:
        r = m.filter(pl.col("error_rate") == rate)
        tp = r.filter(pl.col("is_mirror"))["called_mirror"].sum()
        fp = r.filter(~pl.col("is_mirror"))["called_mirror"].sum() if "called_mirror" in r.columns else 0
        summ.append({"error_rate": rate, "mirror_recall": tp / r.filter(pl.col("is_mirror"))["n"].sum(),
                     "mirror_precision": tp / max(tp + fp, 1), "non_mirror_called_mirror": fp,
                     "non_mirror_reads": r.filter(~pl.col("is_mirror"))["n"].sum()})
    pl.DataFrame(summ).write_csv(out / "mirror_benchmark_summary.tsv", separator="\t")
    print(pl.DataFrame(summ))


if __name__ == "__main__":
    main()
