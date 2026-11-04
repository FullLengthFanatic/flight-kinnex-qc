#!/bin/bash
# Stage 04: lima and isoseq refine on the S-reads of one sample.
#   lima/<sample>.isoseq.*     lima --isoseq --peek-guess, as used for the existing fltnc BAMs
#   lima/<sample>.pairs.bam    plain lima with the Iso-Seq end thresholds: bc tag = primer pair per S-read
#   refine/<sample>.asrun.bam  refine as run for the existing fltnc BAMs (no polyA requirement)
#   refine/<sample>.polyA_N.bam  refine --require-polya --min-polya-length N, N in 5 8 12 20
# At fraction 1 lima reads the sample's segmented BAM itself. Below 1, the S-reads of the hash sample go to a temporary
# BAM that is trashed at the end: lima 2.13.0 and isoseq refine 4.3.0 do not read a pipe (observed: both fail on
# /dev/stdin with "unsupported extension", and on "-" and /dev/fd/N).
# Usage, from the repository root:
#   study/pipeline/s04_lima_refine.sh --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--threads N]
#       [--pbtools-env NAME|PATH]
# lima and isoseq run in the conda environment --pbtools-env (a name, or a path).
# Defaults come from study/study.toml.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY="$REPO/.venv/bin/python"
PRIMERS="$REPO/resources/primers_FlashBD.fasta"

for arg in "$@"; do
  if [[ "$arg" == -h || "$arg" == --help ]]; then sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
done

# The options, and from them: output directory, segmented BAM, conda environment, temporary directory (empty at fraction 1).
IFS=$'\t' read -r SAMPLE FRACTION THREADS OUT SEGMENTED PB TMP < <("$PY" - "$HERE" "$@" <<'EOF'
import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from config import load_config, load_samples, out_dir

config = load_config()
ap = argparse.ArgumentParser(prog="s04_lima_refine.sh")
ap.add_argument("--sample", required=True)
ap.add_argument("--samples", default=config["samples"], help="samples table")
ap.add_argument("--out-root", default=config["out_root"], help="output root")
ap.add_argument("--fraction", type=float, default=config["fraction"], help="hash sample of the cell's ZMWs; 1 reads them all")
ap.add_argument("--threads", type=int, default=config["threads"], help="lima and isoseq threads")
ap.add_argument("--pbtools-env", default=config["pbtools_env"], help="conda environment with lima and isoseq (a name, or a path)")
with contextlib.redirect_stdout(sys.stderr):        # stdout carries the values only
    args = ap.parse_args(sys.argv[2:])
sample = load_samples(args.samples)[args.sample]
out = out_dir(args.out_root, sample.sample, args.fraction)
tmp = Path(args.out_root) / "tmp" / f"{sample.sample}_{out.parent.name}" if args.fraction < 1 else ""   # e.g. BD144_ND_hash_0.18
print(sample.sample, args.fraction, args.threads, out, sample.segmented_bam, args.pbtools_env, tmp, sep="\t")
EOF
)
if [[ "$PB" == */* ]]; then RUN=(conda run --no-capture-output -p "$PB"); else RUN=(conda run --no-capture-output -n "$PB"); fi
mkdir -p "$OUT/lima" "$OUT/refine"

SEG=$SEGMENTED
if [[ -n "$TMP" ]]; then
  mkdir -p "$TMP"
  trap 'gio trash "$TMP"' EXIT
  SEG="$TMP/segmented.bam"
  "$PY" - "$SEGMENTED" "$FRACTION" "$SEG" <<'EOF'
import sys

import pysam

from kinnex_qc.io import iter_bam
from kinnex_qc.sampling import in_hash_sample

src, fraction, dest = sys.argv[1], float(sys.argv[2]), sys.argv[3]
records = iter_bam(src, keep=lambda zm: in_hash_sample(zm, fraction))
first = next(records)
with pysam.AlignmentFile(dest, "wb", header=first.header) as out:   # keeps the read groups lima needs
    out.write(first)
    for r in records:
        out.write(r)
EOF
fi

"${RUN[@]}" lima --isoseq --peek-guess -j "$THREADS" --log-level WARN "$SEG" "$PRIMERS" "$OUT/lima/$SAMPLE.isoseq.bam"
"${RUN[@]}" lima --min-end-score 50 --min-ref-span 0.75 -j "$THREADS" --log-level WARN "$SEG" "$PRIMERS" "$OUT/lima/$SAMPLE.pairs.bam"
FL="$OUT/lima/$SAMPLE.isoseq.FlashBD_5p--FlashBD_3p.bam"
"${RUN[@]}" isoseq refine -j "$THREADS" --log-level WARN "$FL" "$PRIMERS" "$OUT/refine/$SAMPLE.asrun.bam"
for n in 5 8 12 20; do
  "${RUN[@]}" isoseq refine -j "$THREADS" --log-level WARN --require-polya --min-polya-length "$n" \
    "$FL" "$PRIMERS" "$OUT/refine/$SAMPLE.polyA_$n.bam"
done
echo "{\"sample\": \"$SAMPLE\", \"fraction\": $FRACTION, \"primers\": \"$PRIMERS\"}" > "$OUT/lima_refine.done.json"
