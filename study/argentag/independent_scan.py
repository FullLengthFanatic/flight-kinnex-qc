#!/usr/bin/env python3
"""Independent check of the ArgenTag mirror calls, written without the study's code (2026-10-05): it imports nothing
from kinnex_qc (only the study's config reader, for its default paths). On 5,000 random non-passing parent reads
(fixed seed), find the Kinnex adapters (edlib, up to 2 edits, both strands, kept only with a TSO or Read1 hit within
4 bp), call a read a mirror read when two consecutive adapters share a letter in opposite orientation, and compare per
read with the study's calls in full_reads/read_summary.parquet (annotate_non_passing.py). Prints agreement and a
mirror-share estimate.

Inputs, in the ArgenTag folder (--argentag-dir): data/segmented.non_passing.bam (shared by ArgenTag, not public) and
analysis/full_reads/read_summary.parquet.
Writes the printed lines, aggregate numbers only, to <out>/independent_scan.txt (--out defaults to
<argentag_dir>/analysis). The reads the two calls disagree on (ArgenTag ZMW ids, one row each: zm, my_mirror,
their_mirror, my_n_adapters) go to <argentag_dir>/analysis/independent_scan_disagreements.tsv, always inside the
ArgenTag folder, so they never land next to the published text.
Usage, from the repository root:
  study/argentag/independent_scan.py [--argentag-dir DIR] [--out DIR]
--argentag-dir defaults to argentag_dir of study/study.toml and study.local.toml.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pysam
import polars as pl
import edlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402


# Adapter sequences
MAS = {
    "A": "AGCTTACTTGTGAAGAT",
    "B": "ACTTGTAAGCTGTCTAT",
    "C": "ACTCTGTCAGGTCCGAT",
    "D": "ACCTCCTCCTCCAGAAT",
    "E": "AACCGGACACACTTAGT",
    "F": "AGAGTCCAATTCGCAGT",
    "G": "AATCAAGGCTTAACGGT",
    "H": "ATGTTGAATCCTAGCGT",
    "Q": "AGTAGCTGTGTGCA",
}

MAS_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "Q"]
TSO_CORE = "AAGCAGTGGTATCAACGCAGAGTAC"
R1 = "ACACGACGCTCTTCCGATCT"


def revcomp(s: str) -> str:
    """Reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(c, c) for c in reversed(s))


@dataclass
class AdapterHit:
    """A match of an adapter in the sequence."""
    adapter: str
    pos: int
    strand: str
    edits: int

    def __lt__(self, other):
        if self.pos != other.pos:
            return self.pos < other.pos
        return self.edits < other.edits


def find_primer_hits(seq: str, max_edits: int = 3) -> list[tuple[int, int]]:
    """Find all TSO and R1 hits; return list of (start, end) positions."""
    primer_hits = []

    # TSO forward and reverse
    result = edlib.align(TSO_CORE, seq, mode="HW", task="locations", k=max_edits)
    if result["editDistance"] != -1 and result["editDistance"] <= max_edits:
        if result.get("locations"):
            for q_start, q_end in result.get("locations", []):
                primer_hits.append((q_start, q_end + 1))

    result_rc = edlib.align(revcomp(TSO_CORE), seq, mode="HW", task="locations", k=max_edits)
    if result_rc["editDistance"] != -1 and result_rc["editDistance"] <= max_edits:
        if result_rc.get("locations"):
            for q_start, q_end in result_rc.get("locations", []):
                primer_hits.append((q_start, q_end + 1))

    # R1 forward and reverse
    result = edlib.align(R1, seq, mode="HW", task="locations", k=max_edits)
    if result["editDistance"] != -1 and result["editDistance"] <= max_edits:
        if result.get("locations"):
            for q_start, q_end in result.get("locations", []):
                primer_hits.append((q_start, q_end + 1))

    result_rc = edlib.align(revcomp(R1), seq, mode="HW", task="locations", k=max_edits)
    if result_rc["editDistance"] != -1 and result_rc["editDistance"] <= max_edits:
        if result_rc.get("locations"):
            for q_start, q_end in result_rc.get("locations", []):
                primer_hits.append((q_start, q_end + 1))

    return primer_hits


def has_nearby_primer(adapter_hit: AdapterHit, primer_hits: list[tuple[int, int]],
                      mas_flank: int = 4) -> bool:
    """Check if adapter has a TSO or R1 within mas_flank bp."""
    for p_start, p_end in primer_hits:
        # Adapter spans roughly [pos, pos+17]
        # Check if primer start or end is within mas_flank bp of adapter
        if abs(adapter_hit.pos - p_end) <= mas_flank or abs(adapter_hit.pos + 17 - p_start) <= mas_flank:
            return True
    return False


def find_adapter_hits(seq: str, max_edits: int = 2) -> list[AdapterHit]:
    """
    Find Kinnex adapters with edlib HW, filtered by nearby TSO/R1 primers.
    """
    hits = []

    for letter in MAS_ORDER:
        pattern = MAS[letter]

        # Forward strand
        result = edlib.align(pattern, seq, mode="HW", task="locations", k=max_edits)
        if result["editDistance"] != -1 and result["editDistance"] <= max_edits:
            if result.get("locations"):
                for q_start, q_end in result.get("locations", []):
                    hits.append(AdapterHit(
                        adapter=letter,
                        pos=q_start,
                        strand="+",
                        edits=result["editDistance"]
                    ))

        # Reverse complement
        rev_pattern = revcomp(pattern)
        result_rc = edlib.align(rev_pattern, seq, mode="HW", task="locations", k=max_edits)
        if result_rc["editDistance"] != -1 and result_rc["editDistance"] <= max_edits:
            if result_rc.get("locations"):
                for q_start, q_end in result_rc.get("locations", []):
                    hits.append(AdapterHit(
                        adapter=letter,
                        pos=q_start,
                        strand="-",
                        edits=result_rc["editDistance"]
                    ))

    # Find primer hits for filtering
    primer_hits = find_primer_hits(seq, max_edits=3)

    # Filter: keep only adapters with nearby primers
    hits = [h for h in hits if has_nearby_primer(h, primer_hits, mas_flank=4)]

    # Remove overlaps
    hits.sort(key=lambda h: (h.pos, h.edits))

    kept = []
    max_adapter_len = 17

    for hit in hits:
        overlaps = False
        for kept_hit in kept:
            overlap_start = max(hit.pos, kept_hit.pos)
            overlap_end = min(hit.pos + max_adapter_len, kept_hit.pos + max_adapter_len)
            if overlap_end - overlap_start > 3:
                overlaps = True
                break

        if not overlaps:
            kept.append(hit)

    kept.sort(key=lambda h: h.pos)
    return kept


def is_mirror_read(hits: list[AdapterHit]) -> bool:
    """Check if read has consecutive adapters with same letter, opposite strand."""
    if len(hits) < 2:
        return False

    for i in range(len(hits) - 1):
        if hits[i].adapter == hits[i + 1].adapter and hits[i].strand != hits[i + 1].strand:
            return True

    return False


def parse_zm_from_qname(qname: str) -> int:
    """Extract ZMW number from SAM query name."""
    parts = qname.split("/")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def main(argv: list[str] | None = None):
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--argentag-dir", default=config.get("argentag_dir"), help="ArgenTag folder, holding data/ and analysis/")
    ap.add_argument("--out", help="directory for independent_scan.txt (default: <argentag-dir>/analysis)")
    args = ap.parse_args(argv)
    if not args.argentag_dir or args.argentag_dir.startswith("<"):
        ap.error("no ArgenTag folder: give --argentag-dir or set argentag_dir in study/study.local.toml")
    here = Path(args.argentag_dir) / "analysis"
    out = Path(args.out) if args.out else here
    out.mkdir(parents=True, exist_ok=True)
    bam_path = here.parent / "data" / "segmented.non_passing.bam"
    ref_summary_path = here / "full_reads" / "read_summary.parquet"
    lines = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("Reading BAM file...")
    bam = pysam.AlignmentFile(str(bam_path), "rb", check_sq=False)

    all_records = []
    for record in bam:
        zm = parse_zm_from_qname(record.query_name)
        all_records.append((zm, record.query_name, record.query_sequence))

    bam.close()

    say(f"Total records: {len(all_records)}")

    # Sample 5000 records with fixed seed
    random.seed(42)
    sample_indices = random.sample(range(len(all_records)), min(5000, len(all_records)))
    sample_records = [all_records[i] for i in sample_indices]

    say(f"Sample size: {len(sample_records)}")

    # Scan adapters
    say("Scanning adapters in sample...")
    my_results = []

    for zm, qname, seq in sample_records:
        hits = find_adapter_hits(seq, max_edits=2)
        is_mirror = is_mirror_read(hits)

        my_results.append({
            "zm": zm,
            "n_adapters": len(hits),
            "is_mirror": is_mirror,
        })

    # Load reference
    say("Loading reference results...")
    ref_summary = pl.read_parquet(str(ref_summary_path))

    my_zm_set = set(r["zm"] for r in my_results)
    ref_summary_sample = ref_summary.filter(pl.col("zm").is_in(pl.Series(sorted(my_zm_set), dtype=ref_summary["zm"].dtype)))

    # Build concordance
    both_count = 0
    ours_only = 0
    theirs_only = 0
    disagreements = []

    for my_result in my_results:
        zm = my_result["zm"]
        my_mirror = my_result["is_mirror"]

        ref_row = ref_summary_sample.filter(pl.col("zm") == zm)
        if len(ref_row) == 0:
            theirs_only += 1
            continue

        their_mirror = ref_row.select("primary_pattern").to_series()[0] == "mirror"

        if my_mirror == their_mirror:
            both_count += 1
        else:
            if my_mirror and not their_mirror:
                ours_only += 1
            else:
                theirs_only += 1

            disagreements.append({
                "zm": zm,
                "my_mirror": my_mirror,
                "their_mirror": their_mirror,
                "my_n_adapters": my_result["n_adapters"],
            })

    # Report
    total = both_count + ours_only + theirs_only
    say(f"\nConcordance (sample of {len(sample_records)} reads):")
    say(f"  Same call (mirror or not): {both_count} ({100*both_count/total:.1f}%)")
    say(f"  Mirror in this scan only: {ours_only} ({100*ours_only/total:.1f}%)")
    say(f"  Mirror in the pipeline only: {theirs_only} ({100*theirs_only/total:.1f}%)")

    # The per-read list holds ArgenTag ZMW ids: it stays in the ArgenTag folder, never in the published text.
    say("\nDisagreement details (first 10):")
    pl.DataFrame(disagreements, schema={"zm": pl.Int64, "my_mirror": pl.Boolean, "their_mirror": pl.Boolean, "my_n_adapters": pl.Int64}) \
        .write_csv(here / "independent_scan_disagreements.tsv", separator="\t")
    print(f"{len(disagreements)} disagreements written to {here / 'independent_scan_disagreements.tsv'}", file=sys.stderr)

    # Estimate mirror rate
    mirror_in_sample = sum(1 for r in my_results if r["is_mirror"])
    sample_mirror_rate = mirror_in_sample / len(sample_records)

    n_nonpassing = len(all_records)
    estimated_mirror = int(n_nonpassing * sample_mirror_rate)

    se = np.sqrt(sample_mirror_rate * (1 - sample_mirror_rate) / len(sample_records))
    ci_low_rate = max(0, sample_mirror_rate - 1.96 * se)
    ci_high_rate = min(1, sample_mirror_rate + 1.96 * se)
    ci_low_count = int(n_nonpassing * ci_low_rate)
    ci_high_count = int(n_nonpassing * ci_high_rate)

    say("\nMirror rate estimate:")
    say(f"  Sample: {mirror_in_sample}/{len(sample_records)} = {100*sample_mirror_rate:.2f}%")
    say(f"  Estimated mirror ZMWs in {n_nonpassing:,} non-passing: {estimated_mirror:,}")
    say(f"  95% CI: [{ci_low_count:,}, {ci_high_count:,}]")
    say(f"  95% CI as fraction: [{100*ci_low_rate:.2f}%, {100*ci_high_rate:.2f}%]")

    # Reference rate
    ref_mirror_count = ref_summary_sample.filter(
        pl.col("primary_pattern") == "mirror"
    ).select(pl.len()).to_series()[0]

    say("\nReference (in our sample ZMs):")
    say(f"  {ref_mirror_count}/{len(ref_summary_sample)} = {100*ref_mirror_count/len(ref_summary_sample):.2f}%")

    say("\nScript executed successfully.")
    (out / "independent_scan.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
