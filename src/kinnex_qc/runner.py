"""`kinnex-qc run`: per-read annotation of a HiFi BAM, and the CLI command that writes it out."""
import argparse
import functools
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pysam

from . import __version__
from .annotate import adapter_chain, annotate_read, chain_flags, primary_pattern, segment_hits, split_segments
from .classify import call_from_hits
from .io import iter_bam
from .parallel import batched, bounded_imap, process_pool
from .patterns import Chemistry, FAIL_MAX_EDITS, HIFI_MAX_EDITS, KINNEX8_SC3, MAS_ORDER, all_patterns, chemistry_from
from .sampling import in_hash_sample
from .schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_SUMMARY

BATCH = 200


def token_pos(left: Optional[str], right: Optional[str], order=MAS_ORDER) -> Optional[int]:
    """min letter index of two adjacent same-strand tokens in order, else None (contract, ANNOT_SEGMENTS.pos)."""
    if left is None or right is None or left.endswith("'") != right.endswith("'"):
        return None
    i, j = order.index(left.rstrip("'")), order.index(right.rstrip("'"))
    return min(i, j) if abs(i - j) == 1 else None


def tso_faces_lower(tso_side: Optional[str], left: Optional[str], right: Optional[str], order=MAS_ORDER) -> Optional[bool]:
    """For a proper segment between two adapters: does its TSO end face the lower-index adapter?"""
    if tso_side is None or left is None or right is None:
        return None
    i, j = order.index(left.rstrip("'")), order.index(right.rstrip("'"))
    return (tso_side == "head") == (i < j)


@functools.lru_cache(maxsize=None)
def _hifi_patterns(chem: Chemistry):
    return all_patterns(HIFI_MAX_EDITS, chem)


@functools.lru_cache(maxsize=None)
def _fail_patterns(chem: Chemistry):
    return all_patterns(FAIL_MAX_EDITS, chem)


def annotate_batch(batch: list[tuple], chem: Chemistry = KINNEX8_SC3) -> tuple[dict, dict, dict]:
    """Port of the study's stage 02 annotate_batch: same batch items (zm, source, seq, np, rq,
    skera_passing) and the same three column dicts out, with the chemistry's adapters, primers and
    MAS order in place of the study's fixed defaults."""
    feats = {f.name: [] for f in READ_FEATURES}
    summ = {f.name: [] for f in READ_SUMMARY}
    segs = {f.name: [] for f in ANNOT_SEGMENTS}
    for zm, source, seq, np_, rq, skera_passing in batch:
        patterns = _fail_patterns(chem) if source == "fail_sub" else _hifi_patterns(chem)
        hits = annotate_read(seq, patterns)
        for h in hits:
            for k, v in (("zm", zm), ("source", source), ("name", h.name), ("kind", h.kind),
                         ("strand", h.strand), ("start", h.start), ("end", h.end), ("edits", h.edits)):
                feats[k].append(v)
        tokens, flipped = adapter_chain(hits)
        flags = chain_flags(tokens, order=chem.order)
        for k, v in (("zm", zm), ("source", source), ("read_len", len(seq)), ("np", np_), ("rq", rq),
                     ("chain", " ".join(tokens)), ("chain_flipped", flipped), ("flags", ",".join(sorted(flags))),
                     ("primary_pattern", primary_pattern(flags)), ("n_mas", len(tokens)),
                     ("skera_passing", skera_passing)):
            summ[k].append(v)
        for i, (start, end, left, right) in enumerate(split_segments(len(seq), hits)):
            call = call_from_hits(seq[start:end], *segment_hits(hits, start, end))
            row = {"zm": zm, "source": source, "seg_index": i, "start": start, "end": end,
                   "left_token": left, "right_token": right, "pos": token_pos(left, right, chem.order),
                   "sread_len": end - start, "end_class": call.end_class, "tso_side": call.tso_side,
                   "tso_toward_lower": tso_faces_lower(call.tso_side, left, right, chem.order),
                   "internal": call.internal, "n_internal": len(call.internal_hits), "bead_ok": call.bead_ok,
                   "polyT_len": call.polyT_len, "polyT_anchored": call.polyT_anchored, "insert_len": call.insert_len}
            for k, v in row.items():
                segs[k].append(v)
    return feats, summ, segs


def _merge(schema: pa.Schema, parts: list[dict[str, list]]) -> dict[str, list]:
    out = {f.name: [] for f in schema}
    for part in parts:
        for k in out:
            out[k].extend(part[k])
    return out


def _frame(schema: pa.Schema, drop: set[str], cols: dict[str, list]) -> pl.DataFrame:
    fields = [f for f in schema if f.name not in drop]
    table = pa.Table.from_arrays([pa.array(cols[f.name], type=f.type) for f in fields], schema=pa.schema(fields))
    return pl.from_arrow(table)


def annotate(hifi, chem: Chemistry, fraction: float, threads: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Annotate every record of a HiFi BAM (optionally hash-sampled), parallel over batches of BATCH.

    Returns (read_summary, molecules, features): READ_SUMMARY, ANNOT_SEGMENTS and READ_FEATURES
    columns and dtypes, without `source` and `skera_passing`, rows ordered by zm so the output does
    not depend on the worker pool.
    """
    keep = None if fraction >= 1 else (lambda zm: in_hash_sample(zm, fraction))

    def items():
        for r in iter_bam(hifi, threads=threads, keep=keep):
            yield (r.get_tag("zm"), "hifi", r.query_sequence, r.get_tag("np"), r.get_tag("rq"), True)

    feats_parts, summ_parts, segs_parts = [], [], []
    with process_pool(threads) as pool:
        func = functools.partial(annotate_batch, chem=chem)
        for feats, summ, segs in bounded_imap(pool, func, batched(items(), BATCH), 4 * threads):
            feats_parts.append(feats)
            summ_parts.append(summ)
            segs_parts.append(segs)

    read_summary = _frame(READ_SUMMARY, {"source", "skera_passing"}, _merge(READ_SUMMARY, summ_parts)).sort("zm")
    molecules = _frame(ANNOT_SEGMENTS, {"source"}, _merge(ANNOT_SEGMENTS, segs_parts)).sort(["zm", "seg_index"])
    features = _frame(READ_FEATURES, {"source"}, _merge(READ_FEATURES, feats_parts)).sort(["zm", "start"])
    return read_summary, molecules, features


def skera_version(sreads: Path) -> Optional[str]:
    """VN of the first @PG line in the S-read BAM's header whose ID or PN names skera (case-insensitive),
    or None when the header has no such line."""
    with pysam.AlignmentFile(str(sreads), "rb", check_sq=False) as bam:
        pgs = bam.header.to_dict().get("PG", [])
    for pg in pgs:
        if "skera" in pg.get("ID", "").lower() or "skera" in pg.get("PN", "").lower():
            return pg.get("VN")
    return None


def run(args: argparse.Namespace) -> int:
    """Validate `kinnex-qc run`'s options, annotate the HiFi BAM and write reads.tsv.gz, summary.json
    and run.log into --out. Returns the process exit code."""

    def fail(msg: str) -> int:
        print(f"kinnex-qc run: error: {msg}", file=sys.stderr)
        return 2

    hifi = Path(args.hifi)
    if not hifi.exists():
        return fail(f"--hifi: no such file: {hifi}")
    sreads = Path(args.sreads) if args.sreads else None
    if sreads is not None and not sreads.exists():
        return fail(f"--sreads: no such file: {sreads}")
    if args.adapters is not None and not Path(args.adapters).exists():
        return fail(f"--adapters: no such file: {args.adapters}")
    if args.template is not None and not Path(args.template).exists():
        return fail(f"--template: no such file: {args.template}")
    template = Path(args.template) if args.template else sreads

    if args.write_molecules and sreads is None and template is None:
        return fail("--write-molecules needs --sreads or --template")

    fraction_given = args.fraction is not None
    fraction = args.fraction if fraction_given else (1.0 if args.write_molecules else 0.1)
    if not (0 < fraction <= 1):
        return fail(f"--fraction must be in (0, 1]: {fraction}")

    try:
        chem = chemistry_from(args.adapters, args.primers)
    except ValueError as e:
        return fail(str(e))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = []

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)
        lines.append(msg)

    t0 = time.time()
    log(f"kinnex-qc run: start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("command: " + " ".join(["kinnex-qc", "run", *sys.argv[2:]]))
    log(f"options: hifi={hifi} sreads={sreads} out={out} adapters={args.adapters} primers={args.primers} "
        f"fraction={fraction} threads={args.threads} write_molecules={args.write_molecules} "
        f"template={template} examples={args.examples}")
    if not fraction_given and args.write_molecules:
        log("fraction set to 1 because --write-molecules was given without --fraction")

    skv = None
    if sreads is not None:
        skv = skera_version(sreads)
        if skv is None:
            log(f"warning: no @PG line names skera in {sreads}")

    hifi_reads_in_file = sum(1 for _ in iter_bam(hifi, threads=args.threads))
    read_summary, _molecules, _features = annotate(hifi, chem, fraction, args.threads)

    with gzip.open(out / "reads.tsv.gz", "wb") as f:
        read_summary.select("zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas").write_csv(
            f, separator="\t")

    summary = {
        "version": __version__,
        "inputs": {
            "hifi": args.hifi,
            "sreads": args.sreads,
            "template": args.template,
            "adapters": args.adapters,
        },
        "fraction": fraction,
        "chemistry": {
            "adapters": args.adapters if args.adapters is not None else "Kinnex 8-fold default",
            "order": list(chem.order),
            "tso": chem.tso,
            "r1": chem.r1,
            "primers": args.primers,
        },
        "hifi_reads_in_file": hifi_reads_in_file,
        "hifi_reads": read_summary.height,
        "skera_version": skv,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    log(f"records: hifi_reads_in_file={hifi_reads_in_file} hifi_reads={read_summary.height}")
    log(f"run time: {time.time() - t0:.1f} s")
    (out / "run.log").write_text("\n".join(lines) + "\n")
    return 0
