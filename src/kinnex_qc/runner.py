"""`kinnex-qc run`: per-read annotation of a HiFi BAM, and the function that writes its outputs.

Option parsing, the rule checks, the one-line stderr errors and the exit codes live in cli.py; this
module takes plain, already-validated values, so other code (the study's stage 02) can import
it as a library without argparse.
"""
import collections
import contextlib
import functools
import gzip
import json
import math
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pysam

from . import __version__, reader_report, skera
from .annotate import adapter_chain, annotate_read, chain_flags, primary_pattern, segment_hits, split_segments
from .classify import call_from_hits
from .io import iter_bam
from .metrics import artifact_rates, junction_composition, junction_table, mirror_junction_shares, mirror_size_rule, pattern_shares, read_fractions
from .molecules import MoleculeSink
from .parallel import batched, bounded_imap, process_pool
from .patterns import Chemistry, FAIL_MAX_EDITS, HIFI_MAX_EDITS, KINNEX8_SC3, MAS_ORDER, all_patterns
from .sampling import in_hash_sample
from .schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_SUMMARY

BATCH = 200
PROGRESS_EVERY = 1_000_000   # HiFi records read between two progress lines of the annotation pass
SREAD_CHUNK = 100_000
_SREADS = pa.schema([("zm", pa.uint32()), ("di", pa.int16()), ("qs", pa.int32()), ("qe", pa.int32()),
                     ("dl", pa.int8()), ("dr", pa.int8())])


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


def _without(schema: pa.Schema, drop: set[str]) -> pa.Schema:
    return pa.schema([f for f in schema if f.name not in drop])


_SUMMARY_OUT = _without(READ_SUMMARY, {"source", "skera_passing"})
_MOLECULES_OUT = _without(ANNOT_SEGMENTS, {"source"})
_FEATURES_OUT = _without(READ_FEATURES, {"source"})


def _record_batch(schema: pa.Schema, cols: dict[str, list]) -> pa.RecordBatch:
    """The columns of cols that schema names, as arrow with schema's types (other columns of cols are left out)."""
    return pa.RecordBatch.from_arrays([pa.array(cols[f.name], type=f.type) for f in schema], schema=schema)


def _frame(schema: pa.Schema, batches: list[pa.RecordBatch]) -> pl.DataFrame:
    return pl.from_arrow(pa.Table.from_batches(batches, schema=schema))


def annotate(hifi, chem: Chemistry, fraction: float, threads: int, counts: Optional[dict] = None,
             molecules_out: Optional[MoleculeSink] = None, progress: Optional[Callable[[int, int], None]] = None
             ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Annotate the records of a HiFi BAM in the hash sample given by fraction (every record at fraction 1), parallel
    over batches of BATCH.

    Returns (read_summary, molecules, features): READ_SUMMARY, ANNOT_SEGMENTS and READ_FEATURES
    columns and dtypes, without `source` and `skera_passing`, rows ordered by zm so the output does
    not depend on the worker pool.

    molecules_out: when given, every record of the BAM is annotated, not only the sampled ones, and each batch of
    records goes to molecules_out.add with its molecule rows as soon as the workers return it, so the parent holds
    the records of the batches in flight only; the returned frames still hold the sampled reads alone, the same
    frames as without molecules_out.

    counts: when given, receives counts["hifi_reads_in_file"], the number of records in the whole
    HiFi BAM (before the sample), counted during this single pass instead of a second read of the
    file. When every record is read (fraction 1, or molecules_out given) every yielded record is counted; otherwise
    iter_bam calls keep(zm) once per record (a HiFi BAM has one record per ZMW), so counting those
    calls counts the whole file even though only the sampled records are yielded.

    progress: when given, called as progress(records read, records read that are in the hash sample) every
    PROGRESS_EVERY records read from the BAM.
    """
    seen = in_sample_seen = 0

    def _count(zm) -> bool:
        """Count one record read from the BAM; True when its ZMW is in the hash sample."""
        nonlocal seen, in_sample_seen
        seen += 1
        in_sample = fraction >= 1 or in_hash_sample(zm, fraction)
        in_sample_seen += in_sample
        if progress is not None and seen % PROGRESS_EVERY == 0:
            progress(seen, in_sample_seen)
        return in_sample

    keep = None if fraction >= 1 or molecules_out is not None else _count
    # the records of each batch submitted and not yet returned, in submission order, which is the order bounded_imap
    # returns the results in
    in_flight = collections.deque()

    def item_batches():
        for records in batched(iter_bam(hifi, threads=threads, keep=keep), BATCH):
            if keep is None:
                for r in records:
                    _count(r.get_tag("zm"))
            if molecules_out is not None:
                in_flight.append(records)
            yield [(r.get_tag("zm"), "hifi", r.query_sequence, r.get_tag("np"), r.get_tag("rq"), True) for r in records]

    # Each batch's column lists become arrow as soon as they arrive, so the Python ints and strings never pile up over
    # the whole BAM: kept to the end they grew the process by about 200 bytes a row, against about 50 as arrow
    # (observed on 200,000 HiFi reads), tens of GB for the features and molecules of a whole cell.
    batches = {"summ": [], "segs": [], "feats": []}
    with process_pool(threads) as pool:
        func = functools.partial(annotate_batch, chem=chem)
        for feats, summ, segs in bounded_imap(pool, func, item_batches(), 4 * threads):
            parts = {"summ": _record_batch(_SUMMARY_OUT, summ), "segs": _record_batch(_MOLECULES_OUT, segs),
                     "feats": _record_batch(_FEATURES_OUT, feats)}
            if molecules_out is not None:
                molecules_out.add(in_flight.popleft(), pl.from_arrow(parts["segs"]))
                if fraction < 1:
                    sampled = pa.array([zm for zm in summ["zm"] if in_hash_sample(zm, fraction)],
                                       type=_SUMMARY_OUT.field("zm").type)
                    parts = {k: b.filter(pc.is_in(b.column("zm"), value_set=sampled)) for k, b in parts.items()}
            for k, b in parts.items():
                batches[k].append(b)

    # pop: a table's batches are freed once polars holds its own copy, before the sort makes another
    read_summary = _frame(_SUMMARY_OUT, batches.pop("summ")).sort("zm")
    molecules = _frame(_MOLECULES_OUT, batches.pop("segs")).sort(["zm", "seg_index"])
    features = _frame(_FEATURES_OUT, batches.pop("feats")).sort(["zm", "start"])
    if counts is not None:
        counts["hifi_reads_in_file"] = seen
    return read_summary, molecules, features


def _json_safe(value):
    """summary.json's floats go out as plain Python floats, unrounded; a NaN one goes out as null,
    since json.dumps would otherwise write the bare (invalid-JSON) token NaN."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def skera_version(sreads: Path) -> Optional[str]:
    """VN of the first @PG line in the S-read BAM's header whose ID or PN names skera (case-insensitive),
    or None when the header has no such line."""
    with pysam.AlignmentFile(str(sreads), "rb", check_sq=False) as bam:
        pgs = bam.header.to_dict().get("PG", [])
    for pg in pgs:
        if "skera" in pg.get("ID", "").lower() or "skera" in pg.get("PN", "").lower():
            return pg.get("VN")
    return None


def read_sreads(sreads: Path, threads: int, keep=None) -> pl.DataFrame:
    """S-read tags (zm, qs, qe, dl, dr) from skera's S-read BAM, with di set to each record's 0-based
    index within its ZMW (observed in skera 26.1.0 output: the S-reads of one ZMW are consecutive in
    the file, which iter_bam's own keep filter relies on). keep: as iter_bam's, the same hash-sample
    filter the HiFi pass used. Every SREAD_CHUNK records go to arrow, so no more than that many are
    held as Python objects."""
    batches = []
    cols = {f.name: [] for f in _SREADS}
    last_zm, di = None, -1
    for r in iter_bam(sreads, threads=threads, keep=keep):
        zm = r.get_tag("zm")
        di = di + 1 if zm == last_zm else 0
        last_zm = zm
        cols["zm"].append(zm)
        cols["di"].append(di)
        cols["qs"].append(r.get_tag("qs"))
        cols["qe"].append(r.get_tag("qe"))
        cols["dl"].append(r.get_tag("dl"))
        cols["dr"].append(r.get_tag("dr"))
        if len(cols["zm"]) == SREAD_CHUNK:
            batches.append(_record_batch(_SREADS, cols))
            cols = {f.name: [] for f in _SREADS}
    batches.append(_record_batch(_SREADS, cols))
    return _frame(_SREADS, batches)


def example_sequences(hifi: Path, zms: set[int], threads: int) -> dict[int, str]:
    """The HiFi read sequence of each ZMW in zms, from one more pass over the HiFi BAM that stops once all are
    found: only the example reads' sequences are held, never every read's."""
    seqs: dict[int, str] = {}
    if not zms:
        return seqs
    for r in iter_bam(hifi, threads=threads, keep=zms.__contains__):
        seqs[r.get_tag("zm")] = r.query_sequence
        if len(seqs) == len(zms):
            break
    return seqs


def run(hifi: Path, sreads: Optional[Path], out: Path, chem: Chemistry, fraction: float, threads: int,
        write_molecules: bool, template: Optional[Path], examples: int, *, adapters: Optional[str],
        primers: str, hifi_arg: str, sreads_arg: Optional[str],
        template_arg: Optional[str], argv: list[str]) -> int:
    """Annotate the HiFi BAM and write reads.tsv.gz, summary.json, report.html and run.log into `out`.

    Takes plain, already-validated values: option parsing, the rule checks and exit codes are
    cli.py's job. adapters, primers, hifi_arg, sreads_arg and template_arg are the raw option
    strings (or None), kept only for run.log and summary.json, which record them as given rather
    than resolved (summary.json's "template" is None when --template was not given, even when
    --sreads supplies its effective default). argv is the argument list cli.main parsed (without
    the program name), logged shell-quoted. examples: example reads per category in report.html
    (reader_report.pick_examples; 0 for none). Returns the process exit code.
    """
    out.mkdir(parents=True, exist_ok=True)

    lines = []

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)
        lines.append(msg)

    t0 = time.time()

    def elapsed() -> str:
        s = int(time.time() - t0)
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"

    log(f"kinnex-qc run: start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("command: " + shlex.join(["kinnex-qc", *argv]))
    log(f"options: hifi={hifi} sreads={sreads} out={out} adapters={adapters} primers={primers} "
        f"fraction={fraction} threads={threads} write_molecules={write_molecules} "
        f"template={template} examples={examples}")

    skv = None
    if sreads is not None:
        skv = skera_version(sreads)
        if skv is None:
            log(f"warning: no @PG line names skera in {sreads}")

    # --write-molecules: the molecules of every read go to molecules.bam during the one annotation pass, while the
    # frames, and so every number below, hold the hash sample alone
    counts: dict = {}
    with (MoleculeSink(template, out / "molecules.bam") if write_molecules else contextlib.nullcontext()) as sink:
        read_summary, molecules, features = annotate(
            hifi, chem, fraction, threads, counts=counts, molecules_out=sink,
            progress=lambda n, m: log(f"annotated: {n} records read, {m} in the sample, elapsed {elapsed()}"))
    hifi_reads_in_file = counts["hifi_reads_in_file"]

    molecules_written = None
    if write_molecules:
        molecules_written = sink.written
        source = (f"all {hifi_reads_in_file} reads" if fraction >= 1 else
                  f"the hash sample (fraction {fraction}, {read_summary.height} of {hifi_reads_in_file} reads)")
        log(f"molecules written: {molecules_written}, for all {hifi_reads_in_file} HiFi reads in the file; summary "
            f"numbers come from {source}")

    # The S-read BAM is read only after annotate()'s process pool has been created and shut down
    # (fork safety: a pysam file opened with threads > 1 before a fork can leave the child with a
    # half-copied reader), and only for the ZMWs in the sample, with the same hash keep function
    # the HiFi pass used.
    skera_summary = None
    sreads_df = passing = fc_all = None
    reads_out = read_summary.select("zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas")
    read_summary_for_junctions = read_summary.with_columns(pl.lit(None, dtype=pl.Boolean).alias("skera_passing"))
    if sreads is not None:
        keep = None if fraction >= 1 else (lambda zm: in_hash_sample(zm, fraction))
        log(f"S-read pass: start, elapsed {elapsed()}")
        sreads_df = read_sreads(sreads, threads, keep=keep)
        log(f"S-read pass: end, {sreads_df.height} S-reads of the checked reads, elapsed {elapsed()}")
        order_len = len(chem.order)

        passing = skera.passing_from_labels(sreads_df, order_len=order_len)
        fc_all = skera.false_cuts(sreads_df, features)
        budget_cls = skera.budget_classes(molecules, sreads_df)
        usable_not_clean = skera.budget_per_read(budget_cls).select("zm", "usable_not_clean")

        # reads.tsv.gz's false_cuts (below) is every false cut found in a ZMW with S-reads, any read
        # pattern; false_cut_stats' summary false_cuts (below) counts only false cuts in complete,
        # derived-passing ZMWs, so the two numbers are not expected to agree.
        per_read = (read_summary.select("zm")
                    .join(sreads_df.group_by("zm").len().rename({"len": "n_sreads"}), on="zm", how="left")
                    .join(passing, on="zm", how="left")
                    .join(fc_all.group_by("zm").len().rename({"len": "false_cuts"}), on="zm", how="left")
                    .join(usable_not_clean, on="zm", how="left")
                    .with_columns(pl.col("n_sreads", "false_cuts", "usable_not_clean").fill_null(0),
                                  pl.col("passing").fill_null(False)))
        reads_out = reads_out.join(per_read, on="zm", how="left")

        read_summary_for_junctions = read_summary.join(
            passing.rename({"passing": "skera_passing"}), on="zm", how="left"
        ).with_columns(pl.col("skera_passing").fill_null(False))

        skera_summary = {
            **skera.false_cut_stats(read_summary, features, sreads_df, passing=passing, fc=fc_all, order=chem.order),
            "mirror_arms": skera.mirror_arm_stats(molecules, sreads_df),
            "budget": skera.budget_stats(read_summary, budget_cls, order_len=order_len),
        }

    csv_text = reads_out.write_csv(separator="\t")
    with gzip.open(out / "reads.tsv.gz", "wb") as f:
        f.write(csv_text.encode())

    summary = {
        "version": __version__,
        "inputs": {
            "hifi": hifi_arg,
            "sreads": sreads_arg,
            "template": template_arg,
            "adapters": adapters,
        },
        "fraction": fraction,
        "chemistry": {
            "adapters": adapters if adapters is not None else "Kinnex 8-fold default",
            "order": list(chem.order),
            "tso": chem.tso,
            "r1": chem.r1,
            "primers": primers,
        },
        "hifi_reads_in_file": hifi_reads_in_file,
        "hifi_reads": read_summary.height,
        "skera_version": skv,
        "patterns": pattern_shares(read_summary).to_dicts(),
        "fractions": read_fractions(read_summary),
        "artifacts": artifact_rates(molecules),
        "mirror_junction_shares": mirror_junction_shares(molecules),
        "junctions": junction_table(molecules).to_dicts(),
        "junction_composition": junction_composition(molecules, read_summary_for_junctions).to_dicts(),
        "mirror_size_rule": mirror_size_rule(molecules, read_summary, order=chem.order).to_dicts(),
    }
    if skera_summary is not None:
        summary["skera"] = skera_summary
    if molecules_written is not None:
        summary["molecules_written"] = molecules_written
    summary = _json_safe(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    picks = reader_report.pick_examples(read_summary, molecules, features, examples, order=chem.order,
                                        sreads=sreads_df, passing=passing, fc=fc_all)
    seqs, zms = {}, {p["zm"] for p in picks}
    if zms:
        log(f"examples pass: start, {len(zms)} example reads, elapsed {elapsed()}")
        seqs = example_sequences(hifi, zms, threads)
        log(f"examples pass: end, elapsed {elapsed()}")
    report = reader_report.write(out, summary, [{**p, "seq": seqs[p["zm"]]} for p in picks])
    log(f"report: {report} ({len(picks)} example reads)")

    log(f"records: hifi_reads_in_file={hifi_reads_in_file} hifi_reads={read_summary.height}")
    log(f"run time: {time.time() - t0:.1f} s")
    (out / "run.log").write_text("\n".join(lines) + "\n")
    return 0
