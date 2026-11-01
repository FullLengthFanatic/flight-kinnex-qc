"""`kinnex-qc run`: per-read annotation of a HiFi BAM, and the function that writes its outputs.

Option parsing, the rule checks, the one-line stderr errors and the exit codes live in cli.py; this
module takes plain, already-validated values, so other code (the study's stage 02, Task 14) can import
it as a library without argparse.
"""
import functools
import gzip
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pysam

from . import __version__, skera
from .annotate import adapter_chain, annotate_read, chain_flags, primary_pattern, segment_hits, split_segments
from .classify import call_from_hits
from .io import iter_bam
from .metrics import artifact_rates, junction_composition, junction_table, mirror_junction_shares, mirror_size_rule, pattern_shares, read_fractions
from .parallel import batched, bounded_imap, process_pool
from .patterns import Chemistry, FAIL_MAX_EDITS, HIFI_MAX_EDITS, KINNEX8_SC3, MAS_ORDER, all_patterns
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


def annotate(hifi, chem: Chemistry, fraction: float, threads: int,
             counts: Optional[dict] = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Annotate every record of a HiFi BAM (optionally hash-sampled), parallel over batches of BATCH.

    Returns (read_summary, molecules, features): READ_SUMMARY, ANNOT_SEGMENTS and READ_FEATURES
    columns and dtypes, without `source` and `skera_passing`, rows ordered by zm so the output does
    not depend on the worker pool.

    counts: when given, receives counts["hifi_reads_in_file"], the number of records in the whole
    HiFi BAM (before the sample), counted during this single pass instead of a second read of the
    file. At fraction 1 every yielded record is counted (keep=None, nothing is filtered); below 1,
    iter_bam calls keep(zm) once per record (a HiFi BAM has one record per ZMW), so counting those
    calls counts the whole file even though only the sampled records are yielded.
    """
    seen = 0

    def _keep(zm):
        nonlocal seen
        seen += 1
        return in_hash_sample(zm, fraction)

    keep = None if fraction >= 1 else _keep

    def items():
        nonlocal seen
        for r in iter_bam(hifi, threads=threads, keep=keep):
            if keep is None:
                seen += 1
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
    filter the HiFi pass used."""
    cols = {"zm": [], "di": [], "qs": [], "qe": [], "dl": [], "dr": []}
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
    return pl.DataFrame(cols, schema={"zm": pl.UInt32, "di": pl.Int16, "qs": pl.Int32, "qe": pl.Int32,
                                       "dl": pl.Int8, "dr": pl.Int8})


def run(hifi: Path, sreads: Optional[Path], out: Path, chem: Chemistry, fraction: float, threads: int,
        write_molecules: bool, template: Optional[Path], examples: int, *, adapters: Optional[str],
        primers: str, fraction_defaulted: bool, hifi_arg: str, sreads_arg: Optional[str],
        template_arg: Optional[str], argv: list[str]) -> int:
    """Annotate the HiFi BAM and write reads.tsv.gz, summary.json and run.log into `out`.

    Takes plain, already-validated values: option parsing, the rule checks and exit codes are
    cli.py's job. adapters, primers, hifi_arg, sreads_arg and template_arg are the raw option
    strings (or None), kept only for run.log and summary.json, which record them as given rather
    than resolved (summary.json's "template" is None when --template was not given, even when
    --sreads supplies its effective default). argv is the argument list cli.main parsed (without
    the program name), logged shell-quoted. Returns the process exit code.
    """
    out.mkdir(parents=True, exist_ok=True)

    lines = []

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)
        lines.append(msg)

    t0 = time.time()
    log(f"kinnex-qc run: start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("command: " + shlex.join(["kinnex-qc", *argv]))
    log(f"options: hifi={hifi} sreads={sreads} out={out} adapters={adapters} primers={primers} "
        f"fraction={fraction} threads={threads} write_molecules={write_molecules} "
        f"template={template} examples={examples}")
    if fraction_defaulted:
        log("fraction set to 1 because --write-molecules was given without --fraction")

    skv = None
    if sreads is not None:
        skv = skera_version(sreads)
        if skv is None:
            log(f"warning: no @PG line names skera in {sreads}")

    counts: dict = {}
    read_summary, molecules, features = annotate(hifi, chem, fraction, threads, counts=counts)
    hifi_reads_in_file = counts["hifi_reads_in_file"]

    # The S-read BAM is read only after annotate()'s process pool has been created and shut down
    # (fork safety: a pysam file opened with threads > 1 before a fork can leave the child with a
    # half-copied reader), and only for the ZMWs in the sample, with the same hash keep function
    # the HiFi pass used.
    skera_summary = None
    reads_out = read_summary.select("zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas")
    read_summary_for_junctions = read_summary.with_columns(pl.lit(None, dtype=pl.Boolean).alias("skera_passing"))
    if sreads is not None:
        keep = None if fraction >= 1 else (lambda zm: in_hash_sample(zm, fraction))
        sreads_df = read_sreads(sreads, threads, keep=keep)
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
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=1))

    log(f"records: hifi_reads_in_file={hifi_reads_in_file} hifi_reads={read_summary.height}")
    log(f"run time: {time.time() - t0:.1f} s")
    (out / "run.log").write_text("\n".join(lines) + "\n")
    return 0
