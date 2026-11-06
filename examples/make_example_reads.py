#!/usr/bin/env python3
"""Five example HiFi reads from BD144_ND for showing colleagues, as a FASTA file and a colour view (example_reads.fasta and
example_reads.html, in --out):
a complete array, a TSO-TSO mirror, an R1-R1 mirror, a double mirror (TSO-TSO and R1-R1 molecules side by side) and a skera false
cut at a sequence that resembles adapter D. For the double mirror, the fate of skera's S-reads (lima, isoseq refine) comes from
the hash-sample run, so that read is picked among those in the hash sample.

Reads come from the calibration set (the first 200,000 ND HiFi reads), which has the read sequences, our full-read annotation and
the S-reads skera 26.1.0 delivered. In the FASTA, UPPERCASE marks the adapters (and the site skera called adapter D); TSO and Read1
are listed in the header. The HTML colours adapters, TSO, Read1 and that site separately.
A prime marks a sequence found as its reverse complement on the read. Header coordinates are 1-based, inclusive, on the shown sequence.
The reads are picked by the rules of the reader report (kinnex_qc.reader_report.clean and at_adapters) and of kinnex_qc.skera.
Usage: examples/make_example_reads.py [--calibration-dir DIR] [--hash-sample-dir DIR] [--out DIR]
--calibration-dir defaults to calibration_dir, --hash-sample-dir to <out_root>/hash_<fraction>/BD144_ND, of study/study.toml and
study.local.toml; --out defaults to this folder.
"""
import argparse
import sys
from pathlib import Path

import edlib
import polars as pl
import pysam

from kinnex_qc import skera
from kinnex_qc.reader_report import at_adapters, chain_as_read, clean
from kinnex_qc.sampling import in_hash_sample
from kinnex_qc.viewer import CSS, LABEL, html_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study" / "pipeline"))
from config import load_config  # noqa: E402

D = "ACCTCCTCCTCCAGAAT"
LET = "ABCDEFGHQ"
WRAP = 100
SAMPLE = "BD144_ND"
EXAMPLE_LABEL = {**LABEL, "false": "site skera called adapter D"}      # the example page names the site by the adapter it imitates

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Example Kinnex reads</title><style>
body{font:15px/1.5 Helvetica,Arial,sans-serif;color:#1B1F24;background:#FFFFFF;margin:0;padding:24px 20px 60px}
main{max-width:1100px;margin:0 auto} h1{font-size:24px;margin:0 0 6px} h2{font-size:18px;margin:36px 0 6px}
p,li{max-width:95ch} ul{padding-left:20px;margin:4px 0 10px} .note{color:#5B636E;font-size:14px}
""" + CSS + """
</style></head><body><main>
<h1>Example Kinnex 8-fold reads</h1>
<p class="note">HiFi reads from the BD144 ND library (no TSO-TSO depletion), from the first 200,000 reads of the cell. Positions are 1-based on the shown
sequence; hover over a coloured stretch for its name and coordinates. A prime (H', TSO') marks a sequence found as its reverse complement on the read.
The same reads are in example_reads.fasta.</p>
<p class="key"><span class="adapter">adapter</span><span class="tso">TSO</span><span class="r1">Read1</span><span class="false">site skera called D</span>
<span style="padding:0">lowercase: cDNA</span></p>"""
TITLE = {"complete_array": "Complete array (reference)", "TSO-TSO_mirror": "TSO-TSO mirror", "R1-R1_mirror": "R1-R1 mirror",
         "double_mirror": "Double mirror: TSO-TSO and R1-R1 side by side", "skera_false_cut": "skera false cut at a D-like sequence"}


def kind(name: str) -> str:
    return {"TSO": "tso", "R1": "r1", "skera_D": "false"}.get(name.rstrip("'"), "adapter")


def legend(header: str) -> str:
    """Insert the case legend after the read identifier."""
    head, rest = header.split(" | ", 1)
    note = "UPPERCASE: adapters and the site skera called D" if "skera_D" in header else "UPPERCASE: adapters"
    return f"{head} | {note}; TSO and Read1 positions listed below | {rest}"


def main(argv: list[str] | None = None) -> None:
    config = load_config(Path(__file__).resolve().parent.parent / "study" / "study.toml")
    out_root, fraction = config.get("out_root", ""), config.get("fraction", 0.18)
    default_hash = str(Path(out_root) / f"hash_{fraction:g}" / SAMPLE) if out_root and not out_root.startswith("<") else None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-dir", default=config.get("calibration_dir"), help="calibration folder (the first 200,000 ND HiFi reads)")
    ap.add_argument("--hash-sample-dir", default=default_hash, help="the hash-sample output folder of BD144_ND")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="folder for example_reads.fasta and example_reads.html")
    args = ap.parse_args(argv)
    if not args.calibration_dir or args.calibration_dir.startswith("<"):
        ap.error("no calibration folder: give --calibration-dir or set calibration_dir in study/study.local.toml")
    if not args.hash_sample_dir:
        ap.error("no hash-sample folder: give --hash-sample-dir or set out_root in study/study.local.toml")
    cal, hash_dir, out_dir = Path(args.calibration_dir), Path(args.hash_sample_dir), Path(args.out)
    truth = cal / "nd_first200k_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path, html_path = out_dir / "example_reads.fasta", out_dir / "example_reads.html"

    rs = pl.read_parquet(truth / "read_summary.parquet").filter(pl.col("source") == "hifi")
    seg = (pl.read_parquet(truth / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal", "left_token", "right_token"])
           .filter(pl.col("source") == "hifi").sort("zm", "start"))
    feat = pl.read_parquet(truth / "read_features.parquet").filter(pl.col("source") == "hifi").sort("zm", "start")
    sk = pl.read_parquet(truth / "segments.parquet").sort("zm", "qs")

    n_sreads = sk.group_by("zm").len().rename({"len": "n_sreads"})
    reads = rs.join(n_sreads, on="zm", how="left").with_columns(pl.col("n_sreads").fill_null(0))
    chain_classes = pl.col("end_class").list.join(",")
    mols = seg.group_by("zm", maintain_order=True).agg(pl.col("end_class"))
    reads = reads.join(mols, on="zm")

    def features(zm: int) -> list[tuple[str, int, int]]:
        return [(n + ("'" if s == "-" else ""), a, b) for n, s, a, b in feat.filter(pl.col("zm") == zm).select("name", "strand", "start", "end").iter_rows()]

    def sreads(zm: int) -> list[tuple[int, int, int, int]]:
        return sk.filter(pl.col("zm") == zm).select("qs", "qe", "dl", "dr").rows()

    def pick_clean(df: pl.DataFrame) -> pl.DataFrame:
        """The clean reads of df (the reader report's rule), ZMWs in order."""
        return clean(df.drop("n_sreads", "end_class"), seg, feat).join(df.select("zm", "n_sreads", "end_class"), on="zm").sort("zm")

    def first_at_adapters(zms: list[int]) -> int:
        ok = at_adapters(zms, sk, feat)
        return next(z for z in zms if z in ok)

    picks = {}
    c = pick_clean(reads.filter((pl.col("primary_pattern") == "complete") & pl.col("skera_passing") & ~pl.col("chain_flipped")
                                & pl.col("read_len").is_between(5000, 7000) & (pl.col("n_sreads") == 8)))
    picks["complete"] = first_at_adapters(c["zm"].to_list())
    m = pick_clean(reads.filter((pl.col("chain") == "Q' H' H Q") & pl.col("read_len").is_between(4000, 6000) & (pl.col("n_sreads") == 1)))
    m = m.filter(chain_classes == "proper,tso_tso,proper").join(sk.select("zm", "dl", "dr"), on="zm").filter(pl.col("dl") < pl.col("dr"))   # the typical case: S-read on the X..Q arm
    picks["tso_tso"] = first_at_adapters(m["zm"].to_list())
    r = reads.filter((pl.col("chain") == "A B B' A'") & (chain_classes == "proper,r1_r1,proper"))
    picks["r1_r1"] = r.sort("zm")["zm"][0]
    dm = pick_clean(reads.filter(chain_classes.str.contains("tso_tso,r1_r1|r1_r1,tso_tso") & (pl.col("end_class").list.len() == 10)
                                 & pl.col("chain").str.starts_with("A ") & pl.col("chain").str.ends_with(" Q")))
    picks["double"] = next(z for z in dm.sort("read_len", "zm")["zm"] if in_hash_sample(z, fraction))

    # false cut: complete passing read running A to Q, one gap between S-reads with no adapter of ours (skera.false_cuts), skera calls it D,
    # the real D is inside the S-read before it
    cand = pick_clean(reads.filter((pl.col("primary_pattern") == "complete") & pl.col("skera_passing") & ~pl.col("chain_flipped") & (pl.col("n_sreads") == 8)))
    fc = skera.false_cuts(sk.filter(pl.col("zm").is_in(cand["zm"])), feat.filter(pl.col("zm").is_in(cand["zm"])))
    one = fc.filter((pl.len().over("zm") == 1) & (pl.col("dr_before") == 3)).select("zm", "cut_start", "cut_end")
    cuts = {z: (e1, q2) for z, e1, q2 in one.iter_rows()}
    for zm in cand["zm"]:
        if zm not in cuts:
            continue
        e1, q2 = cuts[zm]
        mas = {n: (a, b) for n, a, b in features(zm) if n in LET}
        s = sreads(zm)
        if not (mas["D"][1] < e1 - 5 and mas["E"][0] - 5 <= next(qe for qs, qe, _, _ in s if qs == q2) <= mas["E"][0] + 5):
            continue
        if not (600 <= mas["D"][0] - mas["C"][1] <= 2000 and 600 <= mas["E"][0] - mas["D"][1] <= 2000 and mas["D"][1] + 200 < e1):
            continue
        break
    else:
        raise SystemExit("no false-cut example found")
    picks["false_cut"] = zm

    want = set(picks.values())
    seqs = {}
    with pysam.AlignmentFile(str(cal / "nd_first200k.hifi.nomod.bam"), "rb", check_sq=False) as fh:
        for rec in fh:
            if rec.get_tag("zm") in want:
                seqs[rec.get_tag("zm")] = rec.query_sequence
                if len(seqs) == len(want):
                    break

    def render(zm: int, lo: int, hi: int, extra: list[tuple[str, int, int]] = ()) -> tuple[str, list[str], list[tuple[str, int, int, str]]]:
        """Shown sequence (adapters and extra sites uppercase), feature list for the header, and features in shown coordinates."""
        seq = list(seqs[zm][lo:hi].lower())
        marks = sorted([(n, a, b) for n, a, b in features(zm) if a >= lo and b <= hi] + list(extra), key=lambda x: x[1])
        for n, a, b in marks:
            if kind(n) in ("adapter", "false"):
                seq[a - lo:b - lo] = [x.upper() for x in seq[a - lo:b - lo]]
        return "".join(seq), [f"{n} {a - lo + 1}-{b - lo}" for n, a, b in marks], [(n, a - lo, b - lo, kind(n)) for n, a, b in marks]

    def sread_text(zm: int, lo: int = 0, hi: int = 10**9) -> str:
        return "; ".join(f"{qs - lo + 1}-{qe - lo} ({LET[dl]} to {LET[dr]})" for qs, qe, dl, dr in sreads(zm) if qe > lo and qs < hi) or "none"

    def verdict(zm: int) -> str:
        passing = rs.filter(pl.col("zm") == zm)["skera_passing"][0]
        return "skera passes the read" if passing else "skera puts the whole read in non_passing.bam"

    records = []
    zm = picks["complete"]
    s, f, mk = render(zm, 0, len(seqs[zm]))
    records.append((f"complete_array {SAMPLE} zm={zm} len={len(s)} | reference: adapters A to Q in order, 8 molecules, each Read1 ... cDNA ... TSO | "
                    f"skera passes it and writes 8 S-reads, one per molecule | " + ", ".join(f), s, mk))
    zm = picks["tso_tso"]
    s, f, mk = render(zm, 0, len(seqs[zm]))
    j = seg.filter((pl.col("zm") == zm) & (pl.col("end_class") == "tso_tso")).row(0, named=True)
    records.append((f"TSO-TSO_mirror {SAMPLE} zm={zm} len={len(s)} | Q' H' [TSO-TSO] H Q: a molecule with a TSO at both ends ({j['start'] + 1}-{j['end']}) got adapter H on both ends, "
                    f"so the array grows to Q on both sides | {verdict(zm)} and writes one S-read, {sread_text(zm)}; "
                    f"the Q'..H' arm is not written | " + ", ".join(f), s, mk))
    zm = picks["r1_r1"]
    s, f, mk = render(zm, 0, len(seqs[zm]))
    j = seg.filter((pl.col("zm") == zm) & (pl.col("end_class") == "r1_r1")).row(0, named=True)
    records.append((f"R1-R1_mirror {SAMPLE} zm={zm} len={len(s)} | A B [R1-R1] B' A': a molecule with Read1 at both ends ({j['start'] + 1}-{j['end']}) got adapter B on both ends, "
                    f"so the array grows to A on both sides | {verdict(zm)} and writes one S-read, {sread_text(zm)}; the B'..A' arm is not written | " + ", ".join(f), s, mk))
    zm = picks["double"]
    s, f, mk = render(zm, 0, len(seqs[zm]))
    mm = seg.filter(pl.col("zm") == zm)
    tt, rr = (mm.filter(pl.col("end_class") == k).row(0, named=True) for k in ("tso_tso", "r1_r1"))
    x = tt["left_token"].rstrip("'")
    # every double mirror of the hash sample, one TSO-TSO and one R1-R1 molecule with the same letter; its S-reads covering at least 80% of the TSO-TSO molecule, with their fates
    hseg = (pl.read_parquet(hash_dir / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "left_token", "right_token"])
            .filter(pl.col("source") == "hifi").drop("source"))
    dms = skera.double_mirrors(hseg)
    tto = hseg.filter((pl.col("end_class") == "tso_tso") & pl.col("zm").is_in(dms)).select("zm", "start", "end")
    hsk = pl.read_parquet(hash_dir / "segments.parquet", columns=["zm", "di", "qs", "qe"]).filter(pl.col("zm").is_in(dms))
    lib = (skera.sread_overlaps(tto, hsk).filter(pl.col("ov") >= skera.COVER)
           .join(pl.read_parquet(hash_dir / "fate.parquet", columns=["zm", "di", "lima_pass", "refine_asrun", "mapped_fltnc"]), on=["zm", "di"]))
    own = lib.filter(pl.col("zm") == zm).row(0, named=True)
    held = mm.filter((pl.col("start") < own["qe"]) & (pl.col("end") > own["qs"])).height
    fate = ("lima passes it" if own["lima_pass"] else "lima rejects it") + (", isoseq refine keeps it" if own["refine_asrun"] else ", isoseq refine removes it") + \
           (" and it is in the mapped BAM" if own["mapped_fltnc"] else ", so it never reaches the mapped BAM")
    records.append((f"double_mirror {SAMPLE} zm={zm} len={len(s)} | as sequenced: {chain_as_read(mm)} (reverse complement: A .. {x} [R1-R1] {x}' [TSO-TSO] {x} .. Q): "
                    f"a TSO-TSO molecule ({tt['start'] + 1}-{tt['end']}) and an R1-R1 molecule ({rr['start'] + 1}-{rr['end']}) with the same adapter {x} sit side by side; "
                    f"the arm on the TSO-TSO side grows to Q, the one on the R1-R1 side to A, so the array holds 10 molecules whatever the letter | "
                    f"{verdict(zm)} and writes {len(sreads(zm))} S-reads: {sread_text(zm)}. The one at {own['qs'] + 1}-{own['qe']} holds {held} molecules, the TSO-TSO and "
                    f"the R1-R1 molecule among them, yet starts with a TSO and ends with Read1 like one proper molecule: {fate}. In the {dms.len():,} double mirrors of the hash sample "
                    f"there are {lib.height:,} such S-reads: lima passes {lib['lima_pass'].mean():.0%}, isoseq refine keeps {lib['refine_asrun'].mean():.0%} | " + ", ".join(f), s, mk))
    zm = picks["false_cut"]
    e1, q2 = cuts[zm]
    mas = {n: (a, b) for n, a, b in features(zm) if n in LET}
    lo, hi = mas["C"][0], mas["E"][1]
    win = seqs[zm][e1 - 5:q2 + 5]
    aln = edlib.align(D, win, mode="HW", task="path")
    site = (e1 - 5 + aln["locations"][0][0], e1 - 5 + aln["locations"][0][1] + 1)
    s, f, mk = render(zm, lo, hi, [("skera_D", *site)])
    f = [x.replace("D ", "D(real) ", 1) if x.startswith("D ") else x for x in f]
    records.append((f"skera_false_cut {SAMPLE} zm={zm} read positions {lo + 1}-{hi} of {len(seqs[zm])} (adapter C to adapter E) | skera skips the real adapter D "
                    f"(exact match) and calls D at {seqs[zm][site[0]:site[1]].upper()} inside the next molecule, {aln['editDistance']} edits from D {D} | "
                    f"skera S-reads here: {sread_text(zm, lo, hi)}: the first is a chimera (C..D molecule + real D + start of the D..E molecule), the second a "
                    f"fragment with only the TSO end; across the library lima passes 0.3% of such chimeras and none of the fragments | " + ", ".join(f), s, mk))

    with open(fasta_path, "w") as fh:
        for header, s, _ in records:
            fh.write(f">{legend(header)}\n" + "\n".join(s[i:i + WRAP] for i in range(0, len(s), WRAP)) + "\n")

    page = [HEAD]
    for header, s, mk in records:
        parts = header.split(" | ")
        name = parts[0].split()[0]
        page.append(f"<h2>{TITLE[name]}</h2><p class='note'>{parts[0]}</p><ul>" + "".join(f"<li>{x}</li>" for x in parts[1:-1]) + "</ul>")
        page.append(f"<pre>{html_sequence(s, mk, WRAP, EXAMPLE_LABEL)}</pre>")
    page.append("</main></body></html>")
    html_path.write_text("\n".join(page))
    print(fasta_path, html_path, picks, [len(s) for _, s, _ in records])


if __name__ == "__main__":
    main()
