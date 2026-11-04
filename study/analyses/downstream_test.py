#!/usr/bin/env python3
"""Do the molecules our segmentation finds survive the existing downstream pipeline?

From the first 200,000 BD144_ND HiFi reads (a local BAM of them, MM/ML removed), take 3,000 mirror reads and 3,000
complete reads (our full-read calls, in --truth), cut every molecule we call proper and not fused at our segment
coordinates (kinnex_qc.molecules.write_molecules), and run them through lima --isoseq --peek-guess and isoseq refine
with the existing pipeline's primers, then minimap2 -ax splice:hq -uf to the reference genome (the study's reference is
the GRCh38 primary assembly). Groups: molecules of mirror reads that skera did not write, molecules of mirror reads
that skera did write, and molecules of complete reads. Written by skera here means one delivered S-read covers at least
80% of the molecule (skera.best_cover), looser than the CLEAN rule of mirror_arms.py.
Writes per-group pass rates (downstream_summary.tsv) and per-molecule results (downstream_molecules.tsv).
The molecules are written with the header and read group of a skera S-read BAM of the same cell (--template, whose
read group says READTYPE=SEGMENT), so lima takes each record as a molecule of its own, not one CCS read per ZMW
(observed on the calibration set: lima's output holds one record per molecule it passes, named by that molecule).
--truth holds read_summary, annot_segments and segments (the delivered skera S-reads) for these reads. lima and isoseq
run in the conda environment --pbtools-env (default: pbtools_env of study/study.toml), and so does minimap2 unless
--minimap2 (default: minimap2 of study/study.toml, when set) names an executable. --ref defaults to reference_fasta of study/study.toml and study.local.toml (the placeholder counts as unset). The three tools' versions go to
stderr, the run log.
Usage, from the repository root:
  study/analyses/downstream_test.py --hifi <bam> --template <S-read bam> --truth <dir> [--ref <genome fasta>] --out <dir>
      [--threads 4] [--n 3000] [--pbtools-env NAME|PATH] [--minimap2 PATH] [--score-only]
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import polars as pl
import pysam

from kinnex_qc.io import iter_bam
from kinnex_qc.molecules import USABLE, write_molecules
from kinnex_qc.skera import COVER, best_cover

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402

PRIMERS = Path(__file__).resolve().parents[2] / "resources" / "primers_FlashBD.fasta"
LIMA_FL = "mol.isoseq.FlashBD_5p--FlashBD_3p.bam"
SEED = 7


def choose(hifi: Path, truth: Path, n: int) -> dict[int, str]:
    """Up to n mirror reads and n complete reads of the HiFi BAM, drawn with a fixed seed: zm -> pattern."""
    zms = pl.Series([r.get_tag("zm") for r in iter_bam(hifi)], dtype=pl.UInt32)
    rs = pl.read_parquet(truth / "read_summary.parquet", columns=["zm", "source", "primary_pattern"]).filter(
        (pl.col("source") == "hifi") & pl.col("zm").is_in(zms))
    rng = random.Random(SEED)
    chosen = {}
    for pat in ("mirror", "complete"):
        z = sorted(rs.filter(pl.col("primary_pattern") == pat)["zm"].to_list())
        for x in rng.sample(z, min(n, len(z))):
            chosen[x] = pat
    return chosen


def molecule_groups(truth: Path, chosen: dict[int, str]) -> pl.DataFrame:
    """The proper, unfused molecules of the chosen reads, with the largest share of each that one S-read covers and the
    group it falls in."""
    keys = pl.Series(list(chosen), dtype=pl.UInt32)
    seg = (pl.read_parquet(truth / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal"])
           .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(keys)).filter(USABLE).drop("source")
           .with_columns(pl.col("start", "end").cast(pl.Int64)))
    sk = pl.read_parquet(truth / "segments.parquet", columns=["zm", "qs", "qe"]).filter(pl.col("zm").is_in(keys))
    seg = (seg.join(best_cover(seg.select("zm", "start", "end"), sk), on=["zm", "start", "end"])
           .with_columns(pl.col("zm").replace_strict(chosen, return_dtype=pl.Utf8).alias("pattern")))
    return seg.with_columns(pl.when(pl.col("pattern") == "complete").then(pl.lit("complete reads"))
                            .when(pl.col("cover") >= COVER).then(pl.lit("mirror reads, written by skera"))
                            .otherwise(pl.lit("mirror reads, not written by skera")).alias("group"))


def write_molecules_bam(hifi: Path, seg: pl.DataFrame, template: Path, out: Path) -> dict[str, str]:
    """Write seg's molecules to out (kinnex_qc.molecules.write_molecules); returns each record's name -> group, in the
    BAM's order."""
    write_molecules(hifi, seg, template, out)
    group = {(zm, s, e): g for zm, s, e, g in seg.select("zm", "start", "end", "group").iter_rows()}
    return {r.query_name: group[(r.get_tag("zm"), r.get_tag("qs"), r.get_tag("qe"))] for r in iter_bam(out)}


def molecule_of(name: str, starts: dict) -> str | None:
    """lima and refine rename a read with its trimmed coordinates (movie/zm/ccs/s_e): map it back to our molecule of
    that ZMW whose interval contains the trimmed one."""
    movie, zm, _, se = name.split("/")[:4]
    s, e = (int(x) for x in se.split("_")[:2])
    for a, b, n in starts.get(int(zm), []):
        if a <= s and e <= b:
            return n
    return None


def score(out: Path, names: dict) -> None:
    starts = {}
    for n in names:
        movie, zm, _, se = n.split("/")
        a, b = (int(x) for x in se.split("_"))
        starts.setdefault(int(zm), []).append((a, b, n))
    lima_pass = {molecule_of(r.query_name, starts) for r in pysam.AlignmentFile(str(out / LIMA_FL), "rb", check_sq=False)}
    refine_pass = {molecule_of(r.query_name, starts) for r in pysam.AlignmentFile(str(out / "mol.flnc.bam"), "rb", check_sq=False)}
    mapped = {molecule_of(r.query_name, starts) for r in pysam.AlignmentFile(str(out / "mol.flnc.sam"), "r")
              if not r.is_unmapped and not r.is_secondary and not r.is_supplementary}
    rows = [{"molecule": n, "group": g, "lima_5p3p": n in lima_pass, "refine": n in refine_pass, "mapped": n in mapped}
            for n, g in names.items()]
    t = pl.DataFrame(rows)
    summ = t.group_by("group").agg(pl.len().alias("molecules"), pl.col("lima_5p3p").mean(), pl.col("refine").mean(),
                                   pl.col("mapped").mean()).sort("group")
    summ.write_csv(out / "downstream_summary.tsv", separator="\t")
    t.write_csv(out / "downstream_molecules.tsv", separator="\t")
    print(summ)


def first_line(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()


def run_tools(out: Path, mol_bam: Path, ref: str, threads: int, pbtools_env: str, minimap2: str | None) -> None:
    """lima, isoseq refine, the FASTQ of refine's reads and minimap2's SAM, into out."""
    run = ["conda", "run", "--no-capture-output", "-p" if "/" in pbtools_env else "-n", pbtools_env]
    mm2 = [minimap2] if minimap2 else run + ["minimap2"]
    for line in (first_line(run + ["lima", "--version"]), first_line(run + ["isoseq", "--version"]),
                 "minimap2 " + first_line(mm2 + ["--version"])):
        print(line, file=sys.stderr)
    subprocess.run(run + ["lima", "--isoseq", "--peek-guess", "-j", str(threads), "--log-level", "WARN",
                          str(mol_bam), str(PRIMERS), str(out / "mol.isoseq.bam")], check=True)
    subprocess.run(run + ["isoseq", "refine", "-j", str(threads), "--log-level", "WARN", str(out / LIMA_FL), str(PRIMERS),
                          str(out / "mol.flnc.bam")], check=True)
    fq = out / "mol.flnc.fastq"
    with open(fq, "w") as fh:
        for r in iter_bam(out / "mol.flnc.bam"):
            fh.write(f"@{r.query_name}\n{r.query_sequence}\n+\n{pysam.qualities_to_qualitystring(r.query_qualities)}\n")
    with open(out / "mol.flnc.sam", "w") as fh:
        subprocess.run(mm2 + ["-ax", "splice:hq", "-uf", "--secondary=no", "-t", str(threads), ref, str(fq)],
                       stdout=fh, check=True, stderr=subprocess.DEVNULL)


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hifi", required=True, help="HiFi BAM of the reads")
    ap.add_argument("--template", required=True, help="skera S-read BAM of the same cell, whose header the molecules get")
    ap.add_argument("--truth", required=True, help="directory with read_summary, annot_segments and segments parquets")
    ap.add_argument("--ref", default=config.get("reference_fasta"),
                    help="reference genome FASTA for minimap2 (default: reference_fasta of study/study.toml and study.local.toml)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=3000, help="mirror reads and complete reads drawn, each")
    ap.add_argument("--pbtools-env", default=config["pbtools_env"], help="conda environment with lima and isoseq (a name, or a path)")
    ap.add_argument("--minimap2", default=config.get("minimap2"),
                    help="minimap2 executable (default: minimap2 of study/study.toml, else minimap2 in the --pbtools-env environment)")
    ap.add_argument("--score-only", action="store_true", help="reuse lima, refine and mapping outputs in --out")
    args = ap.parse_args(argv)
    if not args.ref or args.ref.startswith("<"):
        ap.error("no reference genome: give --ref or set reference_fasta in study/study.local.toml")
    out, truth = Path(args.out), Path(args.truth)
    out.mkdir(parents=True, exist_ok=True)
    chosen = choose(Path(args.hifi), truth, args.n)
    seg = molecule_groups(truth, chosen)
    mol_bam = out / "molecules.bam"
    names = write_molecules_bam(Path(args.hifi), seg, Path(args.template), mol_bam)
    print(f"{len(names):,} molecules from {seg['zm'].n_unique():,} reads", file=sys.stderr)
    if not args.score_only:
        run_tools(out, mol_bam, args.ref, args.threads, args.pbtools_env, args.minimap2)
    score(out, names)


if __name__ == "__main__":
    main()
