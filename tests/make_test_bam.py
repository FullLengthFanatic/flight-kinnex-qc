"""Builds the small synthetic HiFi and S-read BAMs `tests/test_cli_run.py` runs `kinnex-qc run` against:
three complete arrays (one with a false cut around D), two TSO-TSO mirrors at H and one R1-R1 mirror at
B. Sequences come from tests/synth.py's array()/molecule()/rand(), on the shared synth.rng; build() seeds
it with a fixed constant and restores its prior state afterward, so the BAM content is the same regardless
of what other tests drew from synth.rng first, and this call leaves their own draws undisturbed.
"""
from pathlib import Path

import pysam

from kinnex_qc.patterns import MAS_ORDER, R1, TSO_CORE, TSO_EXT, revcomp
from synth import array, molecule, rand, rng

CDNA_RANGE = (400, 900)          # cDNA length range for every molecule (brief: "random 400 to 900 bp")
MAS_LEN = {a: (14 if a == "Q" else 17) for a in MAS_ORDER}
FALSE_CUT_OFFSET = 400            # bp into the D-E molecule where the first replacement S-read ends
FALSE_CUT_GAP = 14                # bp between the two replacement S-reads (skera's false cut)
SEED = 11                         # fixed seed for the BAM content, independent of synth.rng's session state


def tso_tso_molecule(n):
    """Two fused 5' ends, no R1: TSO head, random cDNA, TSO' tail."""
    return TSO_CORE + TSO_EXT + rand(n) + revcomp(TSO_CORE + TSO_EXT)


def r1_r1_molecule(n):
    """Two fused 3' ends, no TSO: R1 head, random cDNA, R1' tail."""
    return R1 + rand(n) + revcomp(R1)


def _cdna_mols(n):
    return [molecule(rng.randint(*CDNA_RANGE)) for _ in range(n)]


def _record(name, seq, tags):
    a = pysam.AlignedSegment()
    a.query_name = name
    a.query_sequence = seq
    a.flag = 4
    for k, v, t in tags:
        a.set_tag(k, v, value_type=t)
    return a


def _hifi_record(zm, seq, np=20, rq=0.999):
    return _record(f"movie/{zm}/ccs", seq, [("zm", zm, "i"), ("np", np, "i"), ("rq", rq, "f")])


def _sread_record(zm, qs, qe, dl, dr, seq):
    return _record(f"movie/{zm}/ccs/{qs}_{qe}", seq,
                   [("zm", zm, "i"), ("qs", qs, "i"), ("qe", qe, "i"), ("dl", dl, "i"), ("dr", dr, "i")])


def _header(pg):
    return {"HD": {"VN": "1.6", "SO": "unknown"},
            "RG": [{"ID": "rg1", "PL": "PACBIO", "DS": "READTYPE=CCS"}],
            "PG": pg}


def _complete_array_sreads(zm, seq, starts, false_cut):
    """One S-read per proper molecule, adapter end to next adapter start; with a false cut around D
    (index 3) when false_cut is True: the first replacement S-read ends ~400 bp into the D-E molecule
    (dr = index of D), the second starts 14 bp later (dl = index of D)."""
    d = MAS_ORDER.index("D")
    out = []
    for i in range(len(MAS_ORDER) - 1):
        if false_cut and i == d - 1:        # C-D: stop inside D-E instead of at D's end
            qs, qe = starts[i] + MAS_LEN[MAS_ORDER[i]], starts[d] + MAS_LEN["D"] + FALSE_CUT_OFFSET
        elif false_cut and i == d:          # D-E: start late instead of at D's end
            qs, qe = starts[d] + MAS_LEN["D"] + FALSE_CUT_OFFSET + FALSE_CUT_GAP, starts[i + 1]
        else:
            qs, qe = starts[i] + MAS_LEN[MAS_ORDER[i]], starts[i + 1]
        out.append(_sread_record(zm, qs, qe, i, i + 1, seq[qs:qe]))
    return out


def build(out_dir: Path) -> tuple[Path, Path]:
    """Writes hifi.bam and sreads.bam into out_dir; returns their paths. Draws from synth.rng under a fixed
    seed, saving and restoring its state, so the BAM content does not depend on draws other tests made on
    the shared rng before this ran (and this call does not disturb those tests' own draws)."""
    hifi_path, sreads_path = out_dir / "hifi.bam", out_dir / "sreads.bam"
    hifi_records, sread_records = [], []

    state = rng.getstate()
    rng.seed(SEED)
    try:
        for zm in (1, 2, 3):                                 # three complete arrays A..Q
            seq, starts = array(MAS_ORDER, mols=_cdna_mols(len(MAS_ORDER) - 1))
            hifi_records.append(_hifi_record(zm, seq))
            sread_records += _complete_array_sreads(zm, seq, starts, false_cut=(zm == 1))

        for zm in (4, 5):                                    # two TSO-TSO mirrors at H: Q' H' H Q
            order = ["Q'", "H'", "H", "Q"]
            mols = [molecule(rng.randint(*CDNA_RANGE)), tso_tso_molecule(rng.randint(*CDNA_RANGE)),
                    molecule(rng.randint(*CDNA_RANGE))]
            seq, starts = array(order, mols=mols)
            hifi_records.append(_hifi_record(zm, seq))
            qs, qe = starts[2] + MAS_LEN["H"], starts[3]     # the arm after the junction: H to Q
            sread_records.append(_sread_record(zm, qs, qe, MAS_ORDER.index("H"), MAS_ORDER.index("Q"), seq[qs:qe]))

        zm = 6                                               # one R1-R1 mirror at B: A B B' A'
        order = ["A", "B", "B'", "A'"]
        mols = [molecule(rng.randint(*CDNA_RANGE)), r1_r1_molecule(rng.randint(*CDNA_RANGE)),
                molecule(rng.randint(*CDNA_RANGE))]
        seq, starts = array(order, mols=mols)
        hifi_records.append(_hifi_record(zm, seq))
        qs, qe = starts[2] + MAS_LEN["B"], starts[3]         # the arm after the junction: B' to A'
        sread_records.append(_sread_record(zm, qs, qe, MAS_ORDER.index("B"), MAS_ORDER.index("A"), seq[qs:qe]))
    finally:
        rng.setstate(state)

    with pysam.AlignmentFile(str(hifi_path), "wb", header=_header([{"ID": "ccs", "PN": "ccs"}])) as bam:
        for r in hifi_records:
            bam.write(r)
    with pysam.AlignmentFile(str(sreads_path), "wb",
                              header=_header([{"ID": "skera", "PN": "skera", "VN": "26.1.0"}])) as bam:
        for r in sread_records:
            bam.write(r)
    return hifi_path, sreads_path
