"""Sequence patterns and thresholds for annotation."""

from dataclasses import dataclass


# MAS adapter sequences
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

# TSO (terminal oligo) and Read1 sequences
TSO_CORE = "AAGCAGTGGTATCAACGCAGAGTAC"
TSO_EXT = "ATGGG"
R1 = "ACACGACGCTCTTCCGATCT"

# Bead-side linkers
LINKER1 = "GTGA"
LINKER2 = "GACA"
CLS_LEN = 9
UMI_LEN = 8

# Linker scan parameters
LINKER1_OFFSET = 29
LINKER1_MAX_SHIFT = 3
LINKER2_AFTER_LINKER1 = 13
LINKER_MAX_MISMATCH = 1

# End window size and short segment cutoff
END_WINDOW = 150
SHORT_SEGMENT = 300

# PolyT parameters
POLYT_WINDOW = 8
POLYT_MAX_NONT = 1
POLYT_START_SLACK = 2
POLYT_FALLBACK_SEARCH = 60
POLYT_REPORT_CUTOFFS = (5, 8, 12, 20)

# Max edits per pattern kind for HiFi reads.
HIFI_MAX_EDITS = {"MAS": 2, "TSO": 3, "R1": 3}
# Fail reads (1 to 2 passes): relaxed thresholds, chosen on gates.synthetic_benchmark at 5, 8 and 12% error
# (sweep of MAS 2 to 4 and TSO and R1 3 to 6 on 3,000 mini S-reads per sample). Per kind, the smallest
# max_edits with recall of at least 0.95 at 8% error and precision of at least 0.99 at all three rates: TSO 4 (recall at 8%
# 0.966 to 0.971, against 0.887 to 0.894 at 3). MAS and R1 keep the HiFi value because no setting meets both: MAS at 3 and
# R1 at 4 reach the recall, but their precision falls to 0.985.
FAIL_MAX_EDITS = {"MAS": 2, "TSO": 4, "R1": 3}


@dataclass(frozen=True)
class Pattern:
    """A sequence pattern to search for."""
    name: str              # "A".."H", "Q", "TSO", "R1"
    kind: str              # "MAS" | "TSO" | "R1"
    seq: str               # forward sequence
    max_edits: int         # max edit distance threshold


def revcomp(s: str) -> str:
    """Reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(c, c) for c in reversed(s))


def all_patterns(max_edits: dict[str, int] = HIFI_MAX_EDITS, chem=None) -> list[Pattern]:
    """All 11 patterns: adapters in chem.order, then TSO and R1, with max edits per kind
    (HiFi thresholds by default). chem defaults to KINNEX8_SC3 (the Kinnex 8-fold adapters
    and sc3 primers)."""
    if chem is None:
        chem = KINNEX8_SC3
    patterns = []

    for letter in chem.order:
        patterns.append(Pattern(
            name=letter,
            kind="MAS",
            seq=chem.adapters[letter],
            max_edits=max_edits["MAS"],
        ))

    patterns.append(Pattern(
        name="TSO",
        kind="TSO",
        seq=chem.tso,
        max_edits=max_edits["TSO"],
    ))

    patterns.append(Pattern(
        name="R1",
        kind="R1",
        seq=chem.r1,
        max_edits=max_edits["R1"],
    ))

    return patterns


def primer_patterns() -> list[Pattern]:
    """The two primer patterns, TSO and R1."""
    return [
        Pattern(name="TSO", kind="TSO", seq=TSO_CORE, max_edits=3),
        Pattern(name="R1", kind="R1", seq=R1, max_edits=3),
    ]


@dataclass(frozen=True)
class Chemistry:
    """Adapters (name -> sequence, in array order) and the two cDNA primers whose hits validate an adapter."""
    adapters: dict
    order: tuple
    tso: str
    r1: str

    def __hash__(self):
        return hash((tuple(sorted(self.adapters.items())), self.order, self.tso, self.r1))


KINNEX8_SC3 = Chemistry(adapters=dict(MAS), order=tuple(MAS_ORDER), tso=TSO_CORE, r1=R1)
PRIMER_PRESETS = {"sc3": (TSO_CORE, R1)}          # single-cell 3': TSO core and partial Illumina Read1 (BD Rhapsody, 10x 3', ArgenTag)
_DNA = set("ACGT")


def read_adapter_fasta(path) -> tuple[dict, tuple]:
    """Adapters from a FASTA in skera's format: names and sequences, in the order the adapters occur in an array."""
    adapters, order, name = {}, [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            order.append(name)
            adapters[name] = ""
        elif line:
            adapters[name] += line.upper()
    if not order or any(not s or set(s) - _DNA for s in adapters.values()):
        raise ValueError(f"{path}: not an adapter FASTA (names and A/C/G/T sequences)")
    return adapters, tuple(order)


def chemistry_from(adapters_fasta, primers: str) -> Chemistry:
    """Chemistry from the reader command's options: an adapter FASTA (None: Kinnex 8-fold) and a primer preset or 'TSO,READ1'."""
    adapters, order = read_adapter_fasta(adapters_fasta) if adapters_fasta else (dict(MAS), tuple(MAS_ORDER))
    if primers in PRIMER_PRESETS:
        tso, r1 = PRIMER_PRESETS[primers]
    else:
        parts = primers.upper().split(",")
        if len(parts) != 2 or any(len(p) < 15 or set(p) - _DNA for p in parts):
            raise ValueError(f"--primers: a preset ({', '.join(PRIMER_PRESETS)}) or two sequences 'TSO,READ1' of at least 15 nt")
        tso, r1 = parts
    return Chemistry(adapters=adapters, order=order, tso=tso, r1=r1)
