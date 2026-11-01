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


def all_patterns(max_edits: dict[str, int] = HIFI_MAX_EDITS) -> list[Pattern]:
    """All 11 patterns: MAS A-H-Q, then TSO and R1, with max edits per kind (HiFi thresholds by default)."""
    patterns = []

    for letter in MAS_ORDER:
        patterns.append(Pattern(
            name=letter,
            kind="MAS",
            seq=MAS[letter],
            max_edits=max_edits["MAS"],
        ))

    patterns.append(Pattern(
        name="TSO",
        kind="TSO",
        seq=TSO_CORE,
        max_edits=max_edits["TSO"],
    ))

    patterns.append(Pattern(
        name="R1",
        kind="R1",
        seq=R1,
        max_edits=max_edits["R1"],
    ))

    return patterns


def primer_patterns() -> list[Pattern]:
    """The two primer patterns, TSO and R1."""
    return [
        Pattern(name="TSO", kind="TSO", seq=TSO_CORE, max_edits=3),
        Pattern(name="R1", kind="R1", seq=R1, max_edits=3),
    ]
