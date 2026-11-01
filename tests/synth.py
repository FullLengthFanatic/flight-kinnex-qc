"""Hand-built reads as the frames the pipeline writes, for tests with known answers."""
import polars as pl


def frames(reads: dict[int, dict]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """reads: zm -> {"adapters": [(name, strand, start, end, edits)], "molecules": [(start, end, end_class, internal, left, right)],
    "sreads": [(qs, qe, dl, dr)]}. Returns (features, molecules, sreads) with the study's column names and dtypes."""
    f, m, s = [], [], []
    for zm, r in reads.items():
        f += [{"zm": zm, "kind": "MAS", "name": n, "strand": st, "start": a, "end": b, "edits": e} for n, st, a, b, e in r["adapters"]]
        m += [{"zm": zm, "start": a, "end": b, "end_class": c, "internal": i, "left_token": lt, "right_token": rt}
              for a, b, c, i, lt, rt in r["molecules"]]
        s += [{"zm": zm, "di": k, "qs": a, "qe": b, "dl": dl, "dr": dr} for k, (a, b, dl, dr) in enumerate(r.get("sreads", []))]
    feats = pl.DataFrame(f, schema={"zm": pl.UInt32, "kind": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8, "start": pl.Int32,
                                    "end": pl.Int32, "edits": pl.Int8})
    mols = pl.DataFrame(m, schema={"zm": pl.UInt32, "start": pl.Int32, "end": pl.Int32, "end_class": pl.Utf8, "internal": pl.Boolean,
                                   "left_token": pl.Utf8, "right_token": pl.Utf8})
    srs = pl.DataFrame(s, schema={"zm": pl.UInt32, "di": pl.Int16, "qs": pl.Int32, "qe": pl.Int32, "dl": pl.Int8, "dr": pl.Int8})
    return feats, mols, srs


def complete_with_false_cut() -> dict:
    """Read 1: adapters C(0-17) D(1017-1034) E(2034-2051); molecules C-D and D-E proper; skera skips D and cuts at 1500-1514."""
    return {1: {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0), ("E", "+", 2034, 2051, 0)],
                "molecules": [(17, 1017, "proper", False, "C", "D"), (1034, 2034, "proper", False, "D", "E")],
                "sreads": [(17, 1500, 2, 3), (1514, 2034, 3, 4)]}}


def tso_tso_mirror() -> dict:
    """Read 2: Q' H' [TSO-TSO] H Q, skera writes only the H..Q molecule."""
    return {2: {"adapters": [("Q", "-", 0, 14, 0), ("H", "-", 1014, 1031, 0), ("H", "+", 3031, 3048, 0), ("Q", "+", 4048, 4062, 0)],
                "molecules": [(14, 1014, "proper", False, "Q'", "H'"), (1031, 3031, "tso_tso", False, "H'", "H"),
                              (3048, 4048, "proper", False, "H", "Q")],
                "sreads": [(3048, 4048, 7, 8)]}}


def double_mirror() -> dict:
    """Read 3: A [A-B] B [R1-R1] B' [TSO-TSO] B [B-C] C (shortened)."""
    return {3: {"adapters": [("A", "+", 0, 17, 0), ("B", "+", 500, 517, 0), ("B", "-", 1017, 1034, 0), ("B", "+", 1534, 1551, 0),
                             ("C", "+", 2051, 2068, 0)],
                "molecules": [(17, 500, "proper", False, "A", "B"), (517, 1017, "r1_r1", False, "B", "B'"),
                              (1034, 1534, "tso_tso", False, "B'", "B"), (1551, 2051, "proper", False, "B", "C")]}}
