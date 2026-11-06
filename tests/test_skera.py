import json
import math
import random
from pathlib import Path

import polars as pl
import pytest

from kinnex_qc import skera
from kinnex_qc.patterns import MAS_ORDER
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

GOLD = Path(__file__).parent / "golden" / "v10_inputs"


def _sreads(rows, schema={"zm": pl.UInt32, "di": pl.Int16, "qs": pl.Int32, "qe": pl.Int32, "dl": pl.Int8, "dr": pl.Int8}):
    return pl.DataFrame(rows, schema=schema, orient="row")


def test_false_cut_found_and_real_adapter_inside():
    f, m, s = frames(complete_with_false_cut())
    fc = skera.false_cuts(s, f)
    assert fc.select("zm", "cut_start", "cut_end", "dr_before").rows() == [(1, 1500, 1514, 3)]
    inside = skera.adapters_inside_sreads(s, f)
    assert inside.select("zm", "di", "name").rows() == [(1, 0, "D")]


def test_cut_at_real_adapter_is_not_false():
    f, m, s = frames(complete_with_false_cut())
    s = s.with_columns(pl.when(pl.col("di") == 0).then(1017).otherwise(pl.col("qe")).alias("qe"),
                       pl.when(pl.col("di") == 1).then(1034).otherwise(pl.col("qs")).alias("qs"))
    assert skera.false_cuts(s, f).height == 0


def test_molecule_coverage_classes():
    f, m, s = frames(complete_with_false_cut())
    cov = skera.molecule_coverage(m, s).sort("start")
    assert cov["cls"].to_list() == ["FUSED", "PARTIAL"]   # S-read 0 covers C-D fully and 47% of D-E; S-read 1 covers 52% of D-E


def test_mirror_one_arm_written():
    f, m, s = frames(tso_tso_mirror())
    cov = skera.molecule_coverage(m, s).filter(pl.col("end_class") == "proper").sort("start")   # pass all molecules of a read
    assert cov["cls"].to_list() == ["NOT_WRITTEN", "CLEAN"]
    st = skera.single_tso_tso_mirrors(m)
    assert st.select("zm", "letter").rows() == [(2, "H")]


def test_sread_overlaps_fractions():
    """Every molecule with every S-read of its ZMW, and the fraction of the molecule the S-read covers."""
    f, m, s = frames(complete_with_false_cut())
    ov = skera.sread_overlaps(m, s)
    assert ov.columns == [*m.columns, "di", "qs", "qe", "dl", "dr", "ov"]
    assert ov.select("start", "di", "ov").rows() == [(17, 0, 1.0), (17, 1, 0.0), (1034, 0, 0.466), (1034, 1, 0.52)]


def test_best_cover_is_looser_than_clean():
    f, m, s = frames(complete_with_false_cut())
    bc = skera.best_cover(m, s).sort("start")
    assert bc["cover"][0] == 1.0          # "written" under the downstream test's rule although the S-read is a chimera


def test_double_mirror():
    f, m, s = frames(double_mirror())
    assert skera.double_mirrors(m).to_list() == [3]
    assert skera.single_tso_tso_mirrors(m).height == 0


def test_passing_from_labels_forward_and_reversed():
    s = _sreads([
        (10, 0, 0, 17, 0, 1), (10, 1, 17, 34, 1, 2), (10, 2, 34, 51, 2, 3), (10, 3, 51, 68, 3, 4),
        (10, 4, 68, 85, 4, 5), (10, 5, 85, 102, 5, 6), (10, 6, 102, 119, 6, 7), (10, 7, 119, 136, 7, 8),
        (11, 0, 0, 17, 8, 7), (11, 1, 17, 34, 7, 6), (11, 2, 34, 51, 6, 5), (11, 3, 51, 68, 5, 4),
        (11, 4, 68, 85, 4, 3), (11, 5, 85, 102, 3, 2), (11, 6, 102, 119, 2, 1), (11, 7, 119, 136, 1, 0),
        (12, 0, 0, 17, 0, 1), (12, 1, 17, 34, 1, 2), (12, 2, 34, 51, 2, 4),
    ])
    out = dict(skera.passing_from_labels(s, order_len=9).select("zm", "passing").rows())
    assert out == {10: True, 11: True, 12: False}


def test_passing_from_labels_mirror_arm_alone_is_not_passing():
    f, m, s = frames(tso_tso_mirror())
    out = dict(skera.passing_from_labels(s, order_len=9).select("zm", "passing").rows())
    assert out == {2: False}


def test_skipped_adapters_matches_false_cut():
    f, m, s = frames(complete_with_false_cut())
    fc = skera.false_cuts(s, f)
    sk = skera.skipped_adapters(fc, s, f)
    assert sk.select("zm", "cut_start", "missed", "in_sread", "dist").rows() == [(1, 1500, "D", 1, 490)]


def test_false_cut_stats_uses_the_passed_in_passing_and_fc():
    """Fix round 2, finding 1: false_cut_stats must take its counts from the passing/fc frames it is
    given, not recompute passing_from_labels/false_cuts itself (the runner now computes both once and
    passes them in). A caller-supplied fc with the one real false cut duplicated must show up as 2
    false cuts: if false_cut_stats instead recomputed false_cuts(s, f) internally and ignored the fc
    argument, this would still read 1."""
    f, m, s = frames(complete_with_false_cut())
    read_summary = pl.DataFrame({"zm": [1], "primary_pattern": ["complete"]}, schema={"zm": pl.UInt32, "primary_pattern": pl.Utf8})
    # passing is given directly (zm 1's two partial S-reads would not derive True on their own; the
    # point here is only that false_cut_stats must use the frame it is given, not re-derive it).
    passing = pl.DataFrame({"zm": [1], "passing": [True]}, schema={"zm": pl.UInt32, "passing": pl.Boolean})
    fc = skera.false_cuts(s, f)
    stats = skera.false_cut_stats(read_summary, f, s, passing=passing, fc=fc, order=MAS_ORDER)
    assert stats["false_cuts"] == 1
    stats_dup = skera.false_cut_stats(read_summary, f, s, passing=passing, fc=pl.concat([fc, fc]), order=MAS_ORDER)
    assert stats_dup["false_cuts"] == 2


def test_mirror_regions():
    """Every molecule of a single TSO-TSO mirror read with its columns and region: the TSO-TSO molecule (1031-3031) is
    junction; a molecule ending at or before 1031 is before, one starting at or after 3031 after, any other other.
    zm 1 (no mirror junction) and zm 3 (two junctions) are not single TSO-TSO mirror reads."""
    reads = {**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()}
    reads[2]["molecules"] += [(1000, 1031, "proper", False, None, None), (2000, 2500, "proper", True, None, None),
                              (3031, 3040, "none", False, None, None)]
    f, m, s = frames(reads)
    reg = skera.mirror_regions(m)
    assert reg.columns == [*m.columns, "region"]
    assert reg.sort("start").select("zm", "start", "end", "region").rows() == [
        (2, 14, 1014, "before"), (2, 1000, 1031, "before"), (2, 1031, 3031, "junction"), (2, 2000, 2500, "other"),
        (2, 3031, 3040, "after"), (2, 3048, 4048, "after")]


def test_unwritten_head():
    """zm 1: a complete forward read (S-reads A..Q), not flagged. zm 2: one S-read H..Q starting 3,048 bp in, its span
    1,000 bp: head 3.048 mean written molecules, flagged. zm 3: H..Q with a head of 1.5 molecules, reaches Q but not A
    only. zm 4: H..Q in reverse adapter order (dl > dr), not forward. zm 5: two S-reads G..H..Q, head 2,000 bp over a
    2,000 bp span of 2 S-reads, exactly 2 molecules: flagged (the bound is inclusive)."""
    reads = {1: {"adapters": [], "molecules": [], "sreads": [(17 + 1000 * k, 1000 * (k + 1), k, k + 1) for k in range(8)]},
             2: {"adapters": [], "molecules": [], "sreads": [(3048, 4048, 7, 8)]},
             3: {"adapters": [], "molecules": [], "sreads": [(1500, 2500, 7, 8)]},
             4: {"adapters": [], "molecules": [], "sreads": [(3048, 4048, 8, 7)]},
             5: {"adapters": [], "molecules": [], "sreads": [(2000, 3000, 6, 7), (3000, 4000, 7, 8)]}}
    _, _, s = frames(reads)
    z = skera.unwritten_head(s)
    assert z.columns == ["zm", "first", "last", "n", "head", "span", "fwd", "head_mol", "q_not_a", "flag"]
    assert z.rows() == [
        (1, 0, 8, 8, 17, 7983, True, 17 * 8 / 7983, False, False),
        (2, 7, 8, 1, 3048, 1000, True, 3.048, True, True),
        (3, 7, 8, 1, 1500, 1000, True, 1.5, True, False),
        (4, 7, 8, 1, 3048, 1000, False, 3.048, True, False),
        (5, 6, 8, 2, 2000, 2000, True, 2.0, True, True)]
    assert skera.unwritten_head(s, head_mol=3.1)["flag"].to_list() == [False, False, False, False, False]


def test_mirror_arm_classes_regions():
    f, m, s = frames(tso_tso_mirror())
    arm = skera.mirror_arm_classes(m, s).sort("region")
    assert arm.select("zm", "region", "cls").rows() == [(2, "after", "CLEAN"), (2, "before", "NOT_WRITTEN")]


def test_mirror_arm_stats_shares():
    f, m, s = frames(tso_tso_mirror())
    stats = skera.mirror_arm_stats(m, s)
    assert stats["after_clean_share"] == 1.0
    assert stats["before_clean_share"] == 0.0
    assert stats["after_counts"] == {"CLEAN": 1, "FUSED": 0, "PARTIAL": 0, "NOT_WRITTEN": 0}
    assert stats["before_counts"] == {"CLEAN": 0, "FUSED": 0, "PARTIAL": 0, "NOT_WRITTEN": 1}


def test_budget_stats_by_hand():
    reads = {**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()}
    f, m, s = frames(reads)
    read_summary = pl.DataFrame({"zm": [1, 2, 3]}, schema={"zm": pl.UInt32})
    cls = skera.budget_classes(m, s)
    stats = skera.budget_stats(read_summary, cls, order_len=9)
    assert stats["molecules"] == pytest.approx(3.0)
    assert stats["unfilled_slots"] == pytest.approx(5.0)
    assert stats["artifact_tso_tso"] == pytest.approx(2 / 3)
    assert stats["artifact_r1_r1"] == pytest.approx(1 / 3)
    assert stats["artifact_one_primer_or_none"] == pytest.approx(0.0)
    assert stats["artifact_fused"] == pytest.approx(0.0)
    assert stats["artifact"] == pytest.approx(1.0)
    assert stats["usable"] == pytest.approx(2.0)
    assert stats["usable_clean"] == pytest.approx(1 / 3)
    assert stats["usable_not_clean"] == pytest.approx(5 / 3)


def test_zmw_with_one_sread_has_no_false_cut_and_its_one_molecule_is_clean():
    """zm 8: one S-read over its one molecule, CLEAN. zm 9: one S-read over both its molecules, so each is FUSED, not
    CLEAN. Neither ZMW has a gap between two S-reads, so neither has a false cut."""
    reads = {8: {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0)],
                 "molecules": [(17, 1017, "proper", False, "C", "D")],
                 "sreads": [(17, 1017, 2, 3)]},
             9: {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0), ("E", "+", 2034, 2051, 0)],
                 "molecules": [(17, 1017, "proper", False, "C", "D"), (1034, 2034, "proper", False, "D", "E")],
                 "sreads": [(17, 2034, 2, 4)]}}
    f, m, s = frames(reads)
    assert skera.false_cuts(s, f).height == 0
    assert skera.molecule_coverage(m, s).select("zm", "cls").rows() == [(8, "CLEAN"), (9, "FUSED"), (9, "FUSED")]


def test_zmw_with_sreads_and_no_molecules_gives_no_coverage_rows():
    """zm 9 has two S-reads and no molecule: molecule_coverage and best_cover give rows for zm 1's two molecules only."""
    reads = {**complete_with_false_cut(),
             9: {"adapters": [("C", "+", 0, 17, 0)], "molecules": [], "sreads": [(17, 900, 2, 3), (920, 1800, 3, 4)]}}
    f, m, s = frames(reads)
    assert skera.molecule_coverage(m, s).select("zm", "start").rows() == [(1, 17), (1, 1034)]
    assert skera.best_cover(m, s).select("zm", "start").rows() == [(1, 17), (1, 1034)]


def test_best_cover_is_zero_for_a_molecule_whose_zmw_has_no_sread():
    """zm 7 has one molecule and no S-read, so nothing covers it: cover 0. zm 1 has both and keeps its covers (S-read 0
    covers C-D fully; D-E is covered 47% by S-read 0 and 52% by S-read 1)."""
    reads = {**complete_with_false_cut(),
             7: {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0)],
                 "molecules": [(17, 1017, "proper", False, "C", "D")], "sreads": []}}
    f, m, s = frames(reads)
    assert skera.best_cover(m, s).select("zm", "start", "cover").rows() == [(1, 17, 1.0), (1, 1034, 0.52), (7, 17, 0.0)]


def test_every_public_function_on_empty_frames():
    """Frames with their columns and no rows give empty results with the documented columns, and the summary dicts NaN
    shares, zero counts and empty letter counts, not a crash."""
    f, m, s = frames({})
    rs = pl.DataFrame(schema={"zm": pl.UInt32, "primary_pattern": pl.Utf8})
    passing, fc = skera.passing_from_labels(s), skera.false_cuts(s, f)
    cls = skera.budget_classes(m, s)
    adapter_cols = ["zm", "di", "name", "strand", "start", "end", "edits"]
    coverage_cols = ["mid", *m.columns, "maxov", "clean_di", "cls"]
    for name, out, cols in [
        ("false_cuts", fc, ["zm", "di_before", "di_after", "cut_start", "cut_end", "dr_before"]),
        ("adapters_inside_sreads", skera.adapters_inside_sreads(s, f), adapter_cols),
        ("adapters_inside_sreads strict", skera.adapters_inside_sreads(s, f, strict=True), adapter_cols),
        ("molecule_coverage", skera.molecule_coverage(m, s), coverage_cols),
        ("best_cover", skera.best_cover(m, s), ["zm", "start", "end", "cover"]),
        ("sread_overlaps", skera.sread_overlaps(m, s), [*m.columns, "di", "qs", "qe", "dl", "dr", "ov"]),
        ("mirror_junctions", skera.mirror_junctions(m), [*m.columns, "letter"]),
        ("single_tso_tso_mirrors", skera.single_tso_tso_mirrors(m), ["zm", "js", "je", "letter"]),
        ("mirror_regions", skera.mirror_regions(m), [*m.columns, "region"]),
        ("passing_from_labels", passing, ["zm", "passing"]),
        ("unwritten_head", skera.unwritten_head(s), ["zm", "first", "last", "n", "head", "span", "fwd", "head_mol", "q_not_a", "flag"]),
        ("skipped_adapters", skera.skipped_adapters(fc, s, f), ["zm", "cut_start", "missed", "in_sread", "dist"]),
        ("mirror_arm_classes", skera.mirror_arm_classes(m, s), ["zm", "region", "cls"]),
        ("budget_classes", cls, [*coverage_cols, "usable"]),
        ("budget_per_read", skera.budget_per_read(cls), ["zm", "molecules", "artifact_tso_tso", "artifact_r1_r1",
                                                         "artifact_one_primer_or_none", "artifact_fused", "usable",
                                                         "usable_clean", "usable_not_clean"]),
    ]:
        assert (out.height, out.columns) == (0, cols), name
    dm = skera.double_mirrors(m)
    assert (dm.name, dm.len()) == ("zm", 0)

    stats = skera.false_cut_stats(rs, f, s, passing=passing, fc=fc)
    assert {k: stats[k] for k in ("complete_reads", "complete_passing_reads", "false_cuts", "false_cut_letters",
                                  "skipped_adapter_letters")} == {"complete_reads": 0, "complete_passing_reads": 0, "false_cuts": 0,
                                                                  "false_cut_letters": {}, "skipped_adapter_letters": {}}
    assert math.isnan(stats["false_cut_reads_share"]) and math.isnan(stats["false_cuts_per_read"])
    arms = skera.mirror_arm_stats(m, s)
    assert arms["after_counts"] == arms["before_counts"] == {"CLEAN": 0, "FUSED": 0, "PARTIAL": 0, "NOT_WRITTEN": 0}
    assert math.isnan(arms["after_clean_share"]) and math.isnan(arms["before_clean_share"])
    budget = skera.budget_stats(rs.select("zm"), cls)
    assert list(budget) == ["molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1", "artifact_one_primer_or_none",
                            "artifact_fused", "artifact", "usable", "usable_clean", "usable_not_clean"]
    assert all(math.isnan(v) for v in budget.values())


def _random_reads(n: int = 300, seed: int = 7) -> dict:
    """n random ZMWs for the chunk-count test, as synth.frames takes them (zm 100, 113, 126, ..., spread over every chunk
    count tried): 2 to 9 adapters in MAS order with 300 to 2,000 bp between them, a molecule between each pair, and S-reads
    that follow the molecules within 5 bp, some skipping a molecule, some covering two, some cut in two inside one (a
    false cut). Its own random.Random, so synth.rng's draws stay as other tests expect them."""
    r = random.Random(seed)
    reads = {}
    for i in range(n):
        adapters, pos = [], 0
        for k in range(r.randint(2, len(MAS_ORDER))):
            adapters.append((MAS_ORDER[k], "+", pos, pos + 17, r.randint(0, 2)))
            pos += 17 + r.randint(300, 2000)
        mols = [(a[3], b[2], r.choice(["proper", "tso_tso", "r1_r1", "none"]), r.random() < 0.1, a[0], b[0])
                for a, b in zip(adapters, adapters[1:])]
        sreads, k = [], 0
        while k < len(mols):
            start, end, u = mols[k][0], mols[k][1], r.random()
            if u < 0.15:                                   # not written
                k += 1
                continue
            if u < 0.3 and k + 1 < len(mols):              # one S-read over two molecules
                end = mols[k + 1][1]
                k += 1
            elif u < 0.45:                                 # cut in two inside the molecule
                cut = r.randint(start + 50, end - 50)
                sreads.append((start + r.randint(-5, 5), cut, k, k + 1))
                start = cut + r.randint(0, 20)
            sreads.append((start + r.randint(-5, 5), end + r.randint(-5, 5), k, k + 1))
            k += 1
        reads[100 + 13 * i] = {"adapters": adapters, "molecules": mols, "sreads": sreads}
    return reads


def test_chunked_functions_give_the_same_frames_for_any_chunk_count(monkeypatch):
    """false_cuts, adapters_inside_sreads, molecule_coverage, best_cover and sread_overlaps work through the ZMWs in
    _CHUNKS chunks by zm; 1, 7 and 16 chunks give equal frames on the synth reads plus 300 random ZMWs."""
    f, m, s = frames({**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror(), **_random_reads()})
    results = {}
    for chunks in (1, 7, 16):
        monkeypatch.setattr(skera, "_CHUNKS", chunks)
        results[chunks] = {"false_cuts": skera.false_cuts(s, f), "adapters_inside_sreads": skera.adapters_inside_sreads(s, f),
                           "adapters_inside_sreads strict": skera.adapters_inside_sreads(s, f, strict=True),
                           "molecule_coverage": skera.molecule_coverage(m, s), "best_cover": skera.best_cover(m, s),
                           "sread_overlaps": skera.sread_overlaps(m, s)}
    want = results[16]
    assert all(frame.height > 50 for frame in want.values())
    assert set(want["molecule_coverage"]["cls"]) == {"CLEAN", "FUSED", "PARTIAL", "NOT_WRITTEN"}
    for chunks in (1, 7):
        for name, frame in results[chunks].items():
            assert frame.equals(want[name]), (chunks, name)


@pytest.mark.real
def test_false_cut_letters_and_skipped_adapter_letters_match_v10(local):
    KIN = local["hash_dir"] / "BD144_kin"
    rs = pl.read_parquet(KIN / "read_summary.parquet", columns=["zm", "source", "primary_pattern"]).filter(pl.col("source") == "hifi")
    complete_zm = rs.filter(pl.col("primary_pattern") == "complete")["zm"]
    f = pl.read_parquet(KIN / "read_features.parquet", columns=["zm", "source", "kind", "name", "strand", "start", "end", "edits"]) \
        .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(complete_zm))
    s = pl.read_parquet(KIN / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(complete_zm))
    rs_complete = rs.filter(pl.col("primary_pattern") == "complete").select("zm", "primary_pattern")
    passing = skera.passing_from_labels(s, order_len=len(MAS_ORDER))
    fc = skera.false_cuts(s, f)
    stats = skera.false_cut_stats(rs_complete, f, s, passing=passing, fc=fc, order=MAS_ORDER)
    gold = json.loads((GOLD / "false_cut_anatomy" / "BD144_kin_false_cuts.json").read_text())
    assert stats["complete_passing_reads"] == 1_950_193
    assert stats["false_cuts"] == gold["false_cuts"]
    assert stats["false_cut_reads_share"] == pytest.approx(gold["reads_with_false_cut"], abs=0)
    classes = pl.read_csv(GOLD / "false_cut_anatomy" / "BD144_kin_false_cut_classes.tsv", separator="\t")
    skera_letter = dict(classes.filter(pl.col("field") == "skera_letter").select("value", "false_cuts").rows())
    missed = {(row[0] if row[0] else "none"): row[1]
              for row in classes.filter(pl.col("field") == "missed").select("value", "false_cuts").rows()}
    assert stats["false_cut_letters"] == skera_letter
    assert stats["skipped_adapter_letters"] == missed


@pytest.mark.real
def test_mirror_arms_match_v10_nd(local):
    ND = local["hash_dir"] / "BD144_ND"
    m = pl.read_parquet(ND / "annot_segments.parquet",
                         columns=["zm", "source", "start", "end", "end_class", "internal", "left_token", "right_token"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    s = pl.read_parquet(ND / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"])
    stats = skera.mirror_arm_stats(m, s)
    assert stats["after_clean_share"] == pytest.approx(0.8588701530810475, abs=0)
    assert stats["before_clean_share"] == pytest.approx(0.08099823120678912, abs=0)
    gold = pl.read_csv(GOLD.parent / "v10_tables" / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t") \
        .filter(pl.col("set") == "nd_hash_skera26")
    for region in ("after", "before"):
        want = dict(gold.filter(pl.col("region") == region).select("class", "n").rows())
        assert stats[f"{region}_counts"] == want


@pytest.mark.real
def test_budget_matches_v10_nd(local):
    ND = local["hash_dir"] / "BD144_ND"
    rs = pl.read_parquet(ND / "read_summary.parquet", columns=["zm", "source"]).filter(pl.col("source") == "hifi").select("zm")
    m = pl.read_parquet(ND / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    s = pl.read_parquet(ND / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"])
    cls = skera.budget_classes(m, s)
    stats = skera.budget_stats(rs, cls, order_len=9)
    gold = pl.read_csv(GOLD.parent / "v10_tables" / "fig6_molecule_budget.tsv", separator="\t").filter(pl.col("library") == "ND")
    row = gold.to_dicts()[0]
    for key, gold_key in (("molecules", "molecules"), ("unfilled_slots", "unfilled_slots"),
                          ("artifact_tso_tso", "artifact_tso_tso"), ("artifact_r1_r1", "artifact_r1_r1"),
                          ("artifact_one_primer_or_none", "artifact_one_primer_or_none"),
                          ("artifact_fused", "artifact_fused"), ("usable", "usable"),
                          ("usable_not_clean", "not_written_cleanly")):
        assert stats[key] == pytest.approx(row[gold_key], abs=0), key


@pytest.mark.real
def test_false_cut_rate_kin_matches_v10(local):
    KIN = local["hash_dir"] / "BD144_kin"
    rs = pl.read_parquet(KIN / "read_summary.parquet", columns=["zm", "source", "skera_passing", "primary_pattern"])
    target = rs.filter((pl.col("source") == "hifi") & pl.col("skera_passing") & (pl.col("primary_pattern") == "complete"))["zm"]
    f = pl.read_parquet(KIN / "read_features.parquet", columns=["zm", "source", "kind", "name", "strand", "start", "end", "edits"]) \
        .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(target))
    s = pl.read_parquet(KIN / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(target))
    fc = skera.false_cuts(s, f)
    gold = json.loads((GOLD / "false_cut_anatomy" / "BD144_kin_false_cuts.json").read_text())
    assert fc.height == gold["false_cuts"]
    assert fc["zm"].n_unique() / target.len() == pytest.approx(gold["reads_with_false_cut"], abs=0)
