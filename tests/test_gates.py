"""Tests for study/pipeline/gates.py: known answers on toy frames, the skera agreement against the per-read rule it
replaced, the synthetic benchmark on built molecules, the stage CLI on a toy tree, and the gates on the mini dataset
and on the v10 hash sample."""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from kinnex_qc.patterns import FAIL_MAX_EDITS, HIFI_MAX_EDITS, R1, TSO_CORE, TSO_EXT, revcomp
from kinnex_qc.sampling import keep_zmw_array
from pipeline_helpers import PIPELINE, REPO, bam_records, key_structure, mini_samples, run_stage
from synth import complete_with_false_cut, frames

import gates

sys.path.insert(0, str(REPO / "study"))
from compare_runs import compare_trees  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "v10_inputs" / "hash_0.18"
SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")
HEADER = "sample\tcell\tdepletion\thifi_bam\tfail_bam\tsegmented_bam\tnon_passing_bam\tzmw_metrics\tskera_report\tfltnc_bam\n"
U32 = {"zm": pl.UInt32}
V10_SKERA_TEXT = "reported only (spec 17)"      # v10's threshold text in the frozen gate_skera.json files


def per_read_agreement(segments: pl.DataFrame, features: pl.DataFrame, passing: set[int], tol: int = 5) -> dict:
    """The per-read loop gates.skera_agreement replaced, kept as the reference: a skera-passing ZMW with at least one
    of our HiFi MAS hits agrees when every gap between consecutive S-reads (in qs order) holds a hit within tol bp
    and no hit lies strictly more than tol bp inside an S-read."""
    mas = {}
    for zm, a, b in (features.filter((pl.col("source") == "hifi") & (pl.col("kind") == "MAS"))
                     .select("zm", "start", "end").iter_rows()):
        mas.setdefault(zm, []).append((a, b))
    compared = agreed = 0
    for (zm,), rows in segments.filter(pl.col("zm").is_in(pl.Series(list(mas), dtype=pl.UInt32))).sort("qs").group_by("zm"):
        if zm not in passing:
            continue
        spans = list(zip(rows["qs"], rows["qe"]))
        hits = mas[zm]
        gaps_ok = all(any(a < q2 + tol and b > e1 - tol for a, b in hits) for (_, e1), (q2, _) in zip(spans, spans[1:]))
        inside = any(q + tol < a and b < e - tol for q, e in spans for a, b in hits)
        compared += 1
        agreed += gaps_ok and not inside
    return {"compared_reads": compared, "agreeing_reads": agreed, "agreement": agreed / compared if compared else None}


def sreads(rows) -> pl.DataFrame:
    """(zm, qs, qe), di numbered in order within each ZMW; dl and dr are not used by the agreement."""
    di = {}
    out = []
    for zm, qs, qe in rows:
        out.append((zm, di.setdefault(zm, 0), 0, 1, qs, qe))
        di[zm] += 1
    return pl.DataFrame(out, schema={"zm": pl.UInt32, "di": pl.Int16, "dl": pl.Int8, "dr": pl.Int8, "qs": pl.Int32,
                                     "qe": pl.Int32}, orient="row")


def mas_hits(rows, source="hifi") -> pl.DataFrame:
    """(zm, name, start, end) as read_features rows of kind MAS on the + strand."""
    return pl.DataFrame([(zm, source, name, "MAS", "+", a, b, 0) for zm, name, a, b in rows],
                        schema={"zm": pl.UInt32, "source": pl.Utf8, "name": pl.Utf8, "kind": pl.Utf8, "strand": pl.Utf8,
                                "start": pl.Int32, "end": pl.Int32, "edits": pl.Int8}, orient="row")


def zms(*values) -> pl.Series:
    return pl.Series(values, dtype=pl.UInt32)


def test_mutate_rate():
    rng = random.Random(0)
    s = "ACGT" * 5000
    assert gates.mutate(s, 0.0, rng) == s
    diff = sum(a != b for a, b in zip(gates.mutate(s, 0.3, rng), s))
    assert diff > 2000


def test_skera_agreement():
    seg = sreads([(1, 17, 1000), (1, 1017, 2000), (2, 17, 1000), (2, 1017, 2000)])
    feats = mas_hits([(1, "A", 0, 17), (1, "B", 1000, 1017), (1, "C", 2000, 2017),
                      (2, "A", 0, 17), (2, "B", 500, 517), (2, "C", 2000, 2017)])
    m = gates.skera_agreement(seg, feats, zms(1, 2))
    assert (m["compared_reads"], m["agreeing_reads"]) == (2, 1)        # read 2: a gap without our adapter, an adapter inside
    assert m == per_read_agreement(seg, feats, {1, 2})


def test_skera_agreement_inside_rule_is_strict():
    """A hit starting exactly tol bp inside an S-read is not inside under the strict rule (it is under the inclusive one)."""
    seg = sreads([(1, 17, 1000), (1, 1017, 2000), (2, 17, 1000), (2, 1017, 2000)])
    feats = mas_hits([(1, "A", 0, 17), (1, "B", 1000, 1017), (1, "C", 2000, 2017), (1, "D", 22, 39),
                      (2, "A", 0, 17), (2, "B", 1000, 1017), (2, "C", 2000, 2017), (2, "D", 23, 40)])
    m = gates.skera_agreement(seg, feats, zms(1, 2))
    assert (m["compared_reads"], m["agreeing_reads"]) == (2, 1)        # only read 2's hit lies strictly inside
    assert m == per_read_agreement(seg, feats, {1, 2})


def test_skera_agreement_compares_passing_reads_with_hifi_mas_hits_only():
    f, _, s = frames(complete_with_false_cut())                        # read 1: a false cut, D inside the first S-read
    feats = pl.concat([f.with_columns(pl.lit("hifi").alias("source")).select(mas_hits([]).columns),
                       mas_hits([(3, "A", 0, 17), (3, "B", 1000, 1017)]),                       # read 3: not passing
                       mas_hits([(5, "A", 0, 17), (5, "B", 1000, 1017)], source="fail_sub")])   # read 5: fail hits only
    seg = pl.concat([s.select(sreads([]).columns), sreads([(3, 17, 1000), (4, 17, 1000), (5, 17, 1000), (6, 17, 1000)])])
    feats = pl.concat([feats, mas_hits([(6, "A", 0, 17), (6, "B", 1000, 1017)])])               # read 6: agrees
    m = gates.skera_agreement(seg, feats, zms(1, 4, 5, 6))                                      # read 4: no MAS hit
    assert m == {"compared_reads": 2, "agreeing_reads": 1, "agreement": 0.5}
    assert m == per_read_agreement(seg, feats, {1, 4, 5, 6})
    assert gates.skera_agreement(seg, feats, zms(4))["agreement"] is None


def test_skera_agreement_matches_per_read_rule_on_random_reads():
    """Random S-read layouts with hits near the cuts, at the tol edges and inside: the vectorised rule and the
    per-read loop give the same counts."""
    rng = random.Random(7)
    seg_rows, hit_rows, passing = [], [], set()
    for zm in range(1, 1500):
        pos = rng.randint(0, 30)
        cuts = []
        for _ in range(rng.randint(1, 5)):
            qs = pos + rng.randint(0, 25)
            qe = qs + rng.randint(30, 400)
            seg_rows.append((zm, qs, qe))
            cuts.append((qs, qe))
            pos = qe
        for qs, qe in cuts:
            for edge in (qs, qe):
                if rng.random() < 0.8:
                    a = edge + rng.randint(-12, 12)
                    hit_rows.append((zm, "A", a, a + rng.randint(10, 17)))
            if rng.random() < 0.2:
                a = rng.randint(qs, qe)
                hit_rows.append((zm, "B", a, a + 17))
        if rng.random() < 0.9:
            passing.add(zm)
    seg = sreads(seg_rows)
    feats = mas_hits(hit_rows)
    m = gates.skera_agreement(seg, feats, pl.Series(sorted(passing), dtype=pl.UInt32))
    want = per_read_agreement(seg, feats, passing)
    assert m == want
    assert 0.1 < want["agreement"] < 0.9 and want["compared_reads"] > 1000


def test_adapter_index_agreement():
    seg = pl.DataFrame({"zm": [1, 1], "dl": [0, 2], "dr": [1, 1], "qs": [17, 1017], "qe": [1000, 2000]}, schema_overrides=U32)
    feats = pl.DataFrame({"zm": [1, 1, 1], "source": ["hifi"] * 3, "kind": ["MAS"] * 3, "name": ["A", "B", "C"], "strand": ["+"] * 3,
                          "start": [0, 1000, 2000], "end": [17, 1017, 2017]}, schema_overrides=U32)
    m = gates.adapter_index_agreement(seg, feats)
    assert (m["checked_boundaries"], m["agreeing"]) == (4, 2)          # the second S-read has dl/dr swapped


def test_lima_agreement():
    fate = pl.DataFrame({"end_class": ["proper", "proper", "tso_tso", "tso_only"], "lima_pair": ["5p--3p", "none", "5p--5p", "none"],
                         "mapped_fltnc": [True, False, False, False], "lima_pass": [True, False, False, False]})
    m = gates.lima_agreement(fate)
    assert m["class_pair_agreement"] == 0.75 and m["fltnc_in_lima_pass"] == 1.0


def test_lima_confusion_order_with_ties():
    """Rows by count, largest first; rows tied on count by lima pair, then end class."""
    rows = ([("proper", "5p--3p")] * 3 + [("proper", "5p--5p")] * 2 + [("r1_r1", "3p--3p")] * 2 + [("tso_tso", "5p--5p")] * 2
            + [("other", "none")])
    for seed in range(5):
        fate = (pl.DataFrame(rows, schema=["end_class", "lima_pair"], orient="row")
                .with_columns(pl.lit(False).alias("mapped_fltnc"), pl.lit(False).alias("lima_pass")).sample(fraction=1.0, shuffle=True, seed=seed))
        got = [(r["end_class"], r["lima_pair"], r["len"]) for r in gates.lima_agreement(fate)["confusion"]]
        assert got == [("proper", "5p--3p", 3), ("r1_r1", "3p--3p", 2), ("proper", "5p--5p", 2), ("tso_tso", "5p--5p", 2),
                       ("other", "none", 1)]


def test_write_gate(tmp_path):
    p = gates.write_gate(tmp_path, "x", True, {"a": 1}, {"a": ">= 1"})
    assert p == tmp_path / "gates" / "gate_x.json"
    assert json.loads(p.read_text()) == {"gate": "x", "passed": True, "metrics": {"a": 1}, "threshold": {"a": ">= 1"}}
    assert p.read_text().startswith('{\n "gate": "x",\n "passed": true')            # indent 1, as v10's gate files


def molecule(n, rng):
    return R1 + "".join(rng.choice("ACGT") for _ in range(43)) + "T" * 20 + "".join(rng.choice("ACGT") for _ in range(n)) + revcomp(TSO_CORE + TSO_EXT)


def test_synthetic_benchmark_on_clean_molecules():
    rng = random.Random(3)
    pool = [molecule(rng.randint(400, 2500), rng) for _ in range(60)]
    r = gates.synthetic_benchmark(pool, n_arrays=20, n_fused=20)
    assert r["n_clean_sreads"] >= 55          # random cDNA can carry a primer-like match; such molecules are left out
    assert gates.synthetic_passed(r)
    assert r["rates"]["0.12"]["MAS"]["recall"] < r["rates"]["0.001"]["MAS"]["recall"]


def test_synthetic_benchmark_scores_fail_rates_with_fail_thresholds():
    rng = random.Random(4)
    pool = [molecule(rng.randint(400, 1500), rng) for _ in range(30)]
    strict = gates.synthetic_benchmark(pool, n_arrays=15, n_fused=5, fail_max_edits=HIFI_MAX_EDITS)
    relaxed = gates.synthetic_benchmark(pool, n_arrays=15, n_fused=5)
    assert (strict["fail_max_edits"], relaxed["fail_max_edits"]) == (HIFI_MAX_EDITS, FAIL_MAX_EDITS)
    for r in gates.HIFI_RATES:                      # the same arrays, scored with the HiFi thresholds in both runs
        assert relaxed["rates"][str(r)] == strict["rates"][str(r)]
    assert relaxed["rates"]["0.12"]["TSO"]["recall"] > strict["rates"]["0.12"]["TSO"]["recall"]


def toy_tree(root: Path, modulo: int) -> tuple[Path, int, int]:
    """A sample S with the parquet files the skera, lima and adapter_index gates read: ZMW kept 1 and 2 (the
    ported known-answer reads; read 2 disagrees with skera and has its dl/dr swapped), and ZMW dropped, which the
    1-in-modulo subsample leaves out (it would disagree too). Returns (samples table, kept 1, dropped)."""
    ids = list(range(1, 200))
    keep = keep_zmw_array(ids, modulo)
    kept = [z for z, k in zip(ids, keep) if k][:2]
    dropped = next(z for z, k in zip(ids, keep) if not k)
    k1, k2 = kept
    out = root / "S"
    out.mkdir(parents=True)
    pl.DataFrame({"zm": [k1, k2, dropped]}, schema=U32).write_parquet(out / "hifi_index.parquet")
    rows = []
    for zm in (k1, k2, dropped):
        b = 1000 if zm == k1 else 500
        rows += [(zm, "A", 0, 17), (zm, "B", b, b + 17), (zm, "C", 2000, 2017)]
    mas_hits(rows).write_parquet(out / "read_features.parquet")
    pl.DataFrame({"zm": [k1, k2, dropped], "passing": [True, True, True]}, schema_overrides=U32).write_parquet(out / "reads.parquet")
    pl.DataFrame({"zm": [k1, k1, k2, k2, dropped, dropped], "di": [0, 1, 0, 1, 0, 1], "dl": [0, 1, 1, 2, 0, 1],
                  "dr": [1, 2, 0, 1, 1, 2], "qs": [17, 1017] * 3, "qe": [1000, 2000] * 3},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "dl": pl.Int8, "dr": pl.Int8, "qs": pl.Int32, "qe": pl.Int32}) \
        .write_parquet(out / "segments.parquet")
    pl.DataFrame({"end_class": ["proper"] * 4, "lima_pair": ["5p--3p"] * 4, "mapped_fltnc": [True] * 4,
                  "lima_pass": [True] * 4}).write_parquet(out / "fate.parquet")
    samples = root / "samples.tsv"
    samples.write_text(HEADER + "S\tcell\tnone" + "\tx" * 7 + "\n")
    return samples, k1, dropped


def run_gates(samples: Path, out_root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PIPELINE / "gates.py"), "--sample", "S", "--samples", str(samples),
                           "--out-root", str(out_root), "--fraction", "1", *extra], cwd=REPO, capture_output=True, text=True)


def test_cli_skera_lima_adapter_index_on_toy_tree(tmp_path):
    samples, k1, dropped = toy_tree(tmp_path, 3)
    gate_dir = tmp_path / "S" / "gates"
    r = run_gates(samples, tmp_path, "--gate", "skera", "--fail-subsample", "3")
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in gate_dir.iterdir()) == ["gate_skera.json"]
    g = json.loads((gate_dir / "gate_skera.json").read_text())
    assert g == {"gate": "skera", "passed": True, "metrics": {"compared_reads": 2, "agreeing_reads": 1, "agreement": 0.5},
                 "threshold": {"agreement": "reported only"}}
    r = run_gates(samples, tmp_path, "--gate", "lima", "--fail-subsample", "3")
    assert r.returncode == 0, r.stderr
    assert json.loads((gate_dir / "gate_lima.json").read_text())["passed"] is True
    r = run_gates(samples, tmp_path, "--gate", "adapter_index", "--fail-subsample", "3")
    assert r.returncode == 1                                            # a hard gate: 4 of 6 boundaries agree
    g = json.loads((gate_dir / "gate_adapter_index.json").read_text())
    assert (g["passed"], g["metrics"]["checked_boundaries"], g["metrics"]["agreeing"]) == (False, 6, 4)
    assert g["threshold"] == {"agreement": ">= 0.99"}
    r = run_gates(samples, tmp_path, "--gate", "skera", "--fail-subsample", "1")      # 1 in 1: every ZMW
    assert json.loads((gate_dir / "gate_skera.json").read_text())["metrics"]["compared_reads"] == 3


def test_cli_help():
    r = subprocess.run([sys.executable, str(PIPELINE / "gates.py"), "--help"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "--gate" in r.stdout and "--fail-subsample" in r.stdout


@pytest.mark.real
def test_synthetic_gate_on_mini_sreads(mini):
    sreads_ = [r.query_sequence for r in bam_records(mini_samples(mini)["BD144_kin"].segmented_bam)[:1500]]
    assert gates.synthetic_passed(gates.synthetic_benchmark(sreads_, n_arrays=60, n_fused=60))


def agreement_inputs(out: Path, modulo: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The gate's inputs as the old gate read them: segments and MAS features of the 1-in-modulo HiFi ZMW subsample,
    and reads.parquet's passing column."""
    z = pl.read_parquet(out / "hifi_index.parquet", columns=["zm"])["zm"]
    sub = z.filter(pl.Series(keep_zmw_array(z.to_numpy(), modulo)))
    feats = pl.scan_parquet(out / "read_features.parquet").filter(pl.col("kind") == "MAS", pl.col("zm").is_in(sub)).collect()
    seg = pl.read_parquet(out / "segments.parquet", columns=["zm", "qs", "qe"]).filter(pl.col("zm").is_in(sub))
    return seg, feats, pl.read_parquet(out / "reads.parquet", columns=["zm", "passing"])


@pytest.mark.real
@pytest.mark.parametrize("sample", SAMPLES)
def test_skera_gate_on_mini_equals_per_read_rule(mini_study, sample):
    """The mini_study fixture runs the gate stage (--gate skera, 1 in 10); its agreement equals the per-read rule's
    on the same inputs, and the file has v10's keys."""
    g = json.loads((mini_study / sample / "gates" / "gate_skera.json").read_text())
    seg, feats, reads = agreement_inputs(mini_study / sample, 10)
    want = per_read_agreement(seg, feats, set(reads.filter(pl.col("passing"))["zm"]))
    assert want["compared_reads"] > 100
    assert g["metrics"] == want
    assert key_structure(g) == key_structure(json.loads((GOLDEN / sample / "gates" / "gate_skera.json").read_text()))


@pytest.mark.real
@pytest.mark.parametrize("sample", SAMPLES)
def test_lima_and_adapter_index_gates_pass_on_mini(mini_study, mini, sample):
    """The two hard gates that read stage 03 and 05 outputs pass on the mini, and their files have v10's keys."""
    for gate in ("lima", "adapter_index"):
        run_stage("gates.py", sample, mini_study, mini / "samples_mini.tsv", "--fraction", "1", "--gate", gate,
                  "--fail-subsample", "10")
        g = json.loads((mini_study / sample / "gates" / f"gate_{gate}.json").read_text())
        assert g["passed"] is True
        assert key_structure(g) == key_structure(json.loads((GOLDEN / sample / "gates" / f"gate_{gate}.json").read_text()))


@pytest.mark.real
@pytest.mark.slow
@pytest.mark.parametrize("sample", SAMPLES)
def test_gates_on_v10_hash_sample_equal_golden(local, tmp_path, sample):
    """All four gates on the v10 hash sample (0.18, 1 in 10) into a scratch tree: every gate file equals the frozen
    v10 one (compare_runs), and the skera agreement equals the per-read rule's on the same inputs."""
    src = local["hash_dir"] / sample
    out = tmp_path / "hash_0.18" / sample
    out.mkdir(parents=True)
    for name in ("hifi_index", "read_features", "reads", "segments", "fate"):
        (out / f"{name}.parquet").symlink_to(src / f"{name}.parquet")
    inputs = local["inputs_dir"] / sample
    samples = tmp_path / "samples.tsv"
    samples.write_text(HEADER + f"{sample}\tcell\tnone" + f"\t{inputs}/x" * 2 + f"\t{inputs}/segmented.bam" + f"\t{inputs}/x" * 4 + "\n")
    r = subprocess.run([sys.executable, str(PIPELINE / "gates.py"), "--sample", sample, "--samples", str(samples),
                        "--out-root", str(tmp_path), "--fraction", "0.18", "--fail-subsample", "10", "--gate", "all"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (out / "gates").iterdir()) == sorted(p.name for p in (GOLDEN / sample / "gates").iterdir())
    golden = tmp_path / "golden_gates"
    shutil.copytree(GOLDEN / sample / "gates", golden)
    skera = golden / "gate_skera.json"
    assert skera.read_text().count(V10_SKERA_TEXT) == 1                  # the one known difference, pinned
    skera.write_text(skera.read_text().replace(V10_SKERA_TEXT, "reported only"))
    assert compare_trees(out / "gates", golden) == []                    # every other key of all 12 files, exactly
    seg, feats, reads = agreement_inputs(out, 10)
    want = per_read_agreement(seg, feats, set(reads.filter(pl.col("passing"))["zm"]))
    assert json.loads((out / "gates" / "gate_skera.json").read_text())["metrics"] == want
