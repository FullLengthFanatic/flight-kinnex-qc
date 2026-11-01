import os
import subprocess
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "study" / "prepare_inputs.sh"
COLS = ["sample", "data_tar", "analysis_tar", "hifi_member", "fail_member", "zmw_metrics_member",
        "segmented_member", "non_passing_member", "skera_report_member"]


def make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as t:
        for name, data in members.items():
            src = path.parent / "src" / name
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(data)
            t.add(src, arcname=name)


def run(table: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(table), str(out)], env=os.environ | {"MIN_FREE_GB": "0"},
                          capture_output=True, text=True)


def test_members_with_spaces_extract_and_resume(tmp_path):
    data = {"run 1/hifi reads.bam": b"h" * 10, "run 1/fail.bam": b"f" * 20, "run 1/zmw metrics.csv.gz": b"z" * 30}
    analysis = {"out dir/segmented.bam": b"s" * 40, "out dir/non passing.bam": b"n" * 50, "out dir/report.json": b"{}"}
    make_tar(tmp_path / "data.tar", data)
    make_tar(tmp_path / "analysis.tar", analysis)
    row = ["S1", str(tmp_path / "data.tar"), str(tmp_path / "analysis.tar"), *data, *analysis]
    table = tmp_path / "members.tsv"
    table.write_text("\t".join(COLS) + "\n" + "\t".join(row) + "\n")
    out = tmp_path / "out"
    r = run(table, out)
    assert r.returncode == 0, r.stderr
    sizes = {p.name: p.stat().st_size for p in (out / "S1").iterdir() if p.is_file() and p.suffix != ".log"}
    assert sizes == {"hifi_reads.bam": 10, "fail_reads.bam": 20, "zmw_metrics.csv.gz": 30, "segmented.bam": 40,
                     "segmented.non_passing.bam": 50, "read_segmentation.report.json": 2}
    again = run(table, out)
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("skip (already done") == 6
