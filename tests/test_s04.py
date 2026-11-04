"""Tests for study/pipeline/s04_lima_refine.sh on the mini dataset. The run at fraction 1, which reads the segmented
BAM as it is, is covered with stage 05 in test_s05.py."""
import os
import subprocess

import pytest

from kinnex_qc.sampling import in_hash_sample
from pipeline_helpers import bam_records, mini_samples, run_stage


@pytest.mark.real
def test_lima_refine_hash_sample(mini, tmp_path):
    """Below fraction 1 lima and refine run on a temporary BAM of the hash sample's S-reads, trashed at the end."""
    run_stage("s04_lima_refine.sh", "BD144_ND", tmp_path, mini / "samples_mini.tsv", "--fraction", "0.5")
    out = tmp_path / "hash_0.5" / "BD144_ND"
    assert (out / "lima" / "BD144_ND.isoseq.FlashBD_5p--FlashBD_3p.bam").exists()
    assert (out / "lima" / "BD144_ND.pairs.bam").exists()
    assert (out / "refine" / "BD144_ND.asrun.bam").exists()
    assert all((out / "refine" / f"BD144_ND.polyA_{n}.bam").exists() for n in (5, 8, 12, 20))
    assert not any((tmp_path / "tmp").glob("*/segmented.bam"))          # temporary BAM trashed
    sreads = [r for r in bam_records(mini_samples(mini)["BD144_ND"].segmented_bam) if in_hash_sample(r.get_tag("zm"), 0.5)]
    summary = (out / "lima" / "BD144_ND.pairs.lima.summary").read_text()
    assert f"Reads input                    : {len(sreads)}\n" in summary
    assert 0 < len(sreads) < len(bam_records(mini_samples(mini)["BD144_ND"].segmented_bam))


@pytest.mark.parametrize("env, flag", [("pbname", "-n pbname"), ("/envs/pb", "-p /envs/pb")])
def test_pbtools_env_names_the_conda_environment(tmp_path, env, flag):
    """--pbtools-env names the conda environment lima and isoseq run in: a name with -n, a path with -p. A stand-in
    conda records its calls; at fraction 1 the segmented BAM goes to lima as it is, so no data is read."""
    from pipeline_helpers import PIPELINE, REPO
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "conda").write_text('#!/bin/bash\necho "$*" >> "$CONDA_LOG"\n')
    (tmp_path / "bin" / "conda").chmod(0o755)
    cols = ("sample", "cell", "depletion", "hifi_bam", "fail_bam", "segmented_bam", "non_passing_bam", "zmw_metrics",
            "skera_report", "fltnc_bam")
    (tmp_path / "samples.tsv").write_text("\t".join(cols) + "\n" + "\t".join(["toy"] + [f"/x/{c}" for c in cols[1:]]) + "\n")
    log = tmp_path / "conda.log"
    r = subprocess.run(["bash", str(PIPELINE / "s04_lima_refine.sh"), "--sample", "toy", "--samples", str(tmp_path / "samples.tsv"),
                        "--out-root", str(tmp_path / "out"), "--fraction", "1", "--pbtools-env", env], cwd=REPO,
                       env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "CONDA_LOG": str(log)},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 7                                              # lima twice, isoseq refine five times
    assert all(c.startswith(f"run --no-capture-output {flag} ") for c in calls)


def test_help_exits_0_and_prints_the_usage_line():
    """--help is handled in bash before any Python, so it needs no data, and prints the leading comment block."""
    from pipeline_helpers import PIPELINE
    r = subprocess.run(["bash", str(PIPELINE / "s04_lima_refine.sh"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "study/pipeline/s04_lima_refine.sh --sample <name>" in r.stdout
    assert "Defaults come from study/study.toml" in r.stdout          # the last line of the comment block
    assert "set -euo" not in r.stdout
