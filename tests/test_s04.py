"""Tests for study/pipeline/s04_lima_refine.sh on the mini dataset. The run at fraction 1, which reads the segmented
BAM as it is, is covered with stage 05 in test_s05.py."""
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


def test_help_exits_0_and_prints_the_usage_line():
    """--help is handled in bash before any Python, so it needs no data, and prints the leading comment block."""
    from pipeline_helpers import PIPELINE
    r = subprocess.run(["bash", str(PIPELINE / "s04_lima_refine.sh"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "study/pipeline/s04_lima_refine.sh --sample <name>" in r.stdout
    assert "Defaults come from study/study.toml" in r.stdout          # the last line of the comment block
    assert "set -euo" not in r.stdout
