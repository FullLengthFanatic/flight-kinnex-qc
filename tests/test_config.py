import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study" / "pipeline"))

from config import Sample, load_config, load_samples, out_dir  # noqa: E402

SAMPLES_HEADER = "sample\tcell\tdepletion\thifi_bam\tfail_bam\tsegmented_bam\tnon_passing_bam\tzmw_metrics\tskera_report\tfltnc_bam\n"
SAMPLES_ROW = "BD144_ND\tPB538_3_r0912\tnone\thifi.bam\tfail.bam\tsegmented.bam\tnon_passing.bam\tzmw.csv.gz\treport.json\tfltnc.bam\n"


def test_load_samples_parses_rows_into_sample(tmp_path):
    samples_path = tmp_path / "samples.tsv"
    samples_path.write_text(SAMPLES_HEADER + SAMPLES_ROW)

    samples = load_samples(samples_path)

    assert list(samples) == ["BD144_ND"]
    sample = samples["BD144_ND"]
    assert isinstance(sample, Sample)
    assert sample.sample == "BD144_ND"
    assert sample.cell == "PB538_3_r0912"
    assert sample.depletion == "none"
    assert sample.hifi_bam == Path("hifi.bam")
    assert sample.fail_bam == Path("fail.bam")
    assert sample.segmented_bam == Path("segmented.bam")
    assert sample.non_passing_bam == Path("non_passing.bam")
    assert sample.zmw_metrics == Path("zmw.csv.gz")
    assert sample.skera_report == Path("report.json")
    assert sample.fltnc_bam == Path("fltnc.bam")


def test_load_samples_missing_column_raises_clear_error(tmp_path):
    header = "sample\tcell\tdepletion\thifi_bam\tfail_bam\tsegmented_bam\tnon_passing_bam\tzmw_metrics\tskera_report\n"
    row = "BD144_ND\tPB538_3_r0912\tnone\thifi.bam\tfail.bam\tsegmented.bam\tnon_passing.bam\tzmw.csv.gz\treport.json\n"
    samples_path = tmp_path / "samples.tsv"
    samples_path.write_text(header + row)

    with pytest.raises(ValueError, match="fltnc_bam"):
        load_samples(samples_path)


def test_load_config_overlays_local_file(tmp_path):
    (tmp_path / "study.toml").write_text('fraction = 0.18\npbtools_env = "kinnex-qc-pbtools"\nout_root = "<out>"\n')
    (tmp_path / "study.local.toml").write_text('pbtools_env = "other-pbtools"\nout_root = "overridden/out_root"\n')

    config = load_config(tmp_path / "study.toml")

    assert config["fraction"] == 0.18
    assert config["pbtools_env"] == "other-pbtools"
    assert config["out_root"] == "overridden/out_root"


def test_load_config_without_local_file_keeps_base_values(tmp_path):
    (tmp_path / "study.toml").write_text('fraction = 0.18\npbtools_env = "kinnex-qc-pbtools"\n')

    config = load_config(tmp_path / "study.toml")

    assert config["fraction"] == 0.18
    assert config["pbtools_env"] == "kinnex-qc-pbtools"


def test_out_dir_creates_directory(tmp_path):
    path = out_dir(tmp_path, "BD144_ND")

    assert path == tmp_path / "BD144_ND"
    assert path.is_dir()


def test_out_dir_puts_a_hash_sample_under_hash_fraction(tmp_path):
    assert out_dir(tmp_path, "BD144_ND", 1) == tmp_path / "BD144_ND"
    path = out_dir(tmp_path, "BD144_ND", 0.18)
    assert path == tmp_path / "hash_0.18" / "BD144_ND"
    assert path.is_dir()


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_out_dir_rejects_fraction_outside_0_1(tmp_path, fraction):
    with pytest.raises(ValueError, match="fraction"):
        out_dir(tmp_path, "BD144_ND", fraction)
    assert not (tmp_path / "BD144_ND").exists()
