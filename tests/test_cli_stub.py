import subprocess


def test_version():
    out = subprocess.run(["kinnex-qc", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.startswith("kinnex-qc ")
