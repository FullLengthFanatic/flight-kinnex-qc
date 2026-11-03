import pytest

from kinnex_qc import patterns
from kinnex_qc.annotate import annotate_read, chain_flags
from kinnex_qc.patterns import (KINNEX8_SC3, MAS, MAS_ORDER, R1, TSO_CORE, Chemistry, all_patterns, chemistry_from,
                                read_adapter_fasta)


def test_default_patterns_unchanged():
    assert [(p.name, p.kind, p.seq, p.max_edits) for p in all_patterns()] == \
        [(x, "MAS", MAS[x], 2) for x in MAS_ORDER] + [("TSO", "TSO", TSO_CORE, 3), ("R1", "R1", R1, 3)]


def test_read_adapter_fasta_keeps_order(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text(">X\nACGTACGTACGTACGTA\n>Y\nTTTTGGGGCCCCAAAAT\n")
    adapters, order = read_adapter_fasta(f)
    assert order == ("X", "Y") and adapters["Y"] == "TTTTGGGGCCCCAAAAT"


def test_read_adapter_fasta_rejects_empty_header(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text(">\nACGTACGTACGTACGTA\n")
    with pytest.raises(ValueError, match="empty header"):
        read_adapter_fasta(f)


def test_read_adapter_fasta_rejects_sequence_before_header(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text("ACGTACGTACGTACGTA\n>X\nTTTTGGGGCCCCAAAAT\n")
    with pytest.raises(ValueError, match="before any"):
        read_adapter_fasta(f)


def test_read_adapter_fasta_rejects_duplicate_name(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text(">X\nACGTACGTACGTACGTA\n>X\nTTTTGGGGCCCCAAAAT\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_adapter_fasta(f)


@pytest.mark.parametrize(("text", "error"), [(">X\nACGTACGTACGTACGTA\n", None), (">X\nACGTACGTACGTACGTA\n>X\nACGTACGTACGTACGTA\n", "duplicate")])
def test_read_adapter_fasta_closes_its_file(tmp_path, monkeypatch, text, error):
    """The FASTA handle is closed when read_adapter_fasta returns and when it raises, not left to the garbage collector
    (a ResourceWarning under -W error::ResourceWarning)."""
    f = tmp_path / "a.fasta"
    f.write_text(text)
    opened = []

    def tracking_open(*args, **kwargs):
        opened.append(open(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(patterns, "open", tracking_open, raising=False)
    if error:
        with pytest.raises(ValueError, match=error):
            read_adapter_fasta(f)
    else:
        read_adapter_fasta(f)
    assert len(opened) == 1
    assert opened[0].closed


def test_chemistry_from_defaults():
    assert chemistry_from(None, "sc3") == KINNEX8_SC3


def test_chemistry_from_explicit_primers():
    c = chemistry_from(None, "AAAACCCCGGGGTTTTAAAACCCCG,TTTTGGGGCCCCAAAATTTT")
    assert c.tso == "AAAACCCCGGGGTTTTAAAACCCCG" and c.r1 == "TTTTGGGGCCCCAAAATTTT"


def test_chemistry_from_rejects_bad_primers():
    with pytest.raises(ValueError):
        chemistry_from(None, "ACGT")


def test_chain_flags_custom_order():
    assert chain_flags(["X", "Y"], order=("X", "Y")) == {"complete"}
    assert chain_flags(["X", "X'"], order=("X", "Y")) >= {"mirror"}


def test_annotate_read_custom_chemistry_finds_custom_adapter():
    chem = Chemistry(adapters={"X": "GATTACAGATTACAGAT", "Y": "CCGGTTAACCGGTTAAC"}, order=("X", "Y"), tso=TSO_CORE, r1=R1)
    seq = "GATTACAGATTACAGAT" + R1 + "T" * 300 + TSO_CORE + "CCGGTTAACCGGTTAAC"
    names = [h.name for h in annotate_read(seq, all_patterns(chem=chem))]
    assert names == ["X", "R1", "TSO", "Y"]
