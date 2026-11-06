"""Tests for docs/build_docs.py and the documentation it builds. README.md and docs/*.md are generated from their .md.in
sources, with every number taken from numbers.json (the full rerun's, report/numbers.json, which tests/test_golden.py holds
to v10's apart from the approved number changes) through the Note's own formatter, and must be up to date. Also: the
README's order, length, figure and read excerpt; the sections and PacBio quotes of docs/skera_cutting_error.md; the
per-library false-cut table; each open item in one marked place; no number typed by hand in a source; the text rules on
every Markdown file; the CI and CITATION.cff details the docs rely on."""
import json
import re
import sys

import polars as pl
import pytest

from pipeline_helpers import GOLDEN, REPO
from test_render import banned_hits

import render

sys.path.insert(0, str(REPO / "docs"))
import build_docs  # noqa: E402

GENERATED = ["README.md", "docs/concepts.md", "docs/data.md", "docs/manuscript_methods.md", "docs/reader_guide.md",
             "docs/skera_cutting_error.md", "docs/supplementary_methods.md"]
README_HEADINGS = ["# flight-kinnex-qc", "## What we found", "## Check your own data", "## Reproduce the study", "## Documentation",
                   "## Citation", "## Licence"]
SKERA_SECTIONS = ["What skera does", "How we found it", "How often", "What it costs", "What PacBio documents",
                  "How to check your data and recover the molecules"]
CONCEPTS_SECTIONS = ["Kinnex 8-fold arrays", "What skera does", "How the study cuts reads", "How a TSO-TSO molecule makes a mirror array",
                     "What a false cut is", "Glossary"]
READER_SECTIONS = ["Install", "Run", "Reading the report", "Scope", "Validation"]
# PacBio's own words, as the pages read on 2026-11-05, each with the page it is quoted from.
QUOTES = [
    ("https://skera.how/", "Skera splits Kinnex HiFi reads at adapter positions generating segmented reads (S-reads)."),
    ("https://skera.how/adapters.html", "Adapters must be in FASTA format `<adapters>.fasta` and ordered in the expected order of adapters in "
     "the reads. There should be one entry per adapter (forward or reverse-complement orientation) with no overlapping adapter sequences. "
     "Duplicate names or sequences are not allowed."),
    ("https://skera.how/output.html", "Skera outputs a BAM file containing all segmented reads that have sequential leading and trailing "
     "adapters based on the order of adapters in the input fasta."),
    ("https://skera.how/output.html", "The non_passing.bam file contains the bam records that have adapters found in non-sequential order. "
     "This file is necessary for skera undo."),
    ("https://github.com/PacificBiosciences/pbbioconda/issues/608#issuecomment-1726413858",
     "For example, if `skera` finds 14 / 15 expected segments(fragments), the 14 expected segments will be output to the segmented read "
     "output _and_ the HiFi input read will be output to the `non_passing` bam."),
]
ARGENTAG_SENTENCE = "segmented.non_passing.bam is not public; ArgenTag shared it with us. ARGENTAG_WORDING_TBD"
SCOPE = "validated on BD Rhapsody (FLIGHT-seq) Kinnex 8-fold libraries only"
# Numbers a source may type by hand: definitions, thresholds, parameters, versions, stage numbers, and measurements that
# are not results of the study's run (the reader's validation runs, this machine's disk and run times). Every other
# number comes from numbers.json or the run's tables through a placeholder or a slot.
CONSTANTS = {
    # counts, positions and thresholds that define the chemistry, the rules and the methods
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16", "20", "25", "30", "50", "64", "80", "150", "200",
    "1,000", "0.99", "1.5", "80%", "20%", "12%", "99th", "3'", "5'",
    # the bound render.check_claims holds the Note's "under 1% of the mean partner" to
    "1%",
    # the pipeline's stage numbers
    "01", "02", "03", "04", "05", "06", "07",
    # versions of software and of this package
    "0.1.0", "1.1.0", "1.2.0", "1.4.0", "26.1.0", "26.2.0", "2.13.0", "4.3.0", "2.30", "1.13", "3.12", "1.1",
    # PacBio's pbbioconda issue 608 and the example of its reply (14 of 15 segments found)
    "608", "15",
    # the default fraction of kinnex-qc and the downstream test's design (3,000 reads per group)
    "0.1", "10%", "3,000",
    # dates: the pages were read in 2026, the pbbioconda reply is of 2023
    "2026", "2023",
    # the reader's validation runs (study/validate_reader.py, 2026-11-02): fraction, compared numbers, read counts, the one
    # differing value, run times and peak memory, all measured on those runs and not part of numbers.json
    "0.18", "229", "129", "33,628,976", "6,054,028", "0.0034570434472299045", "313,852", "19", "37", "57", "32",
    "3:45:04", "4:11:28", "4:11:08", "14,797,320", "17,228,784", "16,728,116", "14.1", "16.4", "16.0",
    # this machine: memory, disk of the archives, the inputs and the mapped BAMs, the free-space floor of
    # prepare_inputs.sh (CPUs, times and the rest are among the numbers above)
    "117", "438", "422", "135", "146", "44", "400",
    # the full rerun of 2026-11-05 (its log and du -sh of its out root): stage 06 in minutes on BD144_ND and BD144_HM, the
    # calibration set, analyses and ArgenTag in minutes, the out root, its libraries and the calibration set in GiB, and the
    # bound on the rest (BD144_kin's 64 minutes and the whole run's 14 hours are among the numbers above)
    "56", "69", "46", "151", "45", "53", "2.2", "0.5",
}
NUM_TOKEN = re.compile(r"(?<![\w./:#-])\d(?:[\d,:]*\d)?(?:\.\d+)*(?:%|th)?'?(?!\w)")


def markdown_files() -> list:
    """Every Markdown file of the repository and every .md.in source, outside hidden folders and the git-ignored output folders."""
    skip = {".venv", "out", "qc", "build", "dist"}
    return sorted(p for p in REPO.rglob("*") if p.is_file() and (p.suffix == ".md" or p.name.endswith(".md.in"))
                  and not any(part.startswith(".") or part in skip for part in p.relative_to(REPO).parts))


def section(text: str, heading: str) -> str:
    """The text under a "## heading" line, up to the next "## " heading."""
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end > 0 else None]


def cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


@pytest.fixture(scope="module")
def built() -> dict:
    return build_docs.build(REPO / "report" / "numbers.json", REPO / "report" / "tables")


def test_docs_are_built_from_the_report_of_the_full_rerun():
    assert build_docs.NUMBERS == REPO / "report" / "numbers.json" and build_docs.TABLES == REPO / "report" / "tables"
    assert build_docs.fill is render.fill and build_docs.FORMATS is render.FORMATS
    assert "report/numbers.json and report/tables/" in " ".join(build_docs.__doc__.split())


def test_generated_files_are_up_to_date(built):
    assert sorted(p.relative_to(REPO).as_posix() for p in built) == GENERATED
    stale = [p.relative_to(REPO).as_posix() for p, text in built.items() if not p.is_file() or p.read_text() != text]
    assert stale == [], "out of date: run docs/build_docs.py"


def test_every_generated_file_names_its_source_and_holds_no_placeholder(built):
    for path, text in built.items():
        source = path.relative_to(REPO).as_posix() + ".in"
        assert text.splitlines()[0] == f"<!-- Generated by docs/build_docs.py from {source}; edit that file, then run docs/build_docs.py. -->"
        assert (REPO / source).is_file()
        assert not re.search(r"\{[a-z0-9_.]+\|", text) and "<!--table" not in text and "<!--excerpt" not in text, path.name


def test_every_markdown_doc_has_a_source():
    docs = REPO / "docs"
    assert sorted(p.name for p in docs.glob("*.md")) == sorted(p.name.removesuffix(".in") for p in docs.glob("*.md.in"))
    assert (REPO / "README.md.in").is_file()


def test_a_claim_the_docs_repeat_is_checked_against_the_data(tmp_path):
    numbers = json.loads((GOLDEN / "v10_numbers.json").read_text()) | {"false_cut_reads_share_max": 0.096}
    (tmp_path / "numbers.json").write_text(json.dumps(numbers))
    with pytest.raises(SystemExit, match="in all three libraries alike"):
        build_docs.build(tmp_path / "numbers.json", GOLDEN / "v10_tables")


def test_readme_order_length_and_opening():
    lines = (REPO / "README.md").read_text().splitlines()
    assert len(lines) < 250
    assert [x for x in lines if re.match(r"#{1,2} ", x)] == README_HEADINGS
    opening = " ".join(lines[lines.index(README_HEADINGS[0]) + 2:lines.index(README_HEADINGS[1])]).strip()
    assert len(re.findall(r"[.!?](?:\s|$)", opening)) == 2, opening


def test_readme_findings_put_the_skera_cutting_error_first_with_its_figure_and_read():
    text = (REPO / "README.md").read_text()
    found = section(text, "What we found")
    heads = re.findall(r"^\*\*(\d)\. ([^*]+)\*\*", found, flags=re.M)
    assert [n for n, _ in heads] == ["1", "2", "3", "4"] and "skera" in heads[0][1]
    # skera writes about half the usable molecules of a mirror read (mirror_arms_clean_share_mirror); the heading says it as
    # the Note does ("leaves many molecules of mirror arrays unwritten"), not as "few"
    assert "skera leaves many of their molecules unwritten" in heads[1][1] and "few" not in heads[1][1]
    first = found[found.index("**1. "):found.index("**2. ")]
    assert f"](report/figures/{render.SCHEMATICS['fig1']}.svg)" in first and first.count("![") == 1
    assert f"```text\n{build_docs.excerpt(build_docs.FASTA)}\n```" in first


def test_readme_check_and_reproduce_sections():
    text = (REPO / "README.md").read_text()
    check = section(text, "Check your own data")
    for s in ("uv tool install .", "pip install .", "kinnex-qc run --hifi", "--write-molecules", "not for bulk Iso-Seq", SCOPE):
        assert s in check, s
    repro = section(text, "Reproduce the study")
    for s in ("study/prepare_inputs.sh", "study/run_study.sh", "envs/pbtools.yml", "envs/samtools113.yml", "envs/skera120.yml",
              "uv sync --extra dev", "uv run python -m pytest", "GiB", "hours"):
        assert s in repro, s


def test_excerpt_shows_the_real_and_the_false_d_with_their_positions():
    ex = build_docs.excerpt(build_docs.FASTA).splitlines()
    assert len(ex) == 10
    assert ex[0] == ">skera_false_cut BD144_ND zm=1576268 read positions 2723-5795 of 14028 (adapter C to adapter E)"
    fasta = (REPO / "examples" / "example_reads.fasta").read_text().split(">skera_false_cut")[1].split("\n>")[0].splitlines()[1:]
    shown = {}
    for line in ex[1:]:
        m = re.fullmatch(r" *(\d+) ([ACGTacgt]+)", line)
        if m:
            shown[int(m.group(1))] = m.group(2)
            assert fasta[(int(m.group(1)) - 1) // 100] == m.group(2)
    assert sorted(shown) == [1, 1001, 1101, 2201]
    real_d = (shown[1001] + shown[1101])[1097 - 1000:1114 - 1000]
    false_d = shown[2201][2252 - 2200:2269 - 2200]
    assert (real_d, false_d) == ("ACCTCCTCCTCCAGAAT", "GCATACTCCTCCAGAAT")
    carets = {line.strip().split()[1]: line for line in ex if line.strip().startswith("^")}
    assert set(carets) == {"C", "D(real)", "skera_D"}
    assert "D(real) 1098-1114" in carets["D(real)"] and "skera_D 2253-2269" in carets["skera_D"] and "C 1-17" in carets["C"]
    width = len(ex[1]) - len(shown[1])
    assert carets["skera_D"].index("^") == width + 2252 - 2200 and carets["skera_D"].count("^") == 17
    assert carets["D(real)"].index("^") == width and carets["D(real)"].count("^") == 14


def test_skera_cutting_error_sections_and_pacbio_quotes():
    text = (REPO / "docs" / "skera_cutting_error.md").read_text()
    assert [x[3:] for x in text.splitlines() if x.startswith("## ")] == SKERA_SECTIONS
    documented = section(text, "What PacBio documents")
    quoted = " ".join(x[2:] for x in documented.splitlines() if x.startswith("> "))
    for url, quote in QUOTES:
        assert quote in quoted, quote
        assert url in documented, url
    assert not re.search(r"report(ed|ing)? (it |this |the error )?to PacBio", text, flags=re.I)


def test_false_cut_table_per_library_agrees_with_numbers_json():
    numbers = json.loads((GOLDEN / "v10_numbers.json").read_text())
    t = pl.read_csv(GOLDEN / "v10_tables" / "fig1_skera_false_cut_anatomy.tsv", separator="\t")
    assert (t["reads_with_false_cut"].min(), t["reads_with_false_cut"].max()) == (numbers["false_cut_reads_share_min"],
                                                                                 numbers["false_cut_reads_share_max"])
    table = build_docs.false_cut_table(GOLDEN / "v10_tables").splitlines()
    assert cells(table[0]) == ["", "ND", "HM", "kin"]
    rows = {cells(r)[0]: cells(r)[1:] for r in table[2:]}
    share = [render.pct(t.filter(pl.col("library") == lib)["reads_with_false_cut"].item()) for lib in ("ND", "HM", "kin")]
    assert rows["Complete arrays skera passes with at least one false cut"] == share
    assert rows["False cuts in them"] == ["142,451", "178,916", "175,319"]
    assert "\n".join(table) in (REPO / "docs" / "skera_cutting_error.md").read_text()


def test_passages_two_readers_could_not_follow_are_rewritten():
    """What kinnex-qc checks, the data, the chimera's real adapter and the chance baseline (README); why the PCR products line
    up, why a real adapter sits next to a TSO or Read1, how a TSO-TSO molecule grows a mirror, the mirror size in words with an
    example, the arms' orientation and the false-cut gap (concepts); the false cut's letter and the other strand (the skera
    cutting error). The libraries, the chimera and the mirror example were rewritten again in round 2 (next test)."""
    readme, concepts, skera = (" ".join((REPO / p).read_text().split()) for p in ("README.md", "docs/concepts.md", "docs/skera_cutting_error.md"))
    rewritten = [
        (readme, "a command that checks the array structure and artifact molecules (TSO-TSO, R1-R1, missing primers) in your own "
                 "Kinnex HiFi reads and, with skera's S-reads, skera's cuts."),
        (readme, "one BD Rhapsody cartridge (a bead-based single-cell capture kit)"),
        (readme, "the public ArgenTag 8-fold dataset from the MDA-MB-453 cell line."),
        (readme, "bp shorter than if their lengths were randomly shuffled across arrays, under 1% of the mean partner."),
        (concepts, "Since adjacent slots share an adapter, the eight slots require nine unique adapters: A, B, C, D, E, F, G, H and Q."),
        (concepts, "the B end of the first product joins the B end of the second"),
        (concepts, "or between a molecule and the end of the array for A and Q, so it lies next to the TSO or Read1 end of a molecule"),
        (concepts, "Each X end then joins the X end of an ordinary molecule from the next slot"),
        (concepts, "of the usable molecules on the arm after the junction (forward adapters, X to Q) as clean S-reads"),
        (concepts, "of those on the arm before it (reverse-complement adapters, Q' to X')."),
        (concepts, "(the sequence between where one S-read ends and the next begins) contains none of our adapters within 5 bp."),
        (skera, "when it skips the real D, for example, it labels the false cut D, and the real D gets no label."),
        (skera, "on reads from the other strand (adapters Q to A, so the read covers the array's reverse complement)"),
    ]
    assert [s for text, s in rewritten if s not in text] == []
    replaced = [(readme, "runs the array and skera checks"), (readme, "the skipped adapter and the start of the next molecule"),
                (readme, "shorter than by chance"), (concepts, "An adapter joins only its own letter"),
                (concepts, "(where every real adapter sits)"), (concepts, "primed by the TSO-side primer"), (concepts, "2 x (slots from X to Q)")]
    assert [s for text, s in replaced if s in text] == []


def test_passages_two_readers_could_not_follow_in_round_2_are_rewritten():
    """Round 2: the libraries named where ND is first used, the chimera's real adapter skera did not cut at, the false cut in the
    read excerpt and the mirror junction defined (README); how the cDNA goes into eight PCRs, what qs, qe, dl and dr hold, why a
    read with S-reads can be non-passing, the TSO-TSO paragraph split with the prime defined before its first use, the slots
    of the example mirror, the chimera, and the study's own segmentation under its own heading (concepts); the chimera, the
    false cut in the excerpt, why a real adapter sits next to a TSO or Read1, the CTCCTCC repeat and how close the look-alikes
    are to D (the skera cutting error)."""
    raw = {p: (REPO / p).read_text() for p in ("README.md", "docs/concepts.md", "docs/skera_cutting_error.md")}
    readme, concepts, skera = (" ".join(raw[p].split()) for p in raw)
    chimera = "a whole molecule, then the real adapter skera did not cut at, then the"
    excerpt = "Uppercase marks the adapters and the false cut, the site skera called D;"
    rewritten = [
        (readme, "split into three libraries, each made into Kinnex 8-fold arrays and sequenced on one Revio SMRT Cell: ND, with no "
                 "TSO-TSO depletion; HM, depleted by a home-made biotin-Read1 pulldown; and kin, depleted by the Kinnex TSO artifact removal."),
        (readme, f"{chimera} start of the next molecule."), (readme, excerpt),
        (readme, "of their mirror junctions (an adapter followed by its own reverse complement) hold a TSO-TSO molecule."),
        (concepts, "The cDNA is divided among eight PCRs, each with its own primer pair. The two primers of each pair carry the two "
                   "adapters of one slot"),
        (concepts, "Each S-read records where it starts and ends in the HiFi read (tags `qs` and `qe`) and which adapters lie before and "
                   "after it (tags `dl` and `dr`). `dl` and `dr` give each adapter as its position in that FASTA, counted from 0, so A, "
                   "the first entry, is 0 (documented: https://skera.how/read-segments.html)."),
        (concepts, "One read can be in both files. `segmented.bam` holds S-reads, the pieces of a read, and `non_passing.bam` whole HiFi "
                   "reads; skera can write S-reads from part of a read and still put the whole read into `non_passing.bam`."),
        (concepts, "a read where skera finds 14 of 15 expected segments gets those 14 written as S-reads and the whole HiFi read written "
                   "to `non_passing.bam`"),
        (concepts, "Having S-reads therefore does not make a read passing: we call a read passing when it is not in `non_passing.bam`."),
        (concepts, "whose other end carries that slot's next adapter. That adapter joins a molecule of the slot after, and so on up to Q, "
                   "on both sides. A prime marks an adapter read in reverse complement, that is, on the opposite strand. The array reads:"),
        (concepts, "With the junction at G, two slots lie between the junction and Q: G to H and H to Q. The array then holds two "
                   "molecules on each arm plus the TSO-TSO molecule, five in all."),
        (concepts, f"({chimera} start of the next molecule)"),
        (skera, f"{chimera} first part of the cut molecule."), (skera, excerpt),
        (skera, "On the calibration set (below), the sequence at"),
        (skera, "of false cuts is closest to D, 3 to 5 edits from it."),
        (skera, "keeps an adapter hit only within 4 bp of a TSO or Read1 hit, because every real adapter sits between two molecules "
                "(or at the array ends) and so always lies next to the TSO or Read1 end of a molecule (the full-read segmentation"),
        (skera, "A CTCCTCC repeat sequence appears within 10 bp of"),
        (skera, "of the false-cut sites, so the look-alike sequences are not one fixed motif."),
    ]
    assert [s for text, s in rewritten if s not in text] == []
    replaced = [(readme, "with no TSO-TSO depletion (ND)"), (readme, "the real adapter skera skipped"),
                (readme, "Uppercase marks the adapters and the site skera called D"),
                (concepts, "split into eight PCRs"), (concepts, "the index of the adapter before and after it"),
                (concepts, "can have its whole record in `non_passing.bam` too"), (concepts, "next adapter; that adapter"),
                (concepts, "A prime marks an adapter read on the opposite strand"), (concepts, "two slots lie between G and Q"),
                (concepts, "the skipped adapter and the start"), (skera, "the skipped adapter and the first part"),
                (skera, "Uppercase marks the adapters and the site skera called D"), (skera, "where every real adapter sits (the full-read"),
                (skera, "repeat lies within 10 bp"), (skera, "so the look-alikes are not")]
    assert [s for text, s in replaced if s in text] == []
    # ND is named where it is first used, and the prime is explained before the first adapter written with one
    assert re.search(r"\bND\b.*", readme).group(0).startswith("ND, with no TSO-TSO depletion")
    assert concepts.index("A prime marks") < concepts.index("X'")
    # the study's own segmentation, with the 4 bp rule, has its own section
    assert "within 4 bp of a TSO or Read1 hit" in section(raw["docs/concepts.md"], "How the study cuts reads")
    assert "within 4 bp" not in section(raw["docs/concepts.md"], "What skera does")


def test_concepts_and_reader_guide_sections():
    concepts = (REPO / "docs" / "concepts.md").read_text()
    assert [x[3:] for x in concepts.splitlines() if x.startswith("## ")] == CONCEPTS_SECTIONS
    guide = (REPO / "docs" / "reader_guide.md").read_text()
    assert [x[3:] for x in guide.splitlines() if x.startswith("## ")] == READER_SECTIONS
    for s in ("uv tool install .", "pip install .", SCOPE, "### The one known difference from v10", "### Time and memory"):
        assert s in guide, s


def test_data_md_lists_every_input_and_the_argentag_source():
    data = (REPO / "docs" / "data.md").read_text()
    for column in ("hifi_bam", "fail_bam", "segmented_bam", "non_passing_bam", "zmw_metrics", "skera_report", "fltnc_bam"):
        assert f"`{column}`" in data, column
    assert "https://downloads.pacbcloud.com/public/dataset/Kinnex-single-cell-RNA/DATA-RevioSPRQ-Kinnex-ArgenTag-MDAcellLine/MDA-8fold/0-Sreads/" in data
    assert "## Reference genome" in data and "## Calibration set" in data


def test_methods_texts():
    main = (REPO / "docs" / "manuscript_methods.md").read_text()
    body = " ".join(x for x in main.splitlines()[1:] if not x.startswith("#"))
    assert 150 <= len(body.split()) <= 250, len(body.split())
    assert "Supplementary Note" in main
    # the hash sample covers the three libraries' HiFi reads only: the calibration comparisons, the fail reads and ArgenTag
    # each use their own read set, and the main text names each
    numbers = json.loads((GOLDEN / "v10_numbers.json").read_text())
    assert "Every analysis" not in body
    for s in (f"hash sample of {render.FORMATS['pct0'](numbers['fraction'])} of each cell's ZMWs", "fail reads, a separate 1-in-10 sample",
              f"the first {render.fmt(numbers['cal_hifi_reads'])} ND HiFi reads", "all reads skera did not pass"):
        assert s in body, s
    supp = (REPO / "docs" / "supplementary_methods.md").read_text()
    assert "Supplementary Note" in supp
    for s in ("edlib", "lima 2.13.0", "isoseq refine 4.3.0", "minimap2 2.30", "skera 1.2.0", "26.1.0", "AAGCAGTGGTATCAACGCAGAGTAC",
              "ACACGACGCTCTTCCGATCT"):
        assert s in supp, s


def test_open_items_each_in_one_marked_place():
    skip = {".venv", "out", "qc", "build", "dist", "tests"}
    files = [p for p in REPO.rglob("*") if p.is_file() and p.suffix in {".md", ".in", ".cff", ".toml", ".tsv", ".py", ".sh", ".html", ".txt", ".yml"}
             and not any(part.startswith(".") or part in skip for part in p.relative_to(REPO).parts)]

    def where(marker: str) -> dict:
        return {p.relative_to(REPO).as_posix(): n for p in files if (n := p.read_text(errors="replace").count(marker))}

    assert where("ACCESSION_TBD") == {"README.md.in": 1, "README.md": 1, "docs/data.md.in": 1, "docs/data.md": 1, "CITATION.cff": 1}
    assert where("ARGENTAG_WORDING_TBD") == {"docs/data.md.in": 1, "docs/data.md": 1}
    assert where("AUTHORS_TBD") == {"CITATION.cff": 1}
    assert where("RERUN_TBD") == {}
    assert ARGENTAG_SENTENCE in (REPO / "docs" / "data.md").read_text()


def test_text_rules_on_every_markdown_file():
    files = markdown_files()
    assert {"README.md", "README.md.in", "CHANGELOG.md", "docs/concepts.md.in"} <= {p.relative_to(REPO).as_posix() for p in files}
    for p in files:
        text = p.read_text()
        assert "\u2013" not in text and "\u2014" not in text, p.name
        assert banned_hits(text) == [], p.name


def hand_typed(source: str) -> set[str]:
    """The number tokens of a .md.in source outside placeholders, slots, code, quotes, URLs and link targets."""
    text = re.sub(r"```.*?```", " ", source, flags=re.S)
    text = re.sub(r"^> .*$", " ", text, flags=re.M)
    text = re.sub(r"`[^`]*`|\{[a-z0-9_]+(?:\.\.[a-z0-9_]+)?\|[a-z0-9_]+\}|<!--.*?-->|\]\([^)]*\)|https?://\S+", " ", text, flags=re.S)
    return set(NUM_TOKEN.findall(text))


def test_no_number_typed_by_hand():
    assert hand_typed("a {nd_mirror_share|pct1} b `x 0.25` 16.1% [l](docs/f2.md) https://x.org/608 c 8-fold 99th") == {"16.1%", "8", "99th"}
    assert hand_typed("{lo_1..hi_2|pct1} and {lo_key..hi_key|pct2}") == set()
    assert hand_typed("per 1,000, then Fig. 1: and 3:45:04.") == {"1,000", "1", "3:45:04"}
    sources = [REPO / "README.md.in", *sorted((REPO / "docs").glob("*.md.in"))]
    typed = {p.name: sorted(hand_typed(p.read_text()) - CONSTANTS) for p in sources}
    assert typed == {p.name: [] for p in sources}


def test_ci_installs_the_dev_extra_and_runs_pytest_from_the_venv():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "uv sync --extra dev" in ci and "uv run python -m pytest" in ci and "uv run pytest" not in ci


def test_citation_cff():
    cff = (REPO / "CITATION.cff").read_text()
    for line in ['cff-version: 1.2.0', 'version: "0.1.0"', 'license: MIT', 'family-names: Picelli', 'given-names: Simone',
                 'orcid: "https://orcid.org/0000-0002-1747-614X"', 'repository-code: "https://github.com/FullLengthFanatic/flight-kinnex-qc"']:
        assert line in cff, line
    assert not re.search(r"^doi:", cff, flags=re.M)
    assert "https://skera.how" in cff and "https://github.com/PacificBiosciences/pbbioconda/issues/608" in cff
    assert "\u2013" not in cff and "\u2014" not in cff and banned_hits(cff) == []
