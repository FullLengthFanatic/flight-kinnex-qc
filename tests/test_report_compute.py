"""Tests for study/report/compute.py: the report's source tables and numbers.json. The real test computes them from the v10
inputs and compares the tables with tests/golden/v10_tables/ byte for byte and numbers.json with the frozen
tests/golden/v10_numbers.json (the input of the render step's golden test)."""
import json
import re

import pytest

from pipeline_helpers import GOLDEN, GOLDEN_INPUTS, REPO

import compute
import figures
from config import load_samples
from kinnex_qc import metrics, patterns, skera

GOLDEN_TABLES = GOLDEN / "v10_tables"
V10_NUMBERS = GOLDEN / "v10_numbers.json"
V10_FIG3_ZMWS = [986295, 1051947, 1051859, 1379616]      # the caption of v10's Supplementary Fig. 3
BD144 = (("nd", "ND"), ("hm", "HM"), ("kin", "kin"))      # key prefix, library

# Every number the v10 Note printed in its text (key findings, terms, sections, figure captions, table legends, the
# false-cut drawing and the footer) and every value it derived while building a table, with the numbers.json key that
# now carries it. Left side: the key; right side: the expression of the v10 build script's render step it replaces (bench_*
# cells: the share v10 printed in Supplementary Table 4, before formatting).
# Table cells that are plain values of a source table (Supplementary Tables 1, 3, 5, 6, 7 and 9) are read from
# report/tables/ by the render step and need no key.
REQUIRED = {
    # run and constants
    "fraction": "FRACTION",
    "head_mol": "HEAD_MOL",
    "arg_hifi_reads": "ARG_HIFI",
    "fig3_zmws": "track_zms",
    "kinnex_qc_version": "PIPE / 'COMMIT' (now the package version)",
    "kinnex_qc_commit": "PIPE / 'COMMIT' (now the git HEAD that ran compute)",
    "run_commits": "run_commits()",
    "version_python": "platform.python_version()",
    **{f"version_{k}": f"im.version('{k}')" for k in ("polars", "pysam", "edlib", "numpy", "scipy", "plotly", "kaleido")},
    # libraries and chain patterns
    "hifi_analysed_min": "min(l['hifi_analysed'] for l in lib.values())",
    "hifi_analysed_max": "max(l['hifi_analysed'] for l in lib.values())",
    "nd_mirror_by_tenth_min": "lib['ND']['mirror_by_tenth_min']",
    "nd_mirror_by_tenth_max": "lib['ND']['mirror_by_tenth_max']",
    "nd_mirror_share": "p[('ND', 'mirror')]['share']",
    "hm_mirror_share": "p[('HM', 'mirror')]['share']",
    "kin_mirror_share": "p[('kin', 'mirror')]['share']",
    "nd_mirror_proper_per_read": "nd('mirror')['proper_per_read']",
    "nd_complete_proper_per_read": "nd('complete')['proper_per_read']",
    "nd_mirror_median_read_len": "nd('mirror')['median_read_len']",
    "nd_complete_median_read_len": "nd('complete')['median_read_len']",
    "nd_mirror_skera_pass_share": "nd('mirror')['skera_pass_share']",
    "nd_mirror_skera_sreads_per_read": "nd('mirror')['skera_sreads_per_read']",
    "nd_truncated_skera_sreads_per_read": "nd('truncated')['skera_sreads_per_read']",
    "nd_truncated_proper_per_read": "nd('truncated')['proper_per_read']",
    "nd_truncated_non_passing_share": "1 - nd('truncated')['skera_pass_share']",
    "nd_mirror_molecules_fewer": "nd('complete')['proper_per_read'] - nd('mirror')['proper_per_read']",
    "nd_tso_tso_per_1000": "a[('ND', 'TSO-TSO')]",
    "hm_tso_tso_per_1000": "a[('HM', 'TSO-TSO')]",
    "kin_tso_tso_per_1000": "a[('kin', 'TSO-TSO')]",
    "nd_r1_r1_per_1000": "a[('ND', 'R1-R1')]",
    "tso_tso_ratio_nd_hm": "a[('ND', 'TSO-TSO')] / a[('HM', 'TSO-TSO')]",
    "tso_tso_ratio_nd_kin": "a[('ND', 'TSO-TSO')] / a[('kin', 'TSO-TSO')]",
    "nd_mirror_junction_tso_tso_share": "tt_share_nd",
    "nd_tso_tso_base_share": "d['tso_tso_base_share_ND']",
    "nd_tso_tso_only_skera_pass_share": "tt_only_pass",
    # validation and the benchmark (Supplementary Tables 3 and 4)
    "synthetic_error_rates": "gates['ND']['synthetic_error_rates']",
    "tso_toward_lower_adapter_max": "max(g['tso_toward_lower_adapter'] for g in gates.values())",
    "bench_reads_per_class": "int(d['bench_classes']['n'][0])",
    "bench_hifi_class_min": "hifi_class_min",
    **{f"bench_{c}_{r}": f"bench_rows cell ({c}, error {r.replace('_', '.')})"
       for c in ("complete", "tso_tso_mirror", "r1_r1_mirror", "truncated_no_a", "truncated_no_q", "adapter_destroyed",
                 "decoy_motif", "opposite_join")
       for r in ("0_001", "0_005", "0_01", "0_05")},
    **{f"bench_mirror_precision_{r}": f"bench[{r.replace('_', '.')}]['mirror_precision']" for r in ("0_001", "0_005", "0_01", "0_05")},
    # skera false cuts: anatomy (Supplementary Fig. 1) and calibration set
    "false_cut_reads_share_min": "min(v['reads_with_false_cut'] for v in fca.values())",
    "false_cut_reads_share_max": "max(v['reads_with_false_cut'] for v in fca.values())",
    "nd_false_cut_missed_is_d": "fca['ND']['missed_is_D']",
    "nd_false_cut_missed_same_letter": "fca['ND']['missed_same_letter_as_skera_call']",
    "nd_false_cut_median_cut_to_missed_bp": "fca['ND']['median_cut_to_missed_adapter_bp']",
    "nd_false_cut_fused_far_end_at_real_adapter": "fca['ND']['fused_far_end_at_real_adapter']",
    "false_cut_frag_lima_max": "max(v['fates']['frag_lima'] for v in fca.values())",
    "false_cut_fused_lima_min": "min(v['fates']['fused_lima'] for v in fca.values())",
    "false_cut_fused_lima_max": "max(v['fates']['fused_lima'] for v in fca.values())",
    "false_cut_baseline_lima_pass_min": "min(v['baseline_proper_sreads']['lima_pass'] for v in fca.values())",
    "kin_false_cut_fused_lima": "fca['kin']['fates']['fused_lima']",      # in false_cut_svg(fca['kin'])
    "kin_false_cut_frag_lima": "fca['kin']['fates']['frag_lima']",
    "kin_false_cut_molecules_per_hifi_read": "2 * fca['kin']['false_cuts'] / bud['kin']['hifi_reads']",
    "cal_false_cuts_closest_to_d_3_to_5_edits_share": "fc['cuts_closest_to_D_3_to_5_edits'] / fc['cuts']",
    "cal_false_cuts_within_2_edits_share": "fc['cuts_within_2_edits_of_any_adapter'] / fc['cuts']",
    "cal_false_cuts_ctcctcc_share": "fc['cuts_with_ctcctcc_repeat'] / fc['cuts']",
    "cal_skipped_adapters_le1_edit_with_primer_share": "fc['inside_le1_edit_with_primer'] / fc['our_adapters_inside_sreads']",
    # where our segmentation misses an adapter skera finds (span() in the text and Supplementary Table 2)
    **{f"{k}_{m}": f"{m}(v['{k}'] for v in om.values())" for k in ("missed_adapter_share", "skipped_slot_share", "sread_lima_pass_missed",
                                                                   "sread_lima_pass_complete") for m in ("min", "max")},
    # mirror arrays (section 3)
    "mirror_rule_overall": "d['mirror_rule_overall']",
    "mirror_rule_reads": "d['mirror_rule_reads']",
    "mirror_fold_per_letter": "sum(step) / 2",
    "double_mirror_median_molecules_h": "jc['H']['double_median_molecules']",
    "double_mirror_median_read_len_min": "min(r['double_median_read_len'] for r in jc.values() if r['double_mirrors'])",
    "double_mirror_median_read_len_max": "max(r['double_median_read_len'] for r in jc.values() if r['double_mirrors'])",
    **{f"double_mirrors_{x.lower()}": f"jc['{x}']['double_mirrors']" for x in "BCDEFGH"},
    **{f"ordinary_mirrors_{x.lower()}": f"jc['{x}']['ordinary_mirrors']" for x in "BCDEFGH"},
    "double_mirrors_min": "min(jc[l]['double_mirrors'] for l in 'BCDEFGH')",
    "double_mirrors_max": "max(jc[l]['double_mirrors'] for l in 'BCDEFGH')",
    "hifi_yield_ratio_h_b": "hifi_hb",
    "ordinary_mirror_median_read_len_h": "jc['H']['ordinary_median_read_len']",
    "ordinary_mirror_median_read_len_b": "jc['B']['ordinary_median_read_len']",
    "tso_tso_median_len_q": "jc['Q']['tso_tso_median_len']",
    "tso_tso_median_len_other_min": "min(other_tt)",
    "tso_tso_median_len_other_max": "max(other_tt)",
    # yield (section 4)
    "nd_mapped_fltnc_per_hifi_zmw": "yv['ND']['mapped_fltnc_per_hifi_zmw']",
    "kin_mapped_fltnc_per_hifi_zmw": "yv['kin']['mapped_fltnc_per_hifi_zmw']",
    "loss_ours": "dc['ours_gap'] / yv['kin']['proper_per_hifi_read']",
    "loss_mapped": "dc['mapped_gap'] / yv['kin']['mapped_fltnc_per_hifi_zmw']",
    "gap_ours": "dc['ours_gap']",
    "gap_mirror_capacity": "dc['mirror_capacity']",
    "gap_mapped": "dc['mapped_gap']",
    **{f"downstream_{g}_{f}": f"dsv['{group}']['{f}']" for g, group in (("unwritten", "mirror reads, not written by skera"),
                                                                          ("complete", "complete reads"))
       for f in ("molecules", "lima_5p3p", "refine", "mapped")},
    **{f"budget_{k}_{f}": f"bud['{lib}']['{f}']" for k, lib in BD144 for f in ("unfilled_slots", "not_written_cleanly", "kept_standard")},
    **{f"budget_{k}_{f}": f"bp['{lib}']['{f}']" for k, lib in BD144 for f in ("artifact", "downstream", "total", "unwritten_share", "kept_full")},
    "budget_other_mapped_max": "other_mapped_max",
    # concatenation bias (section 5)
    "partners_shorter_bp_min": "-q1_hi",
    "partners_shorter_bp_max": "-q1_lo",
    "diff_vs_mt_bp_abs_max": "max(abs(r['diff_vs_MT_bp']) for r in q1.values())",
    "ipw_empirical_diff_min": "min(ipw_e)",
    "ipw_empirical_diff_max": "max(ipw_e)",
    "ipw_model_diff_min": "min(ipw_m)",
    "ipw_model_diff_max": "max(ipw_m)",
    **{f"{k}_d_residual{s}": f"q1['{lib}']['D_residual{s}']" for k, lib in BD144 for s in ("", "_lo", "_hi")},
    "lag1_mean": "sum(r['lag1'] for r in q1.values()) / 3",
    # ArgenTag (section 6, Supplementary Tables 1, 7 and 8)
    **{f"arg_{f}": f"af['{f}']" for f in ("mirror_share", "tso_tso_per_1000", "r1_r1_per_1000", "non_passing_reads", "hifi_zmws",
                                           "junctions_tso_tso_share", "mirror_junctions", "single_tt_matching_rule",
                                           "proper_share_of_molecules", "usable_per_mirror_read", "sreads_per_passing_read",
                                           "sreads_per_mirror_read")},
    "arg_junctions_at_a_tso_tso": "af['junctions_at_A_tso_tso']",
    "arg_mirror_shortfall_ours": "af['mirror_shortfall_ours_per_hifi_zmw']",
    "arg_mirror_shortfall_sreads": "af['mirror_shortfall_sreads_per_hifi_zmw']",
    "arg_mirror_shortfall_ours_share": "af['mirror_shortfall_ours_per_hifi_zmw'] / af['sreads_per_passing_read']",
    "arg_mirror_shortfall_sreads_share": "af['mirror_shortfall_sreads_per_hifi_zmw'] / af['sreads_per_passing_read']",
    "arg_independent_agree": "d['arg_indep']['agree']",
    "arg_independent_sample": "d['arg_indep']['sample']",
    "arg_independent_agree_share": "d['arg_indep']['agree'] / d['arg_indep']['sample']",
    "arg_sreads": "st[ARG]['sreads']",
    # the S-read tests (Supplementary Table 8; ArgenTag's row merged with the parent-read truth, av_truth)
    **{f"{k}_{f}": f"st['{lib}']['{f}']" for k, lib in (*BD144, ("arg", "ArgenTag"))
       for f in ("true_mirror", "q_not_a", "q_not_a_ppv", "flag", "flag_ppv", "flag_sensitivity", "flag_non_mirror")},
    "bd144_flag_non_mirror_min": "bg_lo",
    "bd144_flag_non_mirror_max": "bg_hi",
    "non_mirror_flagged_1_0_max": "max(st[l]['non_mirror_flagged_1.0'] for l in ORDER3)",
    "non_mirror_flagged_1_5_max": "max(st[l]['non_mirror_flagged_1.5'] for l in ORDER3)",
    "nd_sensitivity_1_0": "st['ND']['sensitivity_1.0']",
    "nd_sensitivity_1_5": "st['ND']['sensitivity_1.5']",
    "cal_hifi_reads": "200,000 (a literal in v10's text: the calibration set's HiFi reads, hv['1.2.0']['hifi_zmws'])",
    "skera120_flagged": "hv['1.2.0']['flagged']",
    "skera2610_flagged": "hv['26.1.0']['flagged']",
    "skera120_ppv": "hv['1.2.0']['ppv']",
    "skera2610_ppv": "hv['26.1.0']['ppv']",
    "tso_both_ends_background": "tv_bg",
    "tso_both_ends_checked": "tv_n",
    "tso_both_ends_background_share": "tv_bg / tv_n",
    # what skera writes from mirror reads (section 6, Supplementary Table 10)
    "mirror_arms_non_passing_share": "ms['mirror_reads_non_passing_share']",
    "mirror_arms_single_tso_tso_reads": "ms['single_tso_tso_mirror_reads']",
    "mirror_arms_clean_share_mirror": "ms['clean_share_mirror_reads']",
    "mirror_arms_clean_share_complete": "ms['clean_share_complete_arrays']",
    "mirror_arms_complete_reads_baseline": "ms['complete_reads_baseline']",
    **{f"mirror_{reg}_{c.lower()}_share": f"mol[('{reg}', '{c}')]['share']" for reg in ("after", "before", "all_mirror_arms", "complete_array")
       for c in ("CLEAN", "FUSED", "PARTIAL", "NOT_WRITTEN")},
    **{f"mirror_{reg}_molecules": f"sum(mol[('{reg}', c)]['n'] for c in classes)" for reg in ("after", "before", "all_mirror_arms", "complete_array")},
    "mirror_reads_after_only": "rd['AFTER_ONLY']",
    "mirror_reads_spans_junction": "rd['SPANS_JUNCTION']",
    "mirror_reads_no_sread": "rd['NO_SREAD']",
    "mirror_reads_no_sread_at_q_share": "nosread_q",
    "skera120_found_after_arm": "fa['after_arm']",
    "skera120_found_before_arm": "fa['before_arm']",
    "skera120_ligations_same_letter": "s12['ligations_same_letter']",
    "skera120_ligations_total": "s12['ligations_total']",
    "skera_versions_share_diff_max": "ver_diff / 100",
}
STRINGS = {"synthetic_error_rates", "kinnex_qc_version", "kinnex_qc_commit"} | {k for k in REQUIRED if k.startswith("version_")}
LISTS = {"fig3_zmws", "run_commits"}
ENVIRONMENT = {"kinnex_qc_version", "kinnex_qc_commit", "run_commits"} | {k for k in REQUIRED if k.startswith("version_")}


def test_v10_numbers_carry_every_number_of_the_text():
    numbers = json.loads(V10_NUMBERS.read_text())
    assert sorted(set(REQUIRED) - set(numbers)) == []


def test_v10_numbers_are_unformatted():
    """Counts are ints, measured values floats; strings only for version, commit and the error-rate list."""
    for k, v in json.loads(V10_NUMBERS.read_text()).items():
        if k in STRINGS:
            assert isinstance(v, str), k
        elif k in LISTS:
            assert isinstance(v, list), k
        else:
            assert isinstance(v, int | float) and not isinstance(v, bool), k
    numbers = json.loads(V10_NUMBERS.read_text())
    assert numbers["fig3_zmws"] == V10_FIG3_ZMWS and numbers["fraction"] == 0.18
    assert isinstance(numbers["arg_non_passing_reads"], int) and isinstance(numbers["mirror_rule_reads"], int)


def test_rules_are_the_package_ones():
    """compute keeps no copy of a rule kinnex_qc holds: the unwritten-head test and HEAD_MOL (skera.unwritten_head, which
    study/argentag/full_read_view.py and study/calibration/head_version_check.py use for the same Supplementary Tables 7
    and 8), the mirror-junction rule, the ND junction tables, the array-size rule and the letter order (kinnex_qc.metrics,
    patterns.MAS_ORDER). A change there then moves the BD144 rows and the ArgenTag rows together; the rules' own tests
    are in tests/test_skera.py and tests/test_metrics.py."""
    assert compute.unwritten_head is skera.unwritten_head and compute.HEAD_MOL is skera.HEAD_MOL
    assert compute.mirror_junctions is skera.mirror_junctions and compute.metrics is metrics
    assert figures.MAS_ORDER is patterns.MAS_ORDER
    assert not {"sread_zmws", "LETTERS"} & set(vars(compute))


def test_main_reads_the_run_under_the_out_root(tmp_path, monkeypatch):
    """compute.py --config --out-root, as run_study.sh calls it: sample outputs and analyses from <run>, the calibration set
    and ArgenTag outputs from the out root, the report into <run>/report; <run> is <out root>/hash_<fraction> below 1."""
    samples = tmp_path / "samples.tsv"
    samples.write_text((REPO / "study" / "samples.tsv").read_text())
    (tmp_path / "study.toml").write_text(f'fraction = 0.18\nsamples = "{samples}"\nout_root = "<out>"\n')
    out = tmp_path / "out"
    (out / "argentag").mkdir(parents=True)
    for name in compute.ARGENTAG_FILES:
        (out / "argentag" / name).write_bytes(b"")
    (out / "logs" / "BD144_ND").mkdir(parents=True)
    (out / "logs" / "BD144_ND" / "s01_extract_segments.ok").write_text("2026-11-05 01:00:00 abcdef0123 10 fraction 0.18\n")
    seen = {}
    monkeypatch.setattr(compute, "write_report", lambda inputs, cells, fraction, report: seen.update(
        inputs=inputs, cells=cells, fraction=fraction, report=report))
    compute.main(["--config", str(tmp_path / "study.toml"), "--out-root", str(out)])
    run = out / "hash_0.18"
    assert seen["inputs"] == compute.Inputs(run=run, analyses=run / "analyses", calibration=out / "calibration",
                                            argentag=out / "argentag",
                                            markers=(out / "logs" / "BD144_ND" / "s01_extract_segments.ok",))
    assert seen["report"] == run / "report" and seen["report"].is_dir()
    assert seen["fraction"] == 0.18 and seen["cells"]["BD144_ND"] == "PB538_3_r0912"


def test_main_names_the_missing_argentag_outputs(tmp_path):
    """Without the ArgenTag outputs compute stops at the start with one line, not a FileNotFoundError at the end."""
    samples = tmp_path / "samples.tsv"
    samples.write_text((REPO / "study" / "samples.tsv").read_text())
    (tmp_path / "study.toml").write_text(f'fraction = 1\nsamples = "{samples}"\nout_root = "<out>"\n')
    out = tmp_path / "out"
    (out / "argentag").mkdir(parents=True)
    (out / "argentag" / "full_read_summary.tsv").write_text("")
    with pytest.raises(SystemExit) as e:
        compute.main(["--config", str(tmp_path / "study.toml"), "--out-root", str(out)])
    msg = str(e.value)
    assert "\n" not in msg and "the Note needs the ArgenTag check" in msg
    assert str(out / "argentag" / "independent_scan.txt") in msg and str(out / "argentag" / "full_read_summary.tsv") not in msg
    assert not (out / "report").exists() and not (out / "hash_1").exists()


def test_numbers_keys_are_lowercase_words():
    """Every numbers.json key is a template placeholder name: lowercase letters, digits and underscores (no dots in the
    benchmark's error rates)."""
    keys = list(json.loads(V10_NUMBERS.read_text())) + list(REQUIRED)
    assert [k for k in keys if not re.fullmatch(r"[a-z0-9_]+", k)] == []


def test_run_commits_from_the_markers(tmp_path):
    for name, head in (("a.ok", "f856e6a1234"), ("b.ok", "5e988fe999"), ("c.ok", "f856e6a1234")):
        (tmp_path / name).write_text(f"2026-11-05 01:00:00 {head} 10\n")
    assert compute.run_commits(sorted(tmp_path.glob("*.ok"))) == ["5e988fe", "f856e6a"]


@pytest.mark.real
def test_v10_inputs_give_the_golden_tables_and_numbers(local, tmp_path):
    """From the v10 inputs (the hash sample's stage outputs, the v10 analysis outputs frozen in tests/golden/v10_inputs,
    the calibration set and the ArgenTag outputs), the 36 tables equal v10's byte for byte and numbers.json equals the
    frozen tests/golden/v10_numbers.json, apart from the software versions and commits."""
    inputs = compute.Inputs(run=local["hash_dir"], analyses=GOLDEN_INPUTS, calibration=local["calibration_dir"],
                            argentag=local["argentag_dir"] / "analysis", markers=tuple(sorted(local["hash_dir"].glob("*/logs/*.ok"))))
    cells = {s: x.cell for s, x in load_samples(REPO / "study" / "samples.tsv").items()}
    report = tmp_path / "report"
    compute.write_report(inputs, cells, 0.18, report)
    names = sorted(p.name for p in GOLDEN_TABLES.iterdir())
    assert len(names) == 36 and sorted(p.name for p in (report / "tables").iterdir()) == names
    assert [n for n in names if (report / "tables" / n).read_bytes() != (GOLDEN_TABLES / n).read_bytes()] == []
    numbers, frozen = (json.loads(p.read_text()) for p in (report / "numbers.json", V10_NUMBERS))
    assert numbers["run_commits"] == ["5e988fe", "f856e6a"]
    assert {k: v for k, v in numbers.items() if k not in ENVIRONMENT} == {k: v for k, v in frozen.items() if k not in ENVIRONMENT}
    assert list(numbers) == list(frozen)
    tracks = json.loads((report / "fig3_reads.json").read_text())
    assert [t["zm"] for t in tracks] == V10_FIG3_ZMWS
    assert len(figures.fig3(tracks).data) == 33                  # as v10's Supplementary Fig. 3
