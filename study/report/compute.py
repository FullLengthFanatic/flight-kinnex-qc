#!/usr/bin/env python3
"""The compute step of the Supplementary Note: from a study run's outputs, writes into <run>/report/
  tables/*.tsv     the source data of every figure and table of the Note (36 tables),
  numbers.json     every number the Note's text quotes and every value its tables derive, unformatted (the render step
                   formats them), plus the software versions, the commit of the checkout that ran this step and
                   the commits the run's step markers record,
  fig3_reads.json  the track data of the four example reads of Supplementary Fig. 3 (one complete array and three
                   mirror arrays of BD144_ND).
Read: the sample outputs <run>/<sample>/ (read_summary, annot_segments, segments, fate and read_features parquets, q1,
q2 and gates), the analyses <run>/analyses/<analysis>/, the calibration set's checks and the ArgenTag outputs under the
out root (<out root>/calibration, <out root>/argentag), and the cell of each sample from the samples table. <run> is the
out root at fraction 1 and <out root>/hash_<fraction> below it, as in run_study.sh. Nothing is typed in by hand except
the HiFi read count of the ArgenTag cell (ARG_HIFI), from Table 1 of the ArgenTag/PacBio technical note.
Usage: study/report/compute.py --config study/study.toml [--out-root <out root>]
"""
import argparse
import importlib.metadata as im
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from kinnex_qc import metrics
from kinnex_qc.skera import HEAD_MOL, mirror_junctions, unwritten_head

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config, load_samples, out_dir  # noqa: E402
from read_tracks import tracks  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
SAMPLES = [("BD144_ND", "ND", "None"), ("BD144_HM", "HM", "Home-made biotin-Read1 pulldown"),
           ("BD144_kin", "kin", "Kinnex TSO artifact removal")]
ORDER3 = ["ND", "HM", "kin"]
ARG = "ArgenTag"
ARG_HIFI = 1_964_147         # HiFi reads of the ArgenTag 8-fold cell, Table 1 of the ArgenTag/PacBio technical note
KEY = {"ND": "nd", "HM": "hm", "kin": "kin", ARG: "arg"}     # library -> prefix of its numbers.json keys
FIG3_CHAINS = ("Q' H' H Q", "Q' H' G' G H Q", "Q' H' G' F' E' D' D E F G H Q")


@dataclass(frozen=True)
class Inputs:
    """Where compute reads. run: the sample outputs, <run>/<sample>/. analyses: <analyses>/<analysis>/, as run_study.sh
    names them. calibration: head_test_skera_versions.tsv and tso_end_validation.tsv. argentag: argentag_sread_coords.parquet,
    full_read_summary.tsv, full_read_junctions.tsv, flag_validation.tsv and independent_scan.txt. markers: the .ok markers
    of the run's steps (date, time, git HEAD, ...)."""
    run: Path
    analyses: Path
    calibration: Path
    argentag: Path
    markers: tuple[Path, ...] = ()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def per_sample(o: Path) -> dict:
    summ = pl.read_parquet(o / "read_summary.parquet", columns=["zm", "source", "read_len", "chain", "primary_pattern", "n_mas", "skera_passing"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    segs = pl.read_parquet(o / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal", "left_token", "right_token"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    skera = pl.read_parquet(o / "segments.parquet", columns=["zm"])
    fate = pl.read_parquet(o / "fate.parquet", columns=["zm", "skera_passing", "lima_pass", "mapped_fltnc"])
    proper = segs.group_by("zm").agg(((pl.col("end_class") == "proper") & ~pl.col("internal")).sum().alias("proper"))
    nsk = skera.group_by("zm").len().rename({"len": "skera_sreads"})
    reads = summ.join(proper, on="zm", how="left").join(nsk, on="zm", how="left").with_columns(
        pl.col("proper").fill_null(0), pl.col("skera_sreads").fill_null(0),
        pl.col("primary_pattern").replace(metrics._OTHER_PATTERNS).alias("pattern"))
    junction = mirror_junctions(segs)
    bases = segs.group_by("end_class").agg((pl.col("end") - pl.col("start")).cast(pl.Int64).sum().alias("bp"))
    return {"reads": reads, "segs": segs, "junction": junction, "fate": fate, "bases": bases, "q1": load_json(o / "q1" / "q1_summary.json"),
            "q2": load_json(o / "q2" / "q2_summary.json"), "gates": {g: load_json(o / "gates" / f"gate_{g}.json") for g in
                                                                     ("synthetic", "adapter_index", "lima", "skera")},
            "skera_sreads": skera.height}


def compute(inputs: Inputs, cells: dict[str, str]) -> dict:
    """Every table and value of the Note, from the inputs; cells: sample -> SMRT Cell."""
    run, qc = inputs.run, inputs.analyses
    d = {"lib": {}, "pattern": [], "artifacts": [], "skera_by_pattern": [], "yield": [], "q1": [], "gates": []}
    raw = {}
    for s, short, dep in SAMPLES:
        x = per_sample(run / s)
        raw[short] = x
        reads, q2, q1 = x["reads"], x["q2"], x["q1"]
        n = reads.height
        wc = {a["id"]: a["value"] for a in load_json(qc / "skera_whole_cell" / f"{s}.json")["attributes"]}
        order = (pl.read_parquet(run / s / "read_summary.parquet", columns=["source", "primary_pattern"]).filter(pl.col("source") == "hifi")
                 .with_row_index().with_columns((pl.col("index") * 10 // n).alias("tenth"))
                 .group_by("tenth").agg((pl.col("primary_pattern") == "mirror").mean().alias("mirror")))
        d["lib"][short] = {"sample": s, "short": short, "cell": cells[s], "depletion": dep,
                           "hifi_cell": q2["hifi_reads_total"], "hifi_analysed": n, "skera_sreads": x["skera_sreads"],
                           "sreads_per_read_sample": x["skera_sreads"] / n, "sreads_per_read_cell": wc["read_segmentation.mean_array_size"],
                           "passing_sample": float(reads["skera_passing"].mean()), "full_array_cell": wc["read_segmentation.percent_full_array"] / 100,
                           "mirror_by_tenth_min": float(order["mirror"].min()), "mirror_by_tenth_max": float(order["mirror"].max())}
        for pat, g in reads.group_by("pattern"):
            d["pattern"].append({"library": short, "pattern": pat[0], "reads": g.height, "share": g.height / n,
                                 "proper_per_read": g["proper"].mean(), "skera_sreads_per_read": g["skera_sreads"].mean(),
                                 "skera_pass_share": g["skera_passing"].mean(), "median_read_len": g["read_len"].median()})
        for k, lab in (("tso_tso_per_1000", "TSO-TSO"), ("r1_r1_per_1000", "R1-R1"), ("tso_only_per_1000", "TSO only"),
                       ("r1_only_per_1000", "R1 only"), ("internal_per_1000", "Fused (no adapter between)")):
            d["artifacts"].append({"library": short, "class": lab, "per_1000_molecules": q2[k]})
        f = x["fate"].join(reads.select("zm", "pattern"), on="zm", how="left")
        for (pat, sp), g in f.group_by("pattern", "skera_passing"):
            d["skera_by_pattern"].append({"library": short, "pattern": pat, "skera_passing": sp, "sreads": g.height,
                                          "lima_pass": g["lima_pass"].mean(), "mapped": g["mapped_fltnc"].mean()})
        d["yield"].append({"library": short, "usable_per_hifi_zmw": q2["usable_per_hifi_zmw"], "proper_per_hifi_read": reads["proper"].mean(),
                           "mapped_fltnc_per_hifi_zmw": q2["mapped_fltnc_per_hifi_zmw"],
                           "not_recovered_by_skera_per_hifi_zmw": q2["not_recovered_by_skera_per_hifi_zmw"],
                           "rejected_sreads_mapped_share": q2["rejected_sreads_mapped_share"], "polyT_ge_20": q2["polyT_ge_20"],
                           "complete_fraction": q2["complete_fraction"], "mirror_fraction": q2["mirror_fraction"],
                           "skera_agreement": q2["skera_agreement"]})
        v = next(v for v in q1["values"] if v["value"] == "insert_len")
        c4 = next(c for c in v["cutoffs"] if c["cutoff"] == 4000)
        ipw = {r["w"]: r for r in v["ipw"] if r["cutoff"] == 4000}
        d["q1"].append({"library": short, "complete_arrays": v["n_arrays"], "index_ge_4kb": int(c4["n_index"]),
                        "partner_mean": c4["partner_mean"], "null_P_mean": c4["P_mean"], "null_P_lo": c4["P_lo"], "null_P_hi": c4["P_hi"],
                        "M_T_mean": c4["MT_mean"], "M_T_lo": c4["MT_lo"], "M_T_hi": c4["MT_hi"],
                        "diff_vs_P_bp": c4["diff_vs_P_bp"], "diff_vs_MT_bp": c4["diff_vs_MT_bp"],
                        **{f"ipw_{w}_{k}": ipw[w][c] for w in ("empirical", "model")
                           for k, c in (("diff", "diff_vs_null_bp"), ("lo", "diff_lo"), ("hi", "diff_hi"))},
                        "oe_2plus_vs_P": v["oe_2plus"]["null P"]["4000"], "oe_2plus_vs_MT": v["oe_2plus"]["M_T"]["4000"],
                        "D": v["D"]["observed"], "D_boot_lo": v["D"]["boot_lo"], "D_boot_hi": v["D"]["boot_hi"],
                        "D_MT_lo": v["D"]["MT_lo"], "D_MT_hi": v["D"]["MT_hi"], "D_residual": v["D"]["residual"],
                        "D_residual_lo": v["D"]["residual_lo"], "D_residual_hi": v["D"]["residual_hi"], "D_residual_z": v["D"]["residual_z"],
                        "lag1": float(pl.read_csv(run / s / "q1" / "tables" / "lag_insert.tsv", separator="\t")["observed"][0])})
        g = x["gates"]
        syn = g["synthetic"]["metrics"]["rates"]
        hifi_rates = [r for r in syn if float(r) <= 0.01]            # gate 1 judges the HiFi error rates only
        d["gates"].append({"library": short,
                           "synthetic_recall_min": min(syn[r][k]["recall"] for r in hifi_rates for k in ("MAS", "TSO", "R1")),
                           "synthetic_precision_min": min(syn[r][k]["precision"] for r in hifi_rates for k in ("MAS", "TSO", "R1")),
                           "synthetic_error_rates": ", ".join(hifi_rates),
                           "tso_toward_lower_adapter": x["q2"]["tso_toward_lower"],
                           "adapter_index_agreement": g["adapter_index"]["metrics"]["agreement"],
                           "lima_class_pair_agreement": g["lima"]["metrics"]["class_pair_agreement"],
                           "fltnc_in_lima_pass": g["lima"]["metrics"]["fltnc_in_lima_pass"],
                           "skera_agreement": g["skera"]["metrics"]["agreement"],
                           "all_hard_gates_passed": all(g[k]["passed"] for k in ("synthetic", "adapter_index", "lima"))})
    # mirror junction mechanism, BD144_ND
    nd = raw["ND"]
    b = dict(nd["bases"].iter_rows())
    d["tso_tso_base_share_ND"] = b["tso_tso"] / sum(b.values())
    d["mirror_junction_tso_tso_share_ND"] = nd["q2"]["mirror_junction_tso_tso_share"]
    j = nd["junction"]
    d["junction_letters"] = metrics.junction_table(nd["segs"])
    d["junction_composition"] = metrics.junction_composition(nd["segs"], nd["reads"])
    d["mirror_rule"] = metrics.mirror_size_rule(nd["segs"], nd["reads"])
    single = metrics.mirror_size_reads(nd["segs"], nd["reads"])
    d["mirror_rule_overall"] = float((single["molecules"] == single["predicted"]).mean())
    d["mirror_rule_reads"] = single.height
    # why junctions pile up at H: ordinary mirrors need 2 x (8 - i) joins, double mirrors (A..X X' X..Q) the same number at every letter
    jl = j.with_columns((pl.col("end") - pl.col("start")).alias("mol_len"))
    per_read = jl.group_by("zm").agg(pl.len().alias("nj"), (pl.col("end_class") == "tso_tso").sum().alias("tt"), (pl.col("end_class") == "r1_r1").sum().alias("rr"),
                                     pl.col("letter").filter(pl.col("end_class") == "tso_tso").first().alias("tt_letter"),
                                     pl.col("letter").filter(pl.col("end_class") == "r1_r1").first().alias("rr_letter"),
                                     pl.col("mol_len").filter(pl.col("end_class") == "tso_tso").first().alias("tt_len"))
    lens = nd["reads"].select("zm", "read_len", "n_mas")
    ordinary = (per_read.filter((pl.col("nj") == 1) & (pl.col("tt") == 1)).join(lens, on="zm")
                .group_by("tt_letter").agg(pl.len().alias("ordinary_mirrors"), pl.col("read_len").median().alias("ordinary_median_read_len"),
                                           pl.col("tt_len").median().alias("tso_tso_median_len")))
    double = (per_read.filter((pl.col("nj") == 2) & (pl.col("tt") == 1) & (pl.col("rr") == 1) & (pl.col("tt_letter") == pl.col("rr_letter"))).join(lens, on="zm")
              .group_by("tt_letter").agg(pl.len().alias("double_mirrors"), (pl.col("n_mas") - 1).median().alias("double_median_molecules"),
                                         pl.col("read_len").median().alias("double_median_read_len")))
    d["junction_control"] = ordinary.join(double, on="tt_letter", how="left").rename({"tt_letter": "letter"}).sort("letter")
    # ND against kin: where the molecules go
    yv = {r["library"]: r for r in d["yield"]}
    pat = {(r["library"], r["pattern"]): r for r in d["pattern"]}
    capacity = pat[("ND", "mirror")]["share"] * (pat[("ND", "complete")]["proper_per_read"] - pat[("ND", "mirror")]["proper_per_read"])
    d["decomposition"] = {"ours_gap": yv["kin"]["proper_per_hifi_read"] - yv["ND"]["proper_per_hifi_read"], "mirror_capacity": capacity,
                          "mapped_gap": yv["kin"]["mapped_fltnc_per_hifi_zmw"] - yv["ND"]["mapped_fltnc_per_hifi_zmw"],
                          "skera_extra": yv["ND"]["not_recovered_by_skera_per_hifi_zmw"] - yv["kin"]["not_recovered_by_skera_per_hifi_zmw"]}
    # Q1 curves
    d["partner_curves"], d["lag"], d["selection"] = {}, {}, {}
    for s, short, _ in SAMPLES:
        t = run / s / "q1" / "tables"
        d["partner_curves"][short] = pl.read_csv(t / "partner_curve_insert.tsv", separator="\t")
        d["lag"][short] = pl.read_csv(t / "lag_insert.tsv", separator="\t")
        d["selection"][short] = pl.read_csv(t / "selection_T.tsv", separator="\t")
    # What S-reads keep of mirror arrays (section 6): BD144 against the full-read calls, then the public ArgenTag S-reads
    a, c = inputs.argentag, inputs.calibration
    sk = {short: pl.read_parquet(run / s / "segments.parquet", columns=["zm", "dl", "dr", "qs", "qe"]).cast(pl.Int64) for s, short, _ in SAMPLES}
    sk[ARG] = pl.read_parquet(a / "argentag_sread_coords.parquet").cast(pl.Int64)
    hifi = {short: raw[short]["reads"].height for _, short, _ in SAMPLES} | {ARG: ARG_HIFI}
    tests = []
    for lib in ORDER3 + [ARG]:
        z, n = unwritten_head(sk[lib]), hifi[lib]
        row = {"library": lib, "hifi_zmws": n, "sreads": sk[lib].height, "q_not_a": z["q_not_a"].sum() / n, "flag": z["flag"].sum() / n}
        if lib != ARG:
            t = raw[lib]["reads"].select(pl.col("zm").cast(pl.Int64), "pattern").join(z, on="zm", how="left").with_columns(
                pl.col("q_not_a").fill_null(False), pl.col("flag").fill_null(False))
            mirror = t.filter(pl.col("pattern") == "mirror")
            for thr in (1.0, 1.5):
                f_thr = t.filter(pl.col("q_not_a") & pl.col("fwd") & (pl.col("head_mol") >= thr))
                row |= {f"non_mirror_flagged_{thr}": (f_thr["pattern"] != "mirror").sum() / n, f"sensitivity_{thr}": (f_thr["pattern"] == "mirror").sum() / mirror.height}
            qa, fl = t.filter(pl.col("q_not_a")), t.filter(pl.col("flag"))
            row |= {f"flag_non_mirror_{k}": v for k, v in fl.filter(pl.col("pattern") != "mirror").group_by("pattern").len().sort("pattern").iter_rows()}
            row |= {"true_mirror": mirror.height / n, "q_not_a_ppv": (qa["pattern"] == "mirror").mean(),
                    "flag_ppv": (fl["pattern"] == "mirror").mean(), "flag_sensitivity": mirror["flag"].mean(),
                    "flag_non_mirror": (fl["pattern"] != "mirror").sum() / n}
        tests.append(row)
    d["sread_tests"] = pl.DataFrame(tests)
    # what skera writes from ND mirror reads, molecule by molecule (the mirror_arms analysis)
    ma = qc / "skera_mirror_arms"
    d["mirror_arms"] = {lab: {"summary": load_json(ma / f"{lab}_summary.json"),
                              "molecules": pl.read_csv(ma / f"{lab}_molecules.tsv", separator="\t"),
                              "reads": pl.read_csv(ma / f"{lab}_reads.tsv", separator="\t"),
                              "letters": pl.read_csv(ma / f"{lab}_reads_by_letter.tsv", separator="\t")}
                        for lab in ("nd_hash_skera26", "nd_cal_skera26", "nd_cal_skera12")}
    d["mirror_found"] = pl.read_csv(ma / "nd_cal_skera12_found_adapters.tsv", separator="\t")
    # where the 8 slots of a HiFi read go, per library (the molecule_budget analysis)
    d["budget"] = {short: load_json(qc / "molecule_budget" / f"{s}_budget.json") for s, short, _ in SAMPLES}
    # the reverse error: reads skera passes that our segmentation does not call complete (an adapter skera finds and we miss)
    d["our_misses"] = {}
    for _, short, _ in SAMPLES:
        r = raw[short]["reads"].filter(pl.col("skera_passing"))
        f = raw[short]["fate"].join(r.select("zm", "pattern"), on="zm")
        missed = r.filter(pl.col("primary_pattern").is_in(["truncated", "skipped_slot"]))["zm"]    # our chain lacks an adapter of a full array
        d["our_misses"][short] = {"passing_reads": r.height, "missed_adapter_share": missed.len() / r.height,
                                  "skipped_slot_share": float((r["primary_pattern"] == "skipped_slot").mean()),
                                  "sread_lima_pass_missed": float(f.filter(pl.col("zm").is_in(missed))["lima_pass"].mean()),
                                  "sread_lima_pass_complete": float(f.filter(pl.col("pattern") == "complete")["lima_pass"].mean())}
    # what each skera false cut does to the two S-reads around it (the false_cut_anatomy analysis)
    d["false_cut_anatomy"] = {short: load_json(qc / "false_cut_anatomy" / f"{s}_false_cuts.json") for s, short, _ in SAMPLES}
    d["budget_pattern"] = pl.concat([pl.read_csv(qc / "molecule_budget" / f"{s}_budget_by_pattern.tsv", separator="\t").with_columns(pl.lit(short).alias("library"))
                                     for s, short, _ in SAMPLES])
    d["bench"] = pl.read_csv(qc / "mirror_benchmark" / "mirror_benchmark_summary.tsv", separator="\t")
    d["bench_classes"] = pl.read_csv(qc / "mirror_benchmark" / "mirror_benchmark.tsv", separator="\t")
    d["downstream"] = pl.read_csv(qc / "downstream_test" / "downstream_summary.tsv", separator="\t")
    d["discordant"] = load_json(qc / "discordant_reads" / "discordant_summary.json")
    d["false_cuts"] = pl.read_csv(qc / "skera_false_cuts" / "skera_false_cuts_summary.tsv", separator="\t").row(0, named=True)
    d["false_cuts_by_distance"] = pl.read_csv(qc / "skera_false_cuts" / "skera_false_cuts_by_distance.tsv", separator="\t")
    d["head_versions"] = pl.read_csv(c / "head_test_skera_versions.tsv", separator="\t")
    d["tso_validation"] = pl.read_csv(c / "tso_end_validation.tsv", separator="\t")
    # ArgenTag parent reads (skera's non_passing.bam, shared by ArgenTag): our stage 02 on every read skera did not pass
    d["arg_full"] = pl.read_csv(a / "full_read_summary.tsv", separator="\t").row(0, named=True)
    d["arg_junctions"] = pl.read_csv(a / "full_read_junctions.tsv", separator="\t")
    d["arg_validation"] = pl.read_csv(a / "flag_validation.tsv", separator="\t").row(0, named=True)
    scan = (a / "independent_scan.txt").read_text()
    d["arg_indep"] = {"agree": int(re.search(r"Same call \(mirror or not\): (\d+)", scan).group(1)),
                      "sample": int(re.search(r"Sample size: (\d+)", scan).group(1))}
    # BD144 equivalents of the parent-read readouts, per library
    d["bd_full"] = {}
    for _, short, _ in SAMPLES:
        jl = raw[short]["junction"]
        one = metrics.mirror_size_reads(raw[short]["segs"], raw[short]["reads"])
        pr = {r["pattern"]: r for r in d["pattern"] if r["library"] == short}
        d["bd_full"][short] = {
            "junctions": jl.height, "junctions_tso_tso_share": float((jl["end_class"] == "tso_tso").mean()) if jl.height else float("nan"),
            "junctions_at_A_tso_tso": jl.filter((pl.col("letter") == "A") & (pl.col("end_class") == "tso_tso")).height,
            "single_tt_matching_rule": float((one["molecules"] == one["predicted"]).mean()) if one.height else float("nan"),
            "usable_per_mirror_read": pr["mirror"]["proper_per_read"],
            "mirror_shortfall": pr["mirror"]["share"] * (pr["complete"]["proper_per_read"] - pr["mirror"]["proper_per_read"])}
    d["bd_letters"] = (raw["ND"]["junction"].filter(pl.col("end_class") == "tso_tso").group_by("letter").len()
                       .rename({"len": "junctions"}).sort("letter"))
    return d


def run_commits(markers) -> list[str]:
    """The commits recorded in the run's .ok markers (date, time, git HEAD, ...), short form, sorted."""
    return sorted({m.read_text().split()[2][:7] for m in markers})


def write_tables(d: dict, tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(list(d["lib"].values())).write_csv(tables / "S1_libraries.tsv", separator="\t")
    pl.DataFrame(d["gates"]).write_csv(tables / "S3_validation_gates.tsv", separator="\t")
    pl.DataFrame(d["pattern"]).sort("library", "pattern").write_csv(tables / "fig2a_chain_patterns.tsv", separator="\t")
    pl.DataFrame(d["artifacts"]).write_csv(tables / "fig2b_artifact_rates.tsv", separator="\t")
    d["junction_letters"].write_csv(tables / "fig4b_mirror_junction_letters_ND.tsv", separator="\t")
    d["mirror_rule"].write_csv(tables / "fig4c_mirror_array_size_rule_ND.tsv", separator="\t")
    d["junction_control"].write_csv(tables / "fig4b_junction_letters_ordinary_and_double_mirrors_ND.tsv", separator="\t")
    d["bench_classes"].write_csv(tables / "S4_chain_classification_benchmark.tsv", separator="\t")
    d["bench"].write_csv(tables / "S4_chain_classification_benchmark_summary.tsv", separator="\t")
    d["downstream"].write_csv(tables / "downstream_survival_ND.tsv", separator="\t")
    pl.DataFrame([d["discordant"]]).write_csv(tables / "skera_disagreement_ND.tsv", separator="\t")
    pl.DataFrame([d["false_cuts"]]).write_csv(tables / "skera_false_cuts_ND.tsv", separator="\t")
    d["false_cuts_by_distance"].write_csv(tables / "skera_false_cuts_by_distance_ND.tsv", separator="\t")
    d["junction_composition"].write_csv(tables / "S5_mirror_junction_composition_ND.tsv", separator="\t")
    pl.DataFrame(d["yield"]).write_csv(tables / "fig5a_yield.tsv", separator="\t")
    pl.DataFrame(d["skera_by_pattern"]).sort("library", "pattern", "skera_passing").write_csv(tables / "fig5b_skera_fate_by_pattern.tsv", separator="\t")
    pl.DataFrame([d["decomposition"]]).write_csv(tables / "fig5c_ND_gap_decomposition.tsv", separator="\t")
    pl.DataFrame([{"library": k, **v} for k, v in d["budget"].items()]).write_csv(tables / "fig6_molecule_budget.tsv", separator="\t")
    pl.DataFrame([{"library": k, "skera_false_cut_reads_share": d["false_cut_anatomy"][k]["reads_with_false_cut"], **v} for k, v in d["our_misses"].items()]) \
        .write_csv(tables / "S2_segmentation_vs_skera_errors.tsv", separator="\t")
    pl.DataFrame([{"library": k, **{x: y for x, y in v.items() if not isinstance(y, dict)}, **{f"fate_{x}": y for x, y in v["fates"].items()},
                   **{f"proper_sread_{x}": y for x, y in v["baseline_proper_sreads"].items()}} for k, v in d["false_cut_anatomy"].items()]) \
        .write_csv(tables / "fig1_skera_false_cut_anatomy.tsv", separator="\t")
    d["budget_pattern"].write_csv(tables / "fig6_molecule_budget_by_pattern.tsv", separator="\t")
    pl.DataFrame(d["q1"]).write_csv(tables / "S6_concatenation_bias.tsv", separator="\t")
    pl.concat([v.with_columns(pl.lit(k).alias("library")) for k, v in d["partner_curves"].items()]).write_csv(tables / "fig7_partner_curves.tsv", separator="\t")
    pl.concat([v.with_columns(pl.lit(k).alias("library")) for k, v in d["lag"].items()]).write_csv(tables / "fig8a_lag_profiles.tsv", separator="\t")
    pl.concat([v.with_columns(pl.lit(k).alias("library")) for k, v in d["selection"].items()]).write_csv(tables / "fig8b_length_selection.tsv", separator="\t")
    d["sread_tests"].write_csv(tables / "S8_sread_tests.tsv", separator="\t")
    d["head_versions"].write_csv(tables / "S9_skera_version_check.tsv", separator="\t")
    pl.DataFrame([{"library": ARG, **d["arg_full"]}]).write_csv(tables / "S7_argentag_parent_reads.tsv", separator="\t")
    pl.DataFrame([{"library": k, **v} for k, v in d["bd_full"].items()]).write_csv(tables / "S7_bd144_parent_read_equivalents.tsv", separator="\t")
    pl.concat([d["bd_letters"].with_columns(pl.col("junctions").cast(pl.Int64), pl.lit("ND").alias("library")),
               d["arg_junctions"].filter(pl.col("molecule") == "TSO-TSO").select("letter", pl.col("junctions").cast(pl.Int64))
               .with_columns(pl.lit(ARG).alias("library"))]
              ).write_csv(tables / "fig9c_tso_tso_junction_letters.tsv", separator="\t")
    pl.DataFrame([d["arg_validation"]]).write_csv(tables / "S8_argentag_flag_validation.tsv", separator="\t")
    pl.concat([v["molecules"].with_columns(pl.lit(k).alias("set")) for k, v in d["mirror_arms"].items()]).write_csv(
        tables / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t")
    pl.concat([v["reads"].with_columns(pl.lit(k).alias("set")) for k, v in d["mirror_arms"].items()]).write_csv(
        tables / "S10_mirror_reads_by_skera_output_ND.tsv", separator="\t")
    d["mirror_arms"]["nd_hash_skera26"]["letters"].write_csv(tables / "S10_mirror_reads_by_junction_letter_ND.tsv", separator="\t")
    d["mirror_found"].write_csv(tables / "S10_skera120_found_adapters_mirror_ND.tsv", separator="\t")
    d["tso_validation"].write_csv(tables / "tso_both_ends_validation_ND.tsv", separator="\t")


# Supplementary Table 4: the synthetic read classes, by the prefix of their truth label, and the call each should get
# (None: anything but mirror).
BENCH_CLASSES = [("complete", "complete", "complete"), ("tso_tso_mirror", "tso_tso_mirror_", "mirror"),
                 ("r1_r1_mirror", "r1_r1_mirror_", "mirror"), ("truncated_no_a", "truncated_no_A", "truncated"),
                 ("truncated_no_q", "truncated_no_Q", "truncated"), ("adapter_destroyed", "adapter_destroyed", "skipped_slot"),
                 ("decoy_motif", "decoy_motif", "complete"), ("opposite_join", "opposite_join", None)]
DOWNSTREAM_GROUPS = {"complete reads": "complete", "mirror reads, not written by skera": "unwritten",
                     "mirror reads, written by skera": "written"}


def numbers(d: dict, fraction: float, fig3_zmws: list[int], markers=()) -> dict:
    """Every number the Note's text quotes and every value its tables derive, by key, unformatted (floats as computed,
    counts as ints). Library-wise keys start with nd, hm, kin or arg (ArgenTag)."""
    lib, fca, bud, dc = d["lib"], d["false_cut_anatomy"], d["budget"], d["decomposition"]
    yv = {r["library"]: r for r in d["yield"]}
    q1 = {r["library"]: r for r in d["q1"]}
    gates = {r["library"]: r for r in d["gates"]}
    p = {(r["library"], r["pattern"]): r for r in d["pattern"]}
    a = {(r["library"], r["class"]): r["per_1000_molecules"] for r in d["artifacts"]}
    st = {r["library"]: r for r in d["sread_tests"].iter_rows(named=True)}
    bench = {r["error_rate"]: r for r in d["bench"].iter_rows(named=True)}
    dsv = {r["group"]: r for r in d["downstream"].iter_rows(named=True)}
    fc, af, av, om = d["false_cuts"], d["arg_full"], d["arg_validation"], d["our_misses"]
    comp = {r["junction_molecules"]: r for r in d["junction_composition"].iter_rows(named=True)}
    hv = {("1.2.0" if "1.2.0" in r["skera"] else "26.1.0"): r for r in d["head_versions"].iter_rows(named=True)}
    jc = {r["letter"]: r for r in d["junction_control"].iter_rows(named=True)}
    st[ARG] = {**st[ARG], "true_mirror": af["mirror_share"], "q_not_a_ppv": av["q_not_a_ppv"], "flag_ppv": av["flag_ppv"],
               "flag_sensitivity": av["flag_sensitivity"], "flag_non_mirror": av["flag_non_mirror_share"]}
    n = {"fraction": fraction, "head_mol": HEAD_MOL, "arg_hifi_reads": ARG_HIFI, "fig3_zmws": fig3_zmws}

    # libraries and chain patterns
    n |= {"hifi_analysed_min": min(x["hifi_analysed"] for x in lib.values()), "hifi_analysed_max": max(x["hifi_analysed"] for x in lib.values()),
          "nd_mirror_by_tenth_min": lib["ND"]["mirror_by_tenth_min"], "nd_mirror_by_tenth_max": lib["ND"]["mirror_by_tenth_max"]}
    for (library, pattern), r in sorted(p.items()):
        for f in ("share", "proper_per_read", "skera_sreads_per_read", "skera_pass_share", "median_read_len"):
            n[f"{KEY[library]}_{pattern}_{f}"] = r[f]
    n |= {"nd_truncated_non_passing_share": 1 - p[("ND", "truncated")]["skera_pass_share"],
          "nd_mirror_molecules_fewer": p[("ND", "complete")]["proper_per_read"] - p[("ND", "mirror")]["proper_per_read"]}
    for library in ORDER3:
        n[f"{KEY[library]}_tso_tso_per_1000"] = a[(library, "TSO-TSO")]
        n[f"{KEY[library]}_r1_r1_per_1000"] = a[(library, "R1-R1")]
    n |= {"tso_tso_ratio_nd_hm": a[("ND", "TSO-TSO")] / a[("HM", "TSO-TSO")], "tso_tso_ratio_nd_kin": a[("ND", "TSO-TSO")] / a[("kin", "TSO-TSO")],
          "nd_mirror_junction_tso_tso_share": d["mirror_junction_tso_tso_share_ND"], "nd_tso_tso_base_share": d["tso_tso_base_share_ND"],
          "nd_tso_tso_only_skera_pass_share": comp["TSO-TSO only"]["skera_pass_share"]}

    # validation and the classification benchmark
    rates = sorted(bench)
    shares = {}
    for name, prefix, want in BENCH_CLASSES:
        for r in rates:
            g = d["bench_classes"].filter((pl.col("error_rate") == r) & pl.col("truth").str.starts_with(prefix))
            ok = (g["n"].sum() - g["called_mirror"].sum()) if want is None else g[f"called_{want}"].sum()
            shares[(name, r)] = ok / g["n"].sum()
    n |= {"synthetic_error_rates": gates["ND"]["synthetic_error_rates"],
          "tso_toward_lower_adapter_max": max(g["tso_toward_lower_adapter"] for g in gates.values()),
          "bench_reads_per_class": int(d["bench_classes"]["n"][0]),
          "bench_hifi_class_min": min(v for (_, r), v in shares.items() if r <= 0.01)}
    n |= {f"bench_{name}_{r:g}": v for (name, r), v in shares.items()}
    n |= {f"bench_mirror_precision_{r:g}": bench[r]["mirror_precision"] for r in rates}

    # skera false cuts: their anatomy in the hash samples, and the calibration set
    n |= {"false_cut_reads_share_min": min(v["reads_with_false_cut"] for v in fca.values()),
          "false_cut_reads_share_max": max(v["reads_with_false_cut"] for v in fca.values()),
          "nd_false_cut_missed_is_d": fca["ND"]["missed_is_D"],
          "nd_false_cut_missed_same_letter": fca["ND"]["missed_same_letter_as_skera_call"],
          "nd_false_cut_median_cut_to_missed_bp": fca["ND"]["median_cut_to_missed_adapter_bp"],
          "nd_false_cut_fused_far_end_at_real_adapter": fca["ND"]["fused_far_end_at_real_adapter"],
          "false_cut_frag_lima_max": max(v["fates"]["frag_lima"] for v in fca.values()),
          "false_cut_fused_lima_min": min(v["fates"]["fused_lima"] for v in fca.values()),
          "false_cut_fused_lima_max": max(v["fates"]["fused_lima"] for v in fca.values()),
          "false_cut_baseline_lima_pass_min": min(v["baseline_proper_sreads"]["lima_pass"] for v in fca.values()),
          "kin_false_cut_fused_lima": fca["kin"]["fates"]["fused_lima"], "kin_false_cut_frag_lima": fca["kin"]["fates"]["frag_lima"],
          "kin_false_cut_molecules_per_hifi_read": 2 * fca["kin"]["false_cuts"] / bud["kin"]["hifi_reads"],
          "cal_false_cuts_closest_to_d_3_to_5_edits_share": fc["cuts_closest_to_D_3_to_5_edits"] / fc["cuts"],
          "cal_false_cuts_within_2_edits_share": fc["cuts_within_2_edits_of_any_adapter"] / fc["cuts"],
          "cal_false_cuts_ctcctcc_share": fc["cuts_with_ctcctcc_repeat"] / fc["cuts"],
          "cal_skipped_adapters_le1_edit_with_primer_share": fc["inside_le1_edit_with_primer"] / fc["our_adapters_inside_sreads"]}
    for k in ("missed_adapter_share", "skipped_slot_share", "sread_lima_pass_missed", "sread_lima_pass_complete"):
        n[f"{k}_min"], n[f"{k}_max"] = min(v[k] for v in om.values()), max(v[k] for v in om.values())

    # mirror arrays (section 3)
    sel_nd = d["selection"]["ND"]

    def hifi_share(length):
        row = sel_nd.filter((pl.col("T_bin_start") <= length) & (pl.col("T_bin_end") > length))
        return float(row["w_empirical_hifi_share"][0])

    step = [jc["H"]["ordinary_mirrors"] / jc["G"]["ordinary_mirrors"], jc["G"]["ordinary_mirrors"] / jc["F"]["ordinary_mirrors"]]
    other_tt = [jc[x]["tso_tso_median_len"] for x in "BCDEFGH"]
    n |= {"mirror_rule_overall": d["mirror_rule_overall"], "mirror_rule_reads": d["mirror_rule_reads"],
          "mirror_fold_per_letter": sum(step) / 2,
          "double_mirror_median_molecules_h": jc["H"]["double_median_molecules"],
          "double_mirror_median_read_len_min": min(r["double_median_read_len"] for r in jc.values() if r["double_mirrors"]),
          "double_mirror_median_read_len_max": max(r["double_median_read_len"] for r in jc.values() if r["double_mirrors"])}
    n |= {f"double_mirrors_{x.lower()}": jc[x]["double_mirrors"] for x in "BCDEFGH"}
    n |= {f"ordinary_mirrors_{x.lower()}": jc[x]["ordinary_mirrors"] for x in "BCDEFGH"}
    n |= {"double_mirrors_min": min(jc[x]["double_mirrors"] for x in "BCDEFGH"), "double_mirrors_max": max(jc[x]["double_mirrors"] for x in "BCDEFGH"),
          "hifi_yield_ratio_h_b": hifi_share(jc["H"]["ordinary_median_read_len"]) / hifi_share(jc["B"]["ordinary_median_read_len"]),
          "ordinary_mirror_median_read_len_h": jc["H"]["ordinary_median_read_len"], "ordinary_mirror_median_read_len_b": jc["B"]["ordinary_median_read_len"],
          "tso_tso_median_len_q": jc["Q"]["tso_tso_median_len"], "tso_tso_median_len_other_min": min(other_tt),
          "tso_tso_median_len_other_max": max(other_tt)}

    # yield (section 4)
    for library in ORDER3:
        n[f"{KEY[library]}_mapped_fltnc_per_hifi_zmw"] = yv[library]["mapped_fltnc_per_hifi_zmw"]
        n[f"{KEY[library]}_proper_per_hifi_read"] = yv[library]["proper_per_hifi_read"]
    n |= {"loss_ours": dc["ours_gap"] / yv["kin"]["proper_per_hifi_read"], "loss_mapped": dc["mapped_gap"] / yv["kin"]["mapped_fltnc_per_hifi_zmw"],
          "gap_ours": dc["ours_gap"], "gap_mirror_capacity": dc["mirror_capacity"], "gap_mapped": dc["mapped_gap"], "gap_skera_extra": dc["skera_extra"]}
    for group, r in dsv.items():
        for f in ("molecules", "lima_5p3p", "refine", "mapped"):
            n[f"downstream_{DOWNSTREAM_GROUPS[group]}_{f}"] = r[f]
    for library in ORDER3:
        b, k = bud[library], KEY[library]
        art = b["artifact_tso_tso"] + b["artifact_r1_r1"] + b["artifact_one_primer_or_none"] + b["artifact_fused"]
        dn = b["lost_lima"] + b["lost_refine"] + b["lost_mapping"]
        tot = b["unfilled_slots"] + art + dn + b["not_written_cleanly"]
        n |= {f"budget_{k}_{f}": b[f] for f in ("hifi_reads", "unfilled_slots", "not_written_cleanly", "kept_standard", "other_mapped")}
        n |= {f"budget_{k}_artifact": art, f"budget_{k}_downstream": dn, f"budget_{k}_total": tot,
              f"budget_{k}_unwritten_share": b["not_written_cleanly"] / tot, f"budget_{k}_kept_full": b["kept_standard"] + b["not_written_cleanly"]}
    n["budget_other_mapped_max"] = max(bud[x]["other_mapped"] for x in ORDER3)

    # concatenation bias (section 5)
    q1_lo, q1_hi = sorted((min(r["diff_vs_P_bp"] for r in q1.values()), max(r["diff_vs_P_bp"] for r in q1.values())))
    ipw_e, ipw_m = [r["ipw_empirical_diff"] for r in q1.values()], [r["ipw_model_diff"] for r in q1.values()]
    n |= {"partners_shorter_bp_min": -q1_hi, "partners_shorter_bp_max": -q1_lo,
          "diff_vs_mt_bp_abs_max": max(abs(r["diff_vs_MT_bp"]) for r in q1.values()),
          "ipw_empirical_diff_min": min(ipw_e), "ipw_empirical_diff_max": max(ipw_e),
          "ipw_model_diff_min": min(ipw_m), "ipw_model_diff_max": max(ipw_m)}
    for library in ORDER3:
        n |= {f"{KEY[library]}_d_residual{s}": q1[library][f"D_residual{s}"] for s in ("", "_lo", "_hi")}
    n["lag1_mean"] = sum(r["lag1"] for r in q1.values()) / 3

    # ArgenTag parent reads (section 6)
    n |= {f"arg_{f}": af[f] for f in ("mirror_share", "tso_tso_per_1000", "r1_r1_per_1000", "junctions_tso_tso_share",
                                       "single_tt_matching_rule", "proper_share_of_molecules", "usable_per_mirror_read",
                                       "sreads_per_passing_read", "sreads_per_mirror_read")}
    n |= {f"arg_{f}": int(af[f]) for f in ("non_passing_reads", "hifi_zmws", "mirror_junctions", "junctions_at_A_tso_tso")}
    n |= {"arg_mirror_shortfall_ours": af["mirror_shortfall_ours_per_hifi_zmw"],
          "arg_mirror_shortfall_sreads": af["mirror_shortfall_sreads_per_hifi_zmw"],
          "arg_mirror_shortfall_ours_share": af["mirror_shortfall_ours_per_hifi_zmw"] / af["sreads_per_passing_read"],
          "arg_mirror_shortfall_sreads_share": af["mirror_shortfall_sreads_per_hifi_zmw"] / af["sreads_per_passing_read"],
          "arg_independent_agree": d["arg_indep"]["agree"], "arg_independent_sample": d["arg_indep"]["sample"],
          "arg_independent_agree_share": d["arg_indep"]["agree"] / d["arg_indep"]["sample"], "arg_sreads": st[ARG]["sreads"]}

    # the S-read tests against the parent reads (Supplementary Table 8; ArgenTag's truth from its parent reads)
    for library in ORDER3 + [ARG]:
        n |= {f"{KEY[library]}_{f}": st[library][f] for f in ("true_mirror", "q_not_a", "q_not_a_ppv", "flag", "flag_ppv",
                                                               "flag_sensitivity", "flag_non_mirror")}
    n |= {"bd144_flag_non_mirror_min": min(st[x]["flag_non_mirror"] for x in ORDER3),
          "bd144_flag_non_mirror_max": max(st[x]["flag_non_mirror"] for x in ORDER3),
          "non_mirror_flagged_1_0_max": max(st[x]["non_mirror_flagged_1.0"] for x in ORDER3),
          "non_mirror_flagged_1_5_max": max(st[x]["non_mirror_flagged_1.5"] for x in ORDER3),
          "nd_sensitivity_1_0": st["ND"]["sensitivity_1.0"], "nd_sensitivity_1_5": st["ND"]["sensitivity_1.5"],
          "skera120_flagged": hv["1.2.0"]["flagged"], "skera2610_flagged": hv["26.1.0"]["flagged"],
          "skera120_ppv": hv["1.2.0"]["ppv"], "skera2610_ppv": hv["26.1.0"]["ppv"]}
    tv = d["tso_validation"]
    tv_n = int(tv["sreads_checked"][0])
    tv_bg = int(tv.filter(~pl.col("touches_tso_tso") & (pl.col("primary_pattern") == "complete"))["len"].sum())
    n |= {"tso_both_ends_background": tv_bg, "tso_both_ends_checked": tv_n, "tso_both_ends_background_share": tv_bg / tv_n}

    # what skera writes from ND mirror reads (section 6, Supplementary Table 10)
    mh = d["mirror_arms"]["nd_hash_skera26"]
    ms, s12 = mh["summary"], d["mirror_arms"]["nd_cal_skera12"]["summary"]
    n |= {"mirror_arms_non_passing_share": ms["mirror_reads_non_passing_share"], "mirror_arms_single_tso_tso_reads": ms["single_tso_tso_mirror_reads"],
          "mirror_arms_clean_share_mirror": ms["clean_share_mirror_reads"], "mirror_arms_clean_share_complete": ms["clean_share_complete_arrays"],
          "mirror_arms_complete_reads_baseline": ms["complete_reads_baseline"]}
    for r in mh["molecules"].iter_rows(named=True):
        n[f"mirror_{r['region']}_{r['class'].lower()}_share"] = r["share"]
    for (region,), g in mh["molecules"].group_by("region", maintain_order=True):
        n[f"mirror_{region}_molecules"] = int(g["n"].sum())
    n |= {f"mirror_reads_{r['read_class'].lower()}": r["share"] for r in mh["reads"].iter_rows(named=True)}
    let = mh["letters"]
    n["mirror_reads_no_sread_at_q_share"] = (let.filter((pl.col("letter") == "Q") & (pl.col("read_class") == "NO_SREAD"))["n"].sum()
                                             / let.filter(pl.col("read_class") == "NO_SREAD")["n"].sum())
    fa = {r["where"]: r["share_found_by_skera"] for r in d["mirror_found"].iter_rows(named=True)}
    c12, c26 = (d["mirror_arms"][k]["molecules"] for k in ("nd_cal_skera12", "nd_cal_skera26"))
    n |= {"skera120_found_after_arm": fa["after_arm"], "skera120_found_before_arm": fa["before_arm"],
          "skera120_ligations_same_letter": s12["ligations_same_letter"], "skera120_ligations_total": s12["ligations_total"],
          "skera_versions_share_diff_max": c12.join(c26, on=["region", "class"]).select((pl.col("share") - pl.col("share_right")).abs().max()).item()}

    # software and commits
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    n |= {"kinnex_qc_version": im.version("kinnex-qc"), "kinnex_qc_commit": head.stdout.strip() if head.returncode == 0 else "unknown",
          "run_commits": run_commits(markers), "version_python": platform.python_version()}
    n |= {f"version_{k}": im.version(k) for k in ("polars", "pysam", "edlib", "numpy", "scipy", "plotly", "kaleido")}
    return n


def example_reads(nd: Path) -> list[dict]:
    """Supplementary Fig. 3: the first complete array of 9 to 12 kb and the first mirror array under 16 kb of each chain in
    FIG3_CHAINS (3, 5 and 11 molecules) among the HiFi reads of BD144_ND, by ZMW; their tracks in that order, as JSON-ready
    dicts (each hit as a dict of kinnex_qc.scan.Hit's fields, S-reads as [qs, qe])."""
    summary = pl.read_parquet(nd / "read_summary.parquet", columns=["zm", "source", "read_len", "chain", "primary_pattern", "skera_passing"])
    segs = pl.read_parquet(nd / "annot_segments.parquet", columns=["zm", "source", "start", "end", "pos", "end_class", "internal", "polyT_len", "tso_side"])
    skera = pl.read_parquet(nd / "segments.parquet", columns=["zm", "qs", "qe"])
    h = summary.filter(pl.col("source") == "hifi").sort("zm")
    complete = h.filter((pl.col("primary_pattern") == "complete") & pl.col("read_len").is_between(9000, 12000))["zm"].head(1).to_list()
    mirrors = []
    for chain in FIG3_CHAINS:
        mirrors += h.filter((pl.col("chain") == chain) & (pl.col("read_len") < 16000))["zm"].head(1).to_list()
    zms = complete + mirrors
    feats = pl.scan_parquet(nd / "read_features.parquet").filter(pl.col("zm").is_in(zms)).collect()
    order = {z: i for i, z in enumerate(zms)}
    tr = sorted(tracks(zms, summary, segs, feats, skera), key=lambda t: order[t["zm"]])
    return [{**t, "features": [asdict(x) for x in t["features"]], "skera": [list(x) for x in t["skera"]]} for t in tr]


def write_report(inputs: Inputs, cells: dict[str, str], fraction: float, report: Path) -> None:
    """tables/, numbers.json and fig3_reads.json into report."""
    d = compute(inputs, cells)
    write_tables(d, report / "tables")
    reads = example_reads(inputs.run / "BD144_ND")
    (report / "fig3_reads.json").write_text(json.dumps(reads) + "\n")
    n = numbers(d, fraction, [r["zm"] for r in reads], inputs.markers)
    (report / "numbers.json").write_text(json.dumps(n, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="study/study.toml", help="the study's TOML, with the .local.toml beside it laid over it")
    ap.add_argument("--out-root", help="the run's out root (default: out_root of the config)")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    out_root = Path(args.out_root or config["out_root"])
    report = out_dir(out_root, "report", config["fraction"])       # <run>/report
    run = report.parent
    inputs = Inputs(run=run, analyses=run / "analyses", calibration=out_root / "calibration", argentag=out_root / "argentag",
                    markers=tuple(sorted((out_root / "logs").glob("*/*.ok"))))
    cells = {s: x.cell for s, x in load_samples(config["samples"]).items()}
    write_report(inputs, cells, config["fraction"], report)


if __name__ == "__main__":
    main()
