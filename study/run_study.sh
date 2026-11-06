#!/bin/bash
# The whole study in one command, in five phases. In phases 0 and 1 the samples run in parallel, each sample's steps in
# order as one chain in its own process group (setsid), so stopping a chain reaches its running step and that step's
# workers.
#   phase 0, per sample in parallel: gate_synthetic (reads only the sample's segmented BAM)
#   phase 1, per sample in parallel: s01 to s05, then gate_skera, gate_lima and gate_adapter_index (all three, even if
#            one fails)
#   phase 2, per sample, one sample at a time (memory): s06, s07
#   phase 3, cross-sample: s06 --all, s07 --all
#   phase 4, cross-sample: the calibration set (build_calibration_set, skera_split) into <out root>/calibration; the
#            analyses into <run>/analyses/<analysis>, reading that calibration set and this run's sample outputs;
#            head_version_check and tso_end_validation into <out root>/calibration; ArgenTag (sread_coords,
#            annotate_non_passing, full_read_view, independent_scan) into <out root>/argentag, when argentag_dir holds
#            data/segmented.bam and data/segmented.non_passing.bam; then the report steps, when their scripts exist.
#            A part left out is named in the log with the reason.
# <run> is the out root at fraction 1 and <out root>/hash_<fraction> below 1, the folder of the sample outputs.
# The first failing step or hard gate stops every other chain at once and the run (exit 1). gate_skera is a
# measurement: a crash only warns and trashes a stale gate_skera.json, so that no later step reads it, and the run
# goes on. A step that exits 0 writes <out root>/logs/<sample or all>/<step>.ok (date and time, git HEAD, seconds, and
# "fraction F fail_subsample N" when the step's outputs sit under <run>); a rerun skips it. A step without .ok, or whose
# .ok names another fraction or fail_subsample, runs from scratch; the steps that write into <out root>/calibration or
# <out root>/argentag, shared by every fraction, skip whatever fraction their .ok was written at. Step logs
# <out root>/logs/<sample or all>/<step>.log are appended.
# INT, TERM or HUP to run_study.sh stops every chain it started. Only one run_study.sh may run at a time per out root;
# a second one exits 3, naming the lock file it could not take.
# Usage, from the repository root:
#   study/run_study.sh [--config study/study.toml] [--out DIR] [--from STEP] [--fresh]
#   --config  the study's TOML (default study/study.toml), with the .local.toml beside it laid over it
#   --out     out root (default: out_root of the config)
#   --from    trash the .ok markers of the first step whose name starts with STEP and of every later step (every
#             sample and all), then run; for example --from s06, --from gate_lima, --from mirror_arms
#   --fresh   trash every .ok marker of the out root, then run
# From the config: fraction, fail_subsample, processes (per sample), threads (lima, isoseq refine and the downstream
# test), pbtools_env, samples, out_root, argentag_dir, reference_fasta, skera120_env, samtools_env, samtools and
# minimap2. POLARS_MAX_THREADS defaults to processes; s06 gets --mem-budget-gb MEM_BUDGET_GB when it is set in the
# environment, else floor(0.8 x MemAvailable GiB), read when phase 2 starts. RUN_STUDY_SCRIPTS (for the tests) replaces
# the study folder the step scripts are taken from.
set -euo pipefail

for arg in "$@"; do
  if [[ $arg == -h || $arg == --help ]]; then sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
done
usage() { echo "usage: run_study.sh [--config PATH] [--out DIR] [--from STEP] [--fresh]" >&2; exit 2; }
CONFIG=study/study.toml OUT= FROM= FRESH=0
while (( $# )); do
  case $1 in
    --config) [[ -n ${2:-} ]] || usage; CONFIG=$2; shift ;;
    --out)    [[ -n ${2:-} ]] || usage; OUT=$2; shift ;;
    --from)   [[ -n ${2:-} ]] || usage; FROM=$2; shift ;;
    --fresh)  FRESH=1 ;;
    *)        usage ;;
  esac
  shift
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$(realpath -m "$CONFIG")
if [[ -n $OUT ]]; then OUT=$(realpath -m "$OUT"); fi
cd "$REPO"
PY="$REPO/.venv/bin/python"
SCRIPTS=${RUN_STUDY_SCRIPTS:-$REPO/study}

# The config, read through study/pipeline/config.py: one "key<TAB>value" line per value, one "sample<TAB>name" per sample.
CFG=$("$PY" - "$CONFIG" "$OUT" <<'EOF'
import sys

sys.path.insert(0, "study/pipeline")
from config import load_config, load_samples, out_dir

config = load_config(sys.argv[1])
out_root = sys.argv[2] or config["out_root"]
if out_root.startswith("<"):
    sys.exit(f"run_study.sh: no out root: give --out or set out_root in the .local.toml beside {sys.argv[1]}")
keys = ("fraction", "fail_subsample", "processes", "threads", "pbtools_env", "samples", "argentag_dir", "reference_fasta",
        "skera120_env", "samtools_env", "samtools", "minimap2")
values = {k: config.get(k, "") for k in keys}
values["out_root"] = out_root
values["fraction_tag"] = f"{config['fraction']:g}"
values["run_dir"] = out_dir(out_root, "all", config["fraction"]).parent
for k, v in values.items():
    print(k, v, sep="\t")
for s in load_samples(config["samples"]):
    print("sample", s, sep="\t")
EOF
) || { echo "run_study.sh: cannot read the config $CONFIG or its samples table, see above" >&2; exit 2; }
declare -A C=()
SAMPLES=()
while IFS=$'\t' read -r k v; do
  if [[ $k == sample ]]; then SAMPLES+=("$v"); else C[$k]=$v; fi
done <<< "$CFG"
(( ${#SAMPLES[@]} )) || { echo "run_study.sh: no samples in ${C[samples]}" >&2; exit 2; }

PHASE0=(gate_synthetic)
PHASE1=(s01_extract_segments s02_annotate_reads s03_build_reads s04_lima_refine s05_fate gate_skera gate_lima gate_adapter_index)
PHASE2=(s06_concatenation_bias s07_depletion)
PHASE3=(s06_concatenation_bias s07_depletion)
CALIBRATION=(build_calibration_set skera_split)
ANALYSES=(false_cuts_calibration "${SAMPLES[@]/#/false_cut_anatomy.}"
          mirror_arms.nd_hash_skera26 mirror_arms.nd_cal_skera26 mirror_arms.nd_cal_skera12 "${SAMPLES[@]/#/molecule_budget.}"
          downstream_test discordant_reads chain_benchmark skera_whole_cell head_version_check tso_end_validation)
ARGENTAG=(sread_coords annotate_non_passing full_read_view independent_scan)
REPORT=(report_compute report_render)
PHASE4=("${CALIBRATION[@]}" "${ANALYSES[@]}" "${ARGENTAG[@]}" "${REPORT[@]}")
SHARED=" ${CALIBRATION[*]} head_version_check tso_end_validation ${ARGENTAG[*]} "   # outputs outside <run>
ORDER=("${PHASE0[@]}" "${PHASE1[@]}" "${PHASE2[@]}" "${PHASE3[@]/#/all:}" "${PHASE4[@]/#/all:}")   # all: cross-sample steps
FROM_AT=
if [[ -n $FROM ]]; then
  for i in "${!ORDER[@]}"; do
    if [[ ${ORDER[i]#all:} == "$FROM"* ]]; then FROM_AT=$i; break; fi
  done
  [[ -n $FROM_AT ]] || { echo "run_study.sh: no step starts with '$FROM'" >&2; usage; }
fi

export REPO PY SCRIPTS CONFIG SHARED
export OUT_ROOT=${C[out_root]} RUN_DIR=${C[run_dir]} FRACTION=${C[fraction]} FRACTION_TAG=${C[fraction_tag]} \
  FAIL_SUBSAMPLE=${C[fail_subsample]} \
  PROCESSES=${C[processes]} THREADS=${C[threads]} PBTOOLS_ENV=${C[pbtools_env]} SAMPLES_TSV=${C[samples]} \
  ARGENTAG_DIR=${C[argentag_dir]} REFERENCE=${C[reference_fasta]} SKERA120_ENV=${C[skera120_env]} \
  SAMTOOLS_ENV=${C[samtools_env]} SAMTOOLS=${C[samtools]} MINIMAP2=${C[minimap2]}
export CAL=$OUT_ROOT/calibration ARGENTAG_OUT=$OUT_ROOT/argentag LOGS=$OUT_ROOT/logs
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-$PROCESSES}

mkdir -p "$LOGS"
LOCK_FILE=$LOGS/.run_study.lock
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "run_study.sh: run_study.sh or one of its chains holds the lock $LOCK_FILE, exiting (after a KILL of" \
  "run_study.sh its chains keep running and hold the lock: pgrep -af 'run_chain ' lists them)" >&2; exit 3; }

stamp() { date '+%F %T'; }

step_cmd() {   # step_cmd <sample|all> <step>: the command of one step, into the array CMD
  local s=$1 st=$2 a=$RUN_DIR/analyses nd=$RUN_DIR/BD144_ND truth=$CAL/nd_first200k_truth
  local common=(--samples "$SAMPLES_TSV" --out-root "$OUT_ROOT" --fraction "$FRACTION")
  case $st in
    gate_*)             CMD=("$PY" "$SCRIPTS/pipeline/gates.py" --sample "$s" --gate "${st#gate_}" "${common[@]}"
                             --fail-subsample "$FAIL_SUBSAMPLE") ;;
    s04_lima_refine)    CMD=(bash "$SCRIPTS/pipeline/$st.sh" --sample "$s" "${common[@]}" --threads "$THREADS"
                             --pbtools-env "$PBTOOLS_ENV") ;;
    s0[67]_*)           if [[ $s == all ]]; then CMD=("$PY" "$SCRIPTS/pipeline/$st.py" --all "${common[@]}")
                        else CMD=("$PY" "$SCRIPTS/pipeline/$st.py" --sample "$s" "${common[@]}" --processes "$PROCESSES")
                             if [[ $st == s06_* ]]; then CMD+=(--mem-budget-gb "$MEM_BUDGET_GB"); fi
                        fi ;;
    s02_annotate_reads) CMD=("$PY" "$SCRIPTS/pipeline/$st.py" --sample "$s" "${common[@]}" --processes "$PROCESSES"
                             --fail-subsample "$FAIL_SUBSAMPLE") ;;
    s0*)                CMD=("$PY" "$SCRIPTS/pipeline/$st.py" --sample "$s" "${common[@]}" --processes "$PROCESSES") ;;
    build_calibration_set)
      CMD=("$PY" "$SCRIPTS/calibration/$st.py" --samples "$SAMPLES_TSV" --out "$CAL" --processes "$PROCESSES"
           --samtools "$SAMTOOLS" --samtools-env "$SAMTOOLS_ENV") ;;
    skera_split)
      CMD=("$PY" "$SCRIPTS/calibration/$st.py" --calibration-dir "$CAL" --out "$CAL" --skera-env "$SKERA120_ENV") ;;
    head_version_check|tso_end_validation)
      CMD=("$PY" "$SCRIPTS/calibration/$st.py" --calibration-dir "$CAL" --out "$CAL") ;;
    false_cuts_calibration)
      CMD=("$PY" "$SCRIPTS/analyses/$st.py" --truth "$truth" --hifi "$CAL/nd_first200k.hifi.nomod.bam" --out "$a/skera_false_cuts") ;;
    false_cut_anatomy.*|molecule_budget.*)
      CMD=("$PY" "$SCRIPTS/analyses/${st%%.*}.py" --sample-dir "$RUN_DIR/${st#*.}" --label "${st#*.}" --out "$a/${st%%.*}") ;;
    mirror_arms.nd_hash_skera26)
      CMD=("$PY" "$SCRIPTS/analyses/mirror_arms.py" --truth "$nd" --segments "$nd/segments.parquet" --baseline-fraction 0.01
           --label "${st#*.}" --out "$a/skera_mirror_arms") ;;
    mirror_arms.nd_cal_skera26)
      CMD=("$PY" "$SCRIPTS/analyses/mirror_arms.py" --truth "$truth" --segments "$truth/segments.parquet"
           --label "${st#*.}" --out "$a/skera_mirror_arms") ;;
    mirror_arms.nd_cal_skera12)
      CMD=("$PY" "$SCRIPTS/analyses/mirror_arms.py" --truth "$truth" --sreads "$CAL/nd_first200k.skera120.bam"
           --non-passing "$CAL/nd_first200k.skera120.non_passing.bam"
           --found-adapters "$CAL/nd_first200k.skera120.found_adapters.csv.gz"
           --ligations "$CAL/nd_first200k.skera120.ligations.csv" --label "${st#*.}" --out "$a/skera_mirror_arms") ;;
    downstream_test)
      CMD=("$PY" "$SCRIPTS/analyses/$st.py" --hifi "$CAL/nd_first200k.hifi.nomod.bam" --template "$CAL/nd_first200k.skera120.bam"
           --truth "$truth" --ref "$REFERENCE" --threads "$THREADS" --pbtools-env "$PBTOOLS_ENV" --minimap2 "$MINIMAP2"
           --out "$a/downstream_test") ;;
    discordant_reads)   CMD=("$PY" "$SCRIPTS/analyses/$st.py" --sample-dir "$nd" --out "$a/discordant_reads") ;;
    chain_benchmark)    CMD=("$PY" "$SCRIPTS/analyses/$st.py" --sreads "$CAL/nd_first200k.skera120.bam" --out "$a/mirror_benchmark") ;;
    skera_whole_cell)   CMD=("$PY" "$SCRIPTS/analyses/$st.py" --samples "$SAMPLES_TSV" --out "$a/skera_whole_cell") ;;
    sread_coords|independent_scan)
      CMD=("$PY" "$SCRIPTS/argentag/$st.py" --argentag-dir "$ARGENTAG_DIR" --out "$ARGENTAG_OUT") ;;
    annotate_non_passing)
      CMD=("$PY" "$SCRIPTS/argentag/$st.py" --argentag-dir "$ARGENTAG_DIR" --out "$ARGENTAG_OUT" --processes "$PROCESSES") ;;
    full_read_view)     CMD=("$PY" "$SCRIPTS/argentag/$st.py" --out "$ARGENTAG_OUT") ;;
    report_compute)     CMD=("$PY" "$SCRIPTS/report/compute.py" --config "$CONFIG" --out-root "$OUT_ROOT") ;;
    report_render)      CMD=("$PY" "$SCRIPTS/report/render.py" --report-dir "$RUN_DIR/report") ;;
  esac
}

run_stage() {   # run_stage <sample|all> <step>: one step, skipped when its .ok marker exists (and, for a step whose
                # outputs sit under <run>, names this run's fraction and fail_subsample)
  local s=$1 st=$2 log=$LOGS/$1 t start secs rc=0 tag= ok
  local -a CMD
  mkdir -p "$log"
  if [[ $SHARED != *" $st "* ]]; then tag=" fraction $FRACTION_TAG fail_subsample $FAIL_SUBSAMPLE"; fi
  if [[ -f $log/$st.ok ]]; then
    ok=$(<"$log/$st.ok")
    if [[ -z $tag || $ok == *"$tag" ]]; then
      echo "[$(stamp)] $s $st skip (ok: $ok)"
      return 0
    fi
    echo "[$(stamp)] $s $st: the .ok marker is not from fraction $FRACTION_TAG and fail_subsample $FAIL_SUBSAMPLE (ok: $ok), running it again"
  fi
  step_cmd "$s" "$st"
  t=$(stamp)
  echo "[$t] $s $st start"
  echo "[$t] ${CMD[*]}" >> "$log/$st.log"
  start=$SECONDS
  "${CMD[@]}" >> "$log/$st.log" 2>&1 || rc=$?
  secs=$(( SECONDS - start ))
  if (( rc )); then
    echo "[$(stamp)] FAILED: $s $st (exit $rc after $secs s, log: $log/$st.log)" >&2
    return 1
  fi
  echo "$(stamp) $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown) $secs$tag" > "$log/$st.ok"
  echo "[$(stamp)] $s $st done in $secs s"
}

run_chain() {   # run_chain <sample|all> <step>...: the steps in order, up to the first failure. The gates all run and the
                # chain fails after them if a hard gate failed; gate_skera never fails it, but a crash (not a low
                # agreement, which always exits 0) leaves no stale gate file for s07 and the report to read.
  local s=$1 st gate_failed=0 stale
  shift
  for st in "$@"; do
    case $st in
      gate_skera)
        if ! run_stage "$s" "$st"; then
          echo "[$(stamp)] WARNING: $s gate_skera crashed; skera agreement is not measured this run, the run goes on" >&2
          stale=$RUN_DIR/$s/gates/gate_skera.json
          if [[ -f $stale ]]; then gio trash "$stale"; fi
        fi ;;
      gate_*) run_stage "$s" "$st" || gate_failed=1 ;;
      *)      run_stage "$s" "$st" || return 1 ;;
    esac
  done
  return "$gate_failed"
}
export -f stamp step_cmd run_stage run_chain

declare -A ACTIVE=()   # pid of each running chain (its own process group) -> sample

stop_chains() {   # TERM the process group of every running chain, KILL what is left after 10 s, reap the chains
  local g i live=
  (( ${#ACTIVE[@]} )) || return 0
  for g in "${!ACTIVE[@]}"; do kill -TERM -- "-$g" "$g" 2>/dev/null || true; done   # "$g": a chain not yet in its group
  for i in {1..20}; do
    live=$(ps -e -o pgid=,stat= | awk -v gs=" ${!ACTIVE[*]} " '$2 !~ /^Z/ && index(gs, " " $1 " ") {print $1}')
    if [[ -z $live ]]; then break; fi
    sleep 0.5
  done
  for g in $live; do kill -KILL -- "-$g" 2>/dev/null || true; done
  wait "${!ACTIVE[@]}" 2>/dev/null || true
  ACTIVE=()
}

on_signal() {   # on_signal <name> <exit status>
  echo "[$(stamp)] STOPPED: run_study.sh got SIG$1, stopping every chain" >&2
  stop_chains
  exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
trap stop_chains EXIT

phase() {   # phase <sample|all>... -- <step>...: one chain per sample, in parallel; the first failing chain stops the others
  local names=() s p pid rc failed
  while [[ $1 != -- ]]; do names+=("$1"); shift; done
  shift
  for s in "${names[@]}"; do
    setsid bash -c 'set -euo pipefail; run_chain "$@"' run_chain "$s" "$@" &
    ACTIVE[$!]=$s
  done
  while (( ${#ACTIVE[@]} )); do
    rc=0 pid=
    wait -n -p pid "${!ACTIVE[@]}" || rc=$?
    if [[ -z ${pid:-} ]]; then   # observed with bash 5.1: when chains end at nearly the same time, wait -n now and
                                 # then returns 127 and leaves pid unset; then wait for an ended chain by its pid,
                                 # which gives its status
      for p in "${!ACTIVE[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then pid=$p rc=0; wait "$p" || rc=$?; break; fi
      done
      if [[ -z ${pid:-} ]]; then sleep 0.2; continue; fi
    fi
    if (( rc )); then
      failed=${ACTIVE[${pid:-none}]:-unknown}
      stop_chains
      echo "[$(stamp)] STOPPED: the $failed chain exited $rc, every other chain was stopped (see the FAILED lines above)" >&2
      exit 1
    fi
    unset "ACTIVE[$pid]"
  done
}

trash_markers() {   # trash_markers <what> <marker>...: move the markers that exist to the trash
  local what=$1 f have=()
  shift
  for f in "$@"; do if [[ -f $f ]]; then have+=("$f"); fi; done
  if (( ${#have[@]} )); then gio trash "${have[@]}"; fi
  echo "[$(stamp)] $what: trashed ${#have[@]} .ok markers"
}

PGID=$(ps -o pgid= -p $$)
echo "[$(stamp)] run_study.sh pid $$, process group ${PGID// /}; config $CONFIG, out root $OUT_ROOT, fraction $FRACTION;" \
  "samples: ${SAMPLES[*]}; per sample: $PROCESSES processes, $THREADS threads, POLARS_MAX_THREADS=$POLARS_MAX_THREADS"
if (( FRESH )); then
  shopt -s nullglob
  M=("$LOGS"/*/*.ok)
  shopt -u nullglob
  trash_markers --fresh "${M[@]}"
fi
if [[ -n $FROM_AT ]]; then
  M=()
  for st in "${ORDER[@]:FROM_AT}"; do
    if [[ $st == all:* ]]; then M+=("$LOGS/all/${st#all:}.ok")
    else for s in "${SAMPLES[@]}"; do M+=("$LOGS/$s/$st.ok"); done; fi
  done
  trash_markers "--from $FROM" "${M[@]}"
fi

phase "${SAMPLES[@]}" -- "${PHASE0[@]}"
phase "${SAMPLES[@]}" -- "${PHASE1[@]}"
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(awk '/^MemAvailable:/ {print int(0.8 * $2 / 1048576)}' /proc/meminfo)}
export MEM_BUDGET_GB
echo "[$(stamp)] phase 2, one sample at a time: s06 --mem-budget-gb $MEM_BUDGET_GB"
for s in "${SAMPLES[@]}"; do phase "$s" -- "${PHASE2[@]}"; done
phase all -- "${PHASE3[@]}"

RUN4=("${CALIBRATION[@]}" "${ANALYSES[@]}")
ARGENTAG_SKIP=
if [[ -z $ARGENTAG_DIR || $ARGENTAG_DIR == \<* ]]; then ARGENTAG_SKIP="argentag_dir is not set"
elif [[ ! -d $ARGENTAG_DIR ]]; then ARGENTAG_SKIP="$ARGENTAG_DIR does not exist"
else
  for f in "$ARGENTAG_DIR/data/segmented.bam" "$ARGENTAG_DIR/data/segmented.non_passing.bam"; do
    if [[ ! -f $f ]]; then ARGENTAG_SKIP="$f does not exist"; break; fi
  done
fi
if [[ -n $ARGENTAG_SKIP ]]; then echo "[$(stamp)] ArgenTag skipped: $ARGENTAG_SKIP"; else RUN4+=("${ARGENTAG[@]}"); fi
for st in "${REPORT[@]}"; do
  f=$SCRIPTS/report/${st#report_}.py
  if [[ -n $ARGENTAG_SKIP ]]; then echo "[$(stamp)] $st skipped: the Note needs the ArgenTag check, which was skipped ($ARGENTAG_SKIP)"
  elif [[ -f $f ]]; then RUN4+=("$st"); else echo "[$(stamp)] $st skipped: $f does not exist"; fi
done
phase all -- "${RUN4[@]}"
echo "[$(stamp)] done: $RUN_DIR"
