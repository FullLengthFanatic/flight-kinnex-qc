#!/usr/bin/env bash
# Extract the plain-file inputs for the study from instrument tar archives.
#
# Usage: prepare_inputs.sh MEMBER_TABLE OUT_DIR
#
# MEMBER_TABLE is a TSV with one row per sample and columns:
#   sample  data_tar  analysis_tar  hifi_member  fail_member  zmw_metrics_member
#   segmented_member  non_passing_member  skera_report_member
#
# For each sample this writes OUT_DIR/<sample>/ with:
#   hifi_reads.bam  fail_reads.bam  zmw_metrics.csv.gz
#   segmented.bam  segmented.non_passing.bam  read_segmentation.report.json
#
# Each tar is opened once per run (one `tar -xvvf ... member1 member2 ...`
# call per tar, not one call per member). Members land in OUT_DIR/<sample>/.staging/
# first and are moved to their flat final name only once tar has exited 0 and
# the extracted file's size matches the size tar reported for it. A member
# whose final file already exists with the right size is left alone, so a
# rerun resumes instead of re-extracting everything.
#
# Samples are processed in parallel; within one sample, its two tars are
# extracted one after the other.

set -uo pipefail

MIN_FREE_GB="${MIN_FREE_GB:-450}"
# MIN_FREE_GB (default 450, above the 422 GiB the three BD144 cells extract to) sets the free-space floor.

# Size of a member in a `tar -tvv` or `tar -xvv` listing: mode, owner/group, size, date, time, name.
# The name is everything after the fifth field, so names with spaces match; ENVIRON keeps backslashes as they are.
member_size() {
    M="$2" awk '$1 ~ /^-/ { name = $0; sub(/^[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ /, "", name); if (name == ENVIRON["M"]) { print $3; exit } }' "$1"
}

usage() {
    echo "Usage: $0 MEMBER_TABLE OUT_DIR" >&2
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

MEMBER_TABLE="$1"
OUT_DIR="$2"

if [ ! -f "$MEMBER_TABLE" ]; then
    echo "error: member table not found: $MEMBER_TABLE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

avail_gb=$(df --output=avail -BG "$OUT_DIR" | tail -n1 | tr -dc '0-9')
if [ -z "$avail_gb" ] || [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
    echo "error: only ${avail_gb:-0}G free on the filesystem holding $OUT_DIR, need at least ${MIN_FREE_GB}G" >&2
    exit 1
fi
echo "free space OK: ${avail_gb}G available for $OUT_DIR"

expected_cols=(sample data_tar analysis_tar hifi_member fail_member zmw_metrics_member segmented_member non_passing_member skera_report_member)
header_expected=$(IFS=$'\t'; echo "${expected_cols[*]}")
header_actual=$(head -n1 "$MEMBER_TABLE")
if [ "$header_actual" != "$header_expected" ]; then
    echo "error: unexpected header in $MEMBER_TABLE" >&2
    echo "  expected: $header_expected" >&2
    echo "  got:      $header_actual" >&2
    exit 1
fi

# Extract a set of members from one tar into a staging dir, then move each
# to its final flat name once verified.
# Args: sample tar_path staging_dir log_path "member:final" [...]
extract_tar() {
    local sample="$1" tar_path="$2" staging_dir="$3" log_path="$4"
    shift 4
    local pairs=("$@")

    local to_list=()
    local pair member final
    for pair in "${pairs[@]}"; do
        member="${pair%%:*}"
        to_list+=("$member")
    done

    local list_log
    list_log=$(mktemp)
    if ! tar -tvvf "$tar_path" -- "${to_list[@]}" >"$list_log" 2>&1; then
        echo "[$sample] error: failed to list members of $tar_path" >&2
        cat "$list_log" >&2
        rm -f "$list_log"
        return 1
    fi

    local needed=() expected_size have_size
    for pair in "${pairs[@]}"; do
        member="${pair%%:*}"
        final="${pair#*:}"
        expected_size=$(member_size "$list_log" "$member")
        if [ -z "$expected_size" ]; then
            echo "[$sample] error: member not found in $tar_path: $member" >&2
            rm -f "$list_log"
            return 1
        fi
        if [ -f "$final" ]; then
            have_size=$(stat -c%s "$final")
            if [ "$have_size" = "$expected_size" ]; then
                echo "[$sample] skip (already done, ${have_size}B): $final"
                continue
            fi
        fi
        needed+=("$member")
    done
    rm -f "$list_log"

    if [ "${#needed[@]}" -eq 0 ]; then
        echo "[$sample] nothing to extract from $tar_path"
        return 0
    fi

    mkdir -p "$staging_dir"
    echo "[$sample] extracting ${#needed[@]} member(s) from $tar_path"
    if ! tar -xvvf "$tar_path" -C "$staging_dir" -- "${needed[@]}" >"$log_path" 2>&1; then
        echo "[$sample] error: tar extraction failed for $tar_path, see $log_path" >&2
        return 1
    fi

    local ok=0 was_needed n staged got_size
    for pair in "${pairs[@]}"; do
        member="${pair%%:*}"
        final="${pair#*:}"
        was_needed=0
        for n in "${needed[@]}"; do
            [ "$n" = "$member" ] && was_needed=1
        done
        [ "$was_needed" -eq 1 ] || continue

        expected_size=$(member_size "$log_path" "$member")
        staged="$staging_dir/$member"
        if [ -z "$expected_size" ] || [ ! -f "$staged" ]; then
            echo "[$sample] error: extracted file missing: $staged" >&2
            ok=1
            continue
        fi
        got_size=$(stat -c%s "$staged")
        if [ "$got_size" != "$expected_size" ]; then
            echo "[$sample] error: size mismatch for $member (got ${got_size}B, expected ${expected_size}B)" >&2
            ok=1
            continue
        fi
        mkdir -p "$(dirname "$final")"
        mv "$staged" "$final"
        echo "[$sample] ok (${got_size}B): $final"
    done

    return "$ok"
}

process_sample() {
    local sample="$1" data_tar="$2" analysis_tar="$3" hifi_member="$4" fail_member="$5" \
        zmw_metrics_member="$6" segmented_member="$7" non_passing_member="$8" skera_report_member="$9"

    local sample_dir="$OUT_DIR/$sample"
    local staging="$sample_dir/.staging"
    mkdir -p "$sample_dir"

    local status=0
    extract_tar "$sample" "$data_tar" "$staging" "$sample_dir/data_tar_extract.log" \
        "$hifi_member:$sample_dir/hifi_reads.bam" \
        "$fail_member:$sample_dir/fail_reads.bam" \
        "$zmw_metrics_member:$sample_dir/zmw_metrics.csv.gz" \
        || status=1

    extract_tar "$sample" "$analysis_tar" "$staging" "$sample_dir/analysis_tar_extract.log" \
        "$segmented_member:$sample_dir/segmented.bam" \
        "$non_passing_member:$sample_dir/segmented.non_passing.bam" \
        "$skera_report_member:$sample_dir/read_segmentation.report.json" \
        || status=1

    if [ "$status" -eq 0 ]; then
        echo "[$sample] done"
    else
        echo "[$sample] FAILED, see logs in $sample_dir" >&2
    fi
    return "$status"
}

pids=()
sample_names=()
while IFS=$'\t' read -r sample data_tar analysis_tar hifi_member fail_member zmw_metrics_member segmented_member non_passing_member skera_report_member; do
    [ -z "$sample" ] && continue
    process_sample "$sample" "$data_tar" "$analysis_tar" "$hifi_member" "$fail_member" \
        "$zmw_metrics_member" "$segmented_member" "$non_passing_member" "$skera_report_member" &
    pids+=("$!")
    sample_names+=("$sample")
done < <(tail -n +2 "$MEMBER_TABLE")

overall=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "error: sample ${sample_names[$i]} failed" >&2
        overall=1
    fi
done

if [ "$overall" -eq 0 ]; then
    echo "all samples extracted successfully"
else
    echo "one or more samples failed; rerun the script to resume" >&2
fi
exit "$overall"
