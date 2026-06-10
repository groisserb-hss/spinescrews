#!/usr/bin/env bash
# Convert a DICOM series to NIfTI using metadata from survey_dicoms.sh.
#
# Usage:
#   # Filter mode — select series by field:value filters
#   convert_to_nii.sh METADATA_JSON OUTPUT_DIR TAG key:value [key:value ...]
#
#   # Interactive mode — browse studies and series interactively
#   convert_to_nii.sh METADATA_JSON OUTPUT_DIR TAG
#
# Arguments:
#   METADATA_JSON   Path to the JSON file produced by survey_dicoms.sh
#   OUTPUT_DIR      Full path where NIfTI output goes (e.g., /data/specimen_20)
#   TAG             Filename prefix for dcm2niix (e.g., "preop", "postop")
#   key:value       Filter on JSON field names (case-insensitive substring match)
#                   Available keys: series_description, protocol, kernel,
#                   study_description, patient_name, study_date, etc.
#                   All filters are AND-joined.
#
# Examples:
#   convert_to_nii.sh metadata.json /data/specimen_20 postop series_description:"MAZOR BONE"
#   convert_to_nii.sh metadata.json /data/specimen_20 postop kernel:BONE series_description:MAZOR
#   convert_to_nii.sh metadata.json /data/specimen_20 preop   # interactive
#
# Requires: jq, dcm2niix

set -euo pipefail

# ─── Argument parsing ───────────────────────────────────────────────

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 METADATA_JSON OUTPUT_DIR TAG [key:value ...]"
    echo ""
    echo "  Filter mode:      $0 metadata.json /data/specimen_20 postop series_description:\"MAZOR BONE\""
    echo "  Interactive mode:  $0 metadata.json /data/specimen_20 preop"
    exit 1
fi

metadata_json="$1"
output_dir="$2"
tag="$3"
shift 3

# Remaining args are key:value filters (may be empty for interactive mode)
filters=()
for arg in "$@"; do
    key="${arg%%:*}"
    val="${arg#*:}"
    if [[ -z "$key" ]] || [[ "$key" == "$val" ]]; then
        echo "Error: Invalid filter: $arg" >&2
        echo "Expected key:value (e.g., series_description:\"MAZOR BONE\")" >&2
        exit 1
    fi
    filters+=("$arg")
done

# ─── Dependency checks ──────────────────────────────────────────────

command -v jq >/dev/null 2>&1 || { echo "Error: jq not found. Install jq." >&2; exit 1; }
command -v dcm2niix >/dev/null 2>&1 || { echo "Error: dcm2niix not found." >&2; exit 1; }

if [[ ! -f "$metadata_json" ]]; then
    echo "Error: Metadata file not found: $metadata_json" >&2
    exit 1
fi

# Validate JSON
if ! jq empty "$metadata_json" 2>/dev/null; then
    echo "Error: Invalid JSON: $metadata_json" >&2
    exit 1
fi

mkdir -p "$output_dir"

# ─── Helper: format a series for display ─────────────────────────────

format_series() {
    # Reads a series JSON object from stdin, prints a one-line summary.
    jq -r '
        def pad(n): tostring | if length < n then . + (" " * (n - length)) else . end;
        def trunc(n): if length > n then .[:n-1] + "…" else . end;

        (.series_description // "(none)") as $desc |
        (.kernel // "—") as $kern |
        (.slice_thickness // "—") as $thick |
        (.rows // "—") as $r |
        (.columns // "—") as $c |
        (.slice_count // 0) as $sc |
        (.mar // false) as $mar |

        "\"\($desc | trunc(24))\"" | pad(26) | . + "  " +
        ($kern | pad(12)) + "  " +
        ($thick | tostring | pad(6)) + "mm  " +
        ($r | tostring) + "x" + ($c | tostring) | pad(48) | . + "  " +
        "~\($sc) slices" +
        (if $mar then "  [MAR]" else "" end)
    '
}

# ─── Helper: convert selected series ─────────────────────────────────

convert_series() {
    # Takes a series JSON object on stdin, converts to NIfTI.
    local series_json="$1"

    local desc
    desc="$(echo "$series_json" | jq -r '.series_description // "(none)"')"
    local n_files
    n_files="$(echo "$series_json" | jq '.files | length')"

    echo ""
    echo "Selected series: \"$desc\" ($n_files files)"

    # Create temp staging directory for symlinks
    local staging
    staging="$(mktemp -d)"
    trap 'rm -rf "$staging"' RETURN

    # Symlink all files into staging directory
    local missing=0 linked=0
    while IFS= read -r filepath; do
        if [[ -f "$filepath" ]]; then
            ln -s "$(realpath "$filepath")" "$staging/"
            linked=$((linked + 1))
        else
            missing=$((missing + 1))
        fi
    done < <(echo "$series_json" | jq -r '.files[]')

    if [[ "$missing" -gt 0 ]]; then
        echo "Warning: $missing files no longer exist on disk." >&2
    fi

    if [[ "$linked" -eq 0 ]]; then
        echo "Error: No files could be linked. Check that source files still exist." >&2
        return 1
    fi

    echo "Linked $linked files to staging directory."
    echo "Running dcm2niix..."
    echo ""

    dcm2niix -o "$output_dir" -f "$tag" -z y "$staging"

    echo ""
    echo "Output written to: $output_dir/$tag.nii.gz"
}

# ─── Mode selection ──────────────────────────────────────────────────

if [[ ${#filters[@]} -gt 0 ]]; then
    # ═══ Filter mode ═══════════════════════════════════════════════════

    echo "Filter mode"
    echo "Metadata: $metadata_json"
    echo "Output:   $output_dir"
    echo "Tag:      $tag"
    echo "Filters:"
    for f in "${filters[@]}"; do
        echo "  $f"
    done
    echo ""

    # Build jq filter expression from key:value pairs.
    # All filters are AND-joined with case-insensitive substring matching.
    # Uses ascii_downcase + contains() to avoid regex escaping issues.
    jq_filter="[.studies[].series[]]"
    for f in "${filters[@]}"; do
        key="${f%%:*}"
        val="${f#*:}"
        # Lowercase the value for case-insensitive matching; escape for jq string
        lower_val="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
        # Use jq --arg to safely inject the value (handles all special chars)
        jq_filter+=" | [.[] | select((.${key} // \"\") | ascii_downcase | contains(\$f_${key}))]"
    done

    # Build jq args array for safe value injection
    jq_args=()
    for f in "${filters[@]}"; do
        key="${f%%:*}"
        val="${f#*:}"
        lower_val="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
        jq_args+=(--arg "f_${key}" "$lower_val")
    done

    # Run the filter query (pass --arg values for safe string injection)
    matches="$(jq "${jq_args[@]}" "$jq_filter" "$metadata_json")"
    match_count="$(echo "$matches" | jq 'length')"

    if [[ "$match_count" -eq 0 ]]; then
        echo "Error: No series matched all filters." >&2
        echo ""
        echo "Available values for each filter key:" >&2
        for f in "${filters[@]}"; do
            key="${f%%:*}"
            echo "  $key:" >&2
            jq -r "[.studies[].series[].${key} // \"(empty)\"] | unique | .[]" "$metadata_json" 2>/dev/null | sed 's/^/    /' >&2
        done
        exit 1
    fi

    if [[ "$match_count" -eq 1 ]]; then
        selected="$(echo "$matches" | jq '.[0]')"
    else
        echo "Multiple series match ($match_count):"
        echo ""
        for i in $(seq 0 $((match_count - 1))); do
            series_i="$(echo "$matches" | jq ".[$i]")"
            printf "  %d) " "$((i + 1))"
            echo "$series_i" | format_series
        done
        echo ""
        read -rp "Select series [1-$match_count]: " choice
        if [[ -z "$choice" ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt "$match_count" ]]; then
            echo "Error: Invalid selection." >&2
            exit 1
        fi
        selected="$(echo "$matches" | jq ".[$((choice - 1))]")"
    fi

    convert_series "$selected"

else
    # ═══ Interactive mode ═════════════════════════════════════════════

    echo "Interactive mode"
    echo "Metadata: $metadata_json"
    echo "Output:   $output_dir"
    echo "Tag:      $tag"
    echo ""

    study_count="$(jq '.studies | length' "$metadata_json")"

    if [[ "$study_count" -eq 0 ]]; then
        echo "Error: No studies found in metadata." >&2
        exit 1
    fi

    # ── Select study ──
    study_idx=0
    if [[ "$study_count" -gt 1 ]]; then
        echo "Studies:"
        for i in $(seq 0 $((study_count - 1))); do
            pname="$(jq -r ".studies[$i].patient_name // \"(unknown)\"" "$metadata_json")"
            pid="$(jq -r ".studies[$i].patient_id // \"\"" "$metadata_json")"
            sdesc="$(jq -r ".studies[$i].study_description // \"(none)\"" "$metadata_json")"
            sdate="$(jq -r ".studies[$i].study_date // \"(unknown)\"" "$metadata_json")"
            nseries="$(jq ".studies[$i].series | length" "$metadata_json")"
            # Format date if 8 digits
            if [[ ${#sdate} -eq 8 ]]; then
                sdate="${sdate:0:4}-${sdate:4:2}-${sdate:6:2}"
            fi
            label="$pname"
            [[ -n "$pid" ]] && label+=" ($pid)"
            printf "  %d) %s — %s — %s  (%d series)\n" "$((i + 1))" "$label" "$sdesc" "$sdate" "$nseries"
        done
        echo ""
        read -rp "Select study [1-$study_count]: " choice
        if [[ -z "$choice" ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt "$study_count" ]]; then
            echo "Error: Invalid selection." >&2
            exit 1
        fi
        study_idx=$((choice - 1))
    else
        pname="$(jq -r '.studies[0].patient_name // "(unknown)"' "$metadata_json")"
        pid="$(jq -r '.studies[0].patient_id // ""' "$metadata_json")"
        sdesc="$(jq -r '.studies[0].study_description // "(none)"' "$metadata_json")"
        label="$pname"
        [[ -n "$pid" ]] && label+=" ($pid)"
        echo "Study: $label — $sdesc"
    fi

    # ── Select series ──
    nseries="$(jq ".studies[$study_idx].series | length" "$metadata_json")"

    if [[ "$nseries" -eq 0 ]]; then
        echo "Error: No series in selected study." >&2
        exit 1
    fi

    sdesc="$(jq -r ".studies[$study_idx].study_description // \"(none)\"" "$metadata_json")"
    echo ""
    echo "Series in \"$sdesc\":"
    for i in $(seq 0 $((nseries - 1))); do
        series_i="$(jq ".studies[$study_idx].series[$i]" "$metadata_json")"
        printf "  %d) " "$((i + 1))"
        echo "$series_i" | format_series
    done
    echo ""
    read -rp "Select series [1-$nseries]: " choice
    if [[ -z "$choice" ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt "$nseries" ]]; then
        echo "Error: Invalid selection." >&2
        exit 1
    fi
    series_idx=$((choice - 1))

    selected="$(jq ".studies[$study_idx].series[$series_idx]" "$metadata_json")"
    convert_series "$selected"
fi
