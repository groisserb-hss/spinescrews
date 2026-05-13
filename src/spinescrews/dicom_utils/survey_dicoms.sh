#!/usr/bin/env bash
# Survey DICOM directories and output structured JSON metadata.
#
# Recursively scans directories for DICOM files, groups by Study > Series,
# and reports reconstruction protocols and image parameters as JSON.
# Designed to identify MAR vs non-MAR reconstructions, extract dose info,
# and feed convert_to_nii.sh for series selection and NIfTI conversion.
#
# Usage:
#   survey_dicoms.sh [-o metadata.json] DIR [DIR ...]
#
# Architecture (4 phases):
#   Phase 1: Recursively find files, extract Study/Series UIDs via dcmdump.
#            Scans ALL files (no early stopping) so slice counts are exact.
#            Supports GNU parallel for speed if available.
#   Phase 2: Group files by unique Study+Series UID. Pick one representative
#            file per series. Count files per series (= slice count).
#            Store full file list per series.
#   Phase 3: For each representative file, do a single dcmdump and extract
#            all metadata tags (kernel, dose, resolution, etc.) via awk.
#   Phase 4: Assemble JSON output via jq.
#
# Key design decisions:
#   - DICOMDIR files are excluded from scanning because they contain
#     Study/Series UIDs from directory records and would be incorrectly
#     picked as representative files (their dumps lack image-level tags).
#   - Uses ASCII Unit Separator (0x1F) as internal field delimiter instead
#     of tab, because bash 'read' collapses consecutive tabs (whitespace),
#     losing empty fields and shifting all subsequent columns.
#   - The awk tag parser handles BOTH bracketed values (DS [0.625]) and
#     non-bracketed numeric values (US 512). dcmdump uses brackets for
#     string-like VRs but not for integer/float VRs (US, SS, UL, FL, FD).
#   - When a tag appears at multiple nesting levels (Enhanced CT IOD with
#     functional group sequences), the parser prefers the least-indented
#     (most top-level) occurrence.
#   - MAR detection checks SeriesDescription and ConvolutionKernel for
#     vendor-specific keywords (iMAR, OMAR, SEMAR, BONEPLUS, "MAR").
#     ReconstructionAlgorithm (0018,9315) is NOT used — on GE scanners
#     this field often contains unrelated values.
#
# Requires: dcmdump (from DCMTK package), jq
# Optional: GNU parallel (for faster scanning)

set -euo pipefail

# ─── Argument parsing ───────────────────────────────────────────────

output_file=""
dirs=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            output_file="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-o metadata.json] DIR [DIR ...]"
            echo ""
            echo "Recursively scans DICOM directories and reports reconstruction"
            echo "protocols grouped by Study > Series as JSON."
            echo ""
            echo "Options:"
            echo "  -o, --output FILE   Write JSON to file (default: stdout)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            dirs+=("$1")
            shift
            ;;
    esac
done

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "Error: No directories specified." >&2
    echo "Usage: $0 [-o metadata.json] DIR [DIR ...]" >&2
    exit 1
fi

command -v dcmdump >/dev/null 2>&1 || { echo "Error: dcmdump not found. Install DCMTK." >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "Error: jq not found. Install jq." >&2; exit 1; }

# Check for GNU parallel
use_parallel=0
if command -v parallel >/dev/null 2>&1; then
    use_parallel=1
fi

# Field separator for internal detail file.
# Using ASCII Unit Separator (0x1F) instead of tab because bash 'read'
# collapses consecutive tab/space delimiters, losing empty fields.
# Unit separator is non-whitespace so bash preserves empty fields.
SEP=$'\x1f'

# ─── Helper functions ───────────────────────────────────────────────

# Awk script shared by dicom_value() and Phase 3 tag extraction.
# Handles both bracketed values (DS, LO, SH, CS, etc.) and
# non-bracketed numeric values (US, SS, UL, SL, FL, FD).
# When multiple matches exist, prefers the least-indented (most
# top-level) occurrence — important for Enhanced CT IOD files where
# tags appear inside nested functional group sequences.
PARSE_TAG_AWK='
{
    match($0, /^ */)
    indent = RLENGTH
    val = ""

    # Try bracketed value first: [content]
    if (index($0, "[") > 0) {
        s = substr($0, index($0, "[") + 1)
        p = index(s, "]")
        if (p > 0) {
            val = substr(s, 1, p - 1)
        }
    }
    # Non-bracketed numeric: "(tag) VR value  # comment"
    # Matches US, SS, UL, SL, FL, FD (2-letter uppercase VR, no brackets)
    else if (match($0, /\) [A-Z][A-Z] /)) {
        s = substr($0, RSTART + RLENGTH)
        sub(/ *#.*/, "", s)
        sub(/ +$/, "", s)
        # Skip sequence/item descriptions like "(Sequence with ...)"
        if (s != "" && substr(s, 1, 1) != "(") val = s
    }

    if (val == "") next

    if (!found || indent < min_indent) {
        min_indent = indent
        best = val
        found = 1
    }
}
END { if (found) print best }'

dicom_value() {
    # Extract the value of a DICOM tag from a file.
    local tag="$1" file="$2"
    dcmdump +P "$tag" "$file" 2>/dev/null | awk "$PARSE_TAG_AWK"
}
export -f dicom_value

# Export the awk script so parallel subshells can use it
export PARSE_TAG_AWK

extract_uids() {
    # Extract StudyInstanceUID and SeriesInstanceUID from a single file.
    # Prints: study_uid<TAB>series_uid<TAB>filepath
    # Silently skips non-DICOM files.
    local file="$1"
    local study_uid series_uid
    study_uid="$(dicom_value 0020,000d "$file")"
    [[ -z "$study_uid" ]] && return 0
    series_uid="$(dicom_value 0020,000e "$file")"
    [[ -z "$series_uid" ]] && return 0
    printf '%s\t%s\t%s\n' "$study_uid" "$series_uid" "$file"
}
export -f extract_uids

# ─── Phase 1: Find files and extract UIDs ────────────────────────────

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

uid_file="$WORK_DIR/uids.tsv"
touch "$uid_file"

total_files=0

for dir in "${dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "Warning: '$dir' is not a directory, skipping." >&2
        continue
    fi
    echo "Scanning: $dir" >&2

    filelist="$WORK_DIR/filelist.txt"
    find "$dir" -type f \
        -not -name "DICOMDIR" \
        -not -name "README*" \
        -not -name ".DS_Store" \
        -not -name "*.txt" \
        -not -name "*.xml" \
        -not -name "*.json" \
        -not -name "*.csv" \
        -not -name "*.pdf" \
        > "$filelist"
    file_count=$(wc -l < "$filelist" | tr -d ' ')
    total_files=$((total_files + file_count))

    dir_uid="$WORK_DIR/dir_uids.tsv"
    : > "$dir_uid"

    if [[ "$use_parallel" -eq 1 ]]; then
        parallel -j 8 extract_uids {} < "$filelist" >> "$dir_uid" 2>/dev/null || true
    else
        while IFS= read -r f; do
            extract_uids "$f" >> "$dir_uid" 2>/dev/null || true
        done < "$filelist"
    fi

    nseries=$(cut -f1-2 "$dir_uid" | sort -u | wc -l | tr -d ' ')
    ndicom=$(wc -l < "$dir_uid" | tr -d ' ')

    cat "$dir_uid" >> "$uid_file"
    echo "  Found $nseries series in $ndicom DICOM files (of $file_count total)" >&2
done

dicom_count=$(wc -l < "$uid_file" | tr -d ' ')
echo "Total: $total_files files, $dicom_count DICOM instances." >&2

if [[ "$dicom_count" -eq 0 ]]; then
    echo "No DICOM files found. Check that dcmdump can read files in the specified directories." >&2
    exit 1
fi

# ─── Phase 2: Group by series, pick representatives, collect file lists ──

# Sort by study UID then series UID
sort -t$'\t' -k1,1 -k2,2 "$uid_file" > "$WORK_DIR/sorted.tsv"

# For each unique series: first file = representative, count = files seen.
# Output: study_uid<TAB>series_uid<TAB>count<TAB>rep_file
awk -F'\t' '
{
    key = $1 "\t" $2
    count[key]++
    if (count[key] == 1) {
        rep[key] = $3
        study[key] = $1
    }
}
END {
    for (key in count) {
        print key "\t" count[key] "\t" rep[key]
    }
}
' "$WORK_DIR/sorted.tsv" | sort -t$'\t' -k1,1 -k2,2 > "$WORK_DIR/series.tsv"

# Build per-series file lists: one file per line, grouped by study+series UID.
# File: WORK_DIR/files_<md5>.txt for each unique study+series key.
# We use a simple index file to map series keys to file list paths.
filelist_dir="$WORK_DIR/filelists"
mkdir -p "$filelist_dir"

awk -F'\t' '{
    key = $1 "\t" $2
    # Use a sanitized filename based on series UID (replace dots with underscores)
    gsub(/[^a-zA-Z0-9]/, "_", $2)
    fname = $2
    print $3 >> ("'"$filelist_dir"'/" fname ".txt")
}' "$WORK_DIR/sorted.tsv"

# Build an index: study_uid<TAB>series_uid -> filelist path
awk -F'\t' '{
    key = $1 "\t" $2
    if (!(key in seen)) {
        uid2 = $2
        gsub(/[^a-zA-Z0-9]/, "_", uid2)
        print key "\t" "'"$filelist_dir"'/" uid2 ".txt"
        seen[key] = 1
    }
}' "$WORK_DIR/sorted.tsv" > "$WORK_DIR/filelist_index.tsv"

num_series=$(wc -l < "$WORK_DIR/series.tsv" | tr -d ' ')
echo "Found $num_series unique series. Reading metadata..." >&2

# ─── Phase 3: Read full metadata from representatives ────────────────

# Tags to read for each representative file.
# Order matters — must match the field indices in Phase 4.
TAGS=(
    "0010,0010"  # 1  PatientName
    "0010,0020"  # 2  PatientID
    "0008,0020"  # 3  StudyDate
    "0008,1030"  # 4  StudyDescription
    "0008,103e"  # 5  SeriesDescription
    "0018,1030"  # 6  ProtocolName
    "0018,1210"  # 7  ConvolutionKernel
    "0018,0050"  # 8  SliceThickness
    "0028,0030"  # 9  PixelSpacing
    "0028,0010"  # 10 Rows
    "0028,0011"  # 11 Columns
    "0018,0060"  # 12 KVP
    "0018,1151"  # 13 XRayTubeCurrent
    "0018,1152"  # 14 Exposure (mAs)
    "0018,9345"  # 15 CTDIvol
    "0008,0070"  # 16 Manufacturer
    "0008,1090"  # 17 ManufacturerModelName
    "0018,1100"  # 18 ReconstructionDiameter
)

# Read metadata for each series representative.
# Dump each file once, then extract all tags from the cached output.
detail_file="$WORK_DIR/details.dat"
: > "$detail_file"

while IFS=$'\t' read -r study_uid series_uid slice_count rep_file; do
    line="${study_uid}${SEP}${series_uid}${SEP}${slice_count}"
    # Single dcmdump per representative file
    dump="$(dcmdump "$rep_file" 2>/dev/null)"
    for tag in "${TAGS[@]}"; do
        val="$(awk -v pat="($tag)" '
        index($0, pat) == 0 { next }
        {
            match($0, /^ */)
            indent = RLENGTH
            val = ""
            if (index($0, "[") > 0) {
                s = substr($0, index($0, "[") + 1)
                p = index(s, "]")
                if (p > 0) val = substr(s, 1, p - 1)
            } else if (match($0, /\) [A-Z][A-Z] /)) {
                s = substr($0, RSTART + RLENGTH)
                sub(/ *#.*/, "", s)
                sub(/ +$/, "", s)
                if (s != "" && substr(s, 1, 1) != "(") val = s
            }
            if (val == "") next
            if (!found || indent < min_indent) {
                min_indent = indent
                best = val
                found = 1
            }
        }
        END { if (found) print best }' <<< "$dump")"
        line+="${SEP}${val}"
    done
    printf '%s\n' "$line" >> "$detail_file"
done < "$WORK_DIR/series.tsv"

echo "Metadata collected. Generating JSON..." >&2

# ─── Phase 4: Assemble JSON output ──────────────────────────────────

generate_json() {
    local sorted_details
    sorted_details="$(sort -t"$SEP" -k1,1 "$detail_file")"

    # Write each series/study as individual JSON files on disk, then combine.
    # This avoids passing large file-list arrays through CLI args (ARG_MAX).
    local series_json_dir="$WORK_DIR/series_json"
    local study_json_dir="$WORK_DIR/study_json"
    mkdir -p "$series_json_dir" "$study_json_dir"

    local prev_study="" study_count=0 series_count=0 mar_count=0
    local all_kernels=""
    local study_series_files=()   # series JSON files for current study
    local study_json_files=()     # all study JSON files in order

    # Study-level fields (set when study changes)
    local s_patient_name="" s_patient_id="" s_study_date=""
    local s_study_desc="" s_manufacturer="" s_model="" s_study_uid=""

    # Flush accumulated series into a study JSON file
    flush_study() {
        [[ -z "$s_study_uid" ]] && return

        # Combine series JSON files into an array
        local series_arr_file="$WORK_DIR/tmp_series_arr.json"
        if [[ ${#study_series_files[@]} -gt 0 ]]; then
            jq -s '.' "${study_series_files[@]}" > "$series_arr_file"
        else
            echo '[]' > "$series_arr_file"
        fi

        local sfile="$study_json_dir/${study_count}.json"
        jq -n \
            --arg suid "$s_study_uid" \
            --arg pname "$s_patient_name" \
            --arg pid "$s_patient_id" \
            --arg sdate "$s_study_date" \
            --arg sdesc "$s_study_desc" \
            --arg mfr "$s_manufacturer" \
            --arg model "$s_model" \
            --slurpfile series "$series_arr_file" \
            '{
                study_instance_uid: $suid,
                patient_name: $pname,
                patient_id: $pid,
                study_date: $sdate,
                study_description: $sdesc,
                manufacturer: $mfr,
                model: $model,
                series: $series[0]
            }' > "$sfile"
        study_json_files+=("$sfile")
    }

    while IFS="$SEP" read -r study_uid series_uid slice_count \
        patient_name patient_id study_date study_desc \
        series_desc protocol_name kernel \
        slice_thick pixel_spacing rows cols \
        kvp tube_current exposure ctdivol \
        manufacturer model_name recon_diameter; do

        series_count=$((series_count + 1))

        # Detect MAR
        mar=false
        if [[ -n "$series_desc" ]]; then
            if echo "$series_desc" | grep -iqE 'iMAR|OMAR|O-MAR|SEMAR|BONEPLUS'; then
                mar=true
            elif echo "$series_desc" | grep -iqw 'MAR'; then
                mar=true
            fi
        fi
        if [[ "$mar" == "false" ]] && [[ -n "$kernel" ]]; then
            if echo "$kernel" | grep -iqE 'BONEPLUS'; then
                mar=true
            fi
        fi
        if [[ "$mar" == "true" ]]; then
            mar_count=$((mar_count + 1))
        fi

        # Track kernels
        if [[ -n "$kernel" ]]; then
            all_kernels+="$kernel"$'\n'
        fi

        # New study? Flush the previous one.
        if [[ "$study_uid" != "$prev_study" ]]; then
            flush_study
            study_count=$((study_count + 1))
            prev_study="$study_uid"
            study_series_files=()
            s_study_uid="$study_uid"
            s_patient_name="$patient_name"
            s_patient_id="$patient_id"
            s_study_date="$study_date"
            s_study_desc="$study_desc"
            s_manufacturer="$manufacturer"
            s_model="$model_name"
        fi

        # Build files JSON array on disk (avoids ARG_MAX for large series)
        local filelist_path=""
        filelist_path="$(awk -F'\t' -v su="$study_uid" -v se="$series_uid" \
            '$1==su && $2==se {print $3; exit}' "$WORK_DIR/filelist_index.tsv")"

        local files_json_file="$series_json_dir/${series_count}_files.json"
        if [[ -n "$filelist_path" ]] && [[ -f "$filelist_path" ]]; then
            jq -R . < "$filelist_path" | jq -s . > "$files_json_file"
        else
            echo '[]' > "$files_json_file"
        fi

        # Build series JSON object — file list read via --slurpfile
        local sfile="$series_json_dir/${series_count}.json"
        jq -n \
            --arg seuid "$series_uid" \
            --arg sdesc "$series_desc" \
            --arg proto "$protocol_name" \
            --arg kern "$kernel" \
            --argjson mar "$mar" \
            --arg sthick "$slice_thick" \
            --arg pxsp "$pixel_spacing" \
            --arg r "$rows" \
            --arg c "$cols" \
            --argjson sc "$slice_count" \
            --arg rd "$recon_diameter" \
            --arg kv "$kvp" \
            --arg tc "$tube_current" \
            --arg exp "$exposure" \
            --arg ctdi "$ctdivol" \
            --slurpfile files "$files_json_file" \
            '{
                series_instance_uid: $seuid,
                series_description: $sdesc,
                protocol: $proto,
                kernel: $kern,
                mar: $mar,
                slice_thickness: (if $sthick == "" then null else $sthick end),
                pixel_spacing: (if $pxsp == "" then null else $pxsp end),
                rows: (if $r == "" then null else $r end),
                columns: (if $c == "" then null else $c end),
                slice_count: $sc,
                recon_diameter: (if $rd == "" then null else $rd end),
                kvp: (if $kv == "" then null else $kv end),
                tube_current: (if $tc == "" then null else $tc end),
                exposure: (if $exp == "" then null else $exp end),
                ctdi_vol: (if $ctdi == "" then null else $ctdi end),
                files: $files[0]
            }' > "$sfile"

        study_series_files+=("$sfile")

    done <<< "$sorted_details"

    # Flush last study
    flush_study

    # Combine all study JSON files into a single array on disk
    local all_studies_file="$WORK_DIR/all_studies.json"
    if [[ ${#study_json_files[@]} -gt 0 ]]; then
        jq -s '.' "${study_json_files[@]}" > "$all_studies_file"
    else
        echo '[]' > "$all_studies_file"
    fi

    # Build unique kernels list
    local kernels_json="[]"
    if [[ -n "$all_kernels" ]]; then
        kernels_json="$(printf '%s' "$all_kernels" | grep -v '^$' | sort -u | jq -R . | jq -s .)"
    fi

    # Build directory list
    local dirs_json
    dirs_json="$(printf '%s\n' "${dirs[@]}" | jq -R . | jq -s .)"

    # Assemble top-level JSON — studies read from disk via --slurpfile
    jq -n \
        --arg gen "$(date '+%Y-%m-%d %H:%M')" \
        --argjson directories "$dirs_json" \
        --argjson total_files "$total_files" \
        --argjson dicom_files "$dicom_count" \
        --slurpfile studies "$all_studies_file" \
        --argjson study_count "$study_count" \
        --argjson series_count "$series_count" \
        --argjson kernels "$kernels_json" \
        --argjson mar_count "$mar_count" \
        '{
            generated: $gen,
            directories: $directories,
            total_files: $total_files,
            dicom_files: $dicom_files,
            studies: $studies[0],
            summary: {
                study_count: $study_count,
                series_count: $series_count,
                unique_kernels: $kernels,
                mar_series_count: $mar_count
            }
        }'
}

if [[ -n "$output_file" ]]; then
    generate_json > "$output_file"
    echo "JSON written to: $output_file" >&2
else
    generate_json
fi
