#!/usr/bin/env python3
"""Convert a DICOM series to NIfTI using metadata from survey_dicoms.py.

Python port of convert_to_nii.sh. Reads the survey JSON with the standard-library
``json`` module instead of ``jq``, and stages the series with hard links (falling
back to copies) instead of POSIX symlinks — so it needs no ``jq`` and no bash, and
runs unchanged on Windows, macOS, and Linux. The actual DICOM->NIfTI conversion is
still done by ``dcm2niix`` (the cross-platform binary), invoked as a subprocess.

Usage::

    # Filter mode — select series by field:value filters (AND-joined)
    convert_to_nii.py METADATA_JSON OUTPUT_DIR TAG key:value [key:value ...]

    # Interactive mode — browse studies and series interactively
    convert_to_nii.py METADATA_JSON OUTPUT_DIR TAG

    # On Windows, prefix with `python ` and use double quotes:
    python convert_to_nii.py metadata.json C:\\data\\specimen_20 postop "series_description:MAZOR BONE"

Arguments:
    METADATA_JSON   JSON produced by survey_dicoms.py
    OUTPUT_DIR      Where the NIfTI output goes (created if absent)
    TAG             Filename prefix for dcm2niix (e.g. "preop", "postop")
    key:value       Filter on series JSON fields (case-insensitive substring).
                    Keys: series_description, protocol, kernel, slice_thickness,
                    rows, columns, ... All filters are AND-joined.

Requires: dcm2niix (on PATH).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


# ── Pure helpers (unit-tested without DICOM data or dcm2niix) ──────────────────

def flatten_series(metadata):
    """Return every series object across all studies, in document order."""
    return [s for study in metadata.get("studies", []) for s in study.get("series", [])]


def parse_filters(tokens):
    """Parse ``key:value`` filter tokens into a list of (key, value) pairs.

    Raises ValueError on a token without a colon or with an empty key.
    """
    filters = []
    for tok in tokens:
        key, sep, val = tok.partition(":")
        if not sep or key == "":
            raise ValueError(
                "Invalid filter: %r (expected key:value, e.g. "
                'series_description:"MAZOR BONE")' % tok)
        filters.append((key, val))
    return filters


def match_series(series_list, filters):
    """Filter series by case-insensitive substring match on each key (AND-joined)."""
    result = list(series_list)
    for key, val in filters:
        needle = val.lower()
        result = [s for s in result if needle in str(s.get(key) or "").lower()]
    return result


def format_series(series):
    """One-line human summary of a series object for selection menus."""
    desc = series.get("series_description") or "(none)"
    if len(desc) > 24:
        desc = desc[:23] + "…"
    kernel = series.get("kernel") or "—"
    thick = series.get("slice_thickness") or "—"
    rows = series.get("rows") or "—"
    cols = series.get("columns") or "—"
    count = series.get("slice_count") or 0
    mar = "  [MAR]" if series.get("mar") else ""
    return '%-26s  %-12s  %smm  %sx%s  ~%s slices%s' % (
        '"%s"' % desc, kernel, thick, rows, cols, count, mar)


def stage_files(file_paths, staging_dir):
    """Materialize ``file_paths`` into ``staging_dir`` for dcm2niix.

    Prefers hard links (fast, no admin on Windows when on the same volume) and
    falls back to copies across volumes. Basename collisions are uniquified.
    Returns ``(staged, missing)`` counts.
    """
    staged = missing = 0
    used = set()
    for path in file_paths:
        if not os.path.isfile(path):
            missing += 1
            continue
        base = os.path.basename(path)
        if base in used:
            stem, ext = os.path.splitext(base)
            i = 1
            while "%s_%d%s" % (stem, i, ext) in used:
                i += 1
            base = "%s_%d%s" % (stem, i, ext)
        used.add(base)
        dst = os.path.join(staging_dir, base)
        try:
            os.link(path, dst)
        except OSError:
            shutil.copy2(path, dst)
        staged += 1
    return staged, missing


def available_values(series_list, key):
    """Sorted unique display values of ``key`` across series (for error help)."""
    seen = set()
    for s in series_list:
        v = s.get(key)
        seen.add("(empty)" if v in (None, "") else str(v))
    return sorted(seen)


# ── Conversion (touches the filesystem / runs dcm2niix) ───────────────────────

def convert_series(series, output_dir, tag, log=print):
    """Stage ``series`` and run dcm2niix into ``output_dir`` as ``<tag>.nii.gz``."""
    desc = series.get("series_description") or "(none)"
    files = series.get("files", [])
    log("")
    log('Selected series: "%s" (%d files)' % (desc, len(files)))

    staging = tempfile.mkdtemp(prefix="dcm2nii_")
    try:
        staged, missing = stage_files(files, staging)
        if missing:
            print("Warning: %d files no longer exist on disk." % missing, file=sys.stderr)
        if staged == 0:
            print("Error: No files could be staged. Check that source files still exist.",
                  file=sys.stderr)
            return 1
        log("Staged %d files." % staged)
        log("Running dcm2niix...\n")
        subprocess.run(["dcm2niix", "-o", output_dir, "-f", tag, "-z", "y", staging],
                       check=True)
        log("\nOutput written to: %s" % os.path.join(output_dir, tag + ".nii.gz"))
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _prompt_choice(prompt, count):
    """Read a 1-based selection from stdin; return 0-based index or None."""
    raw = input(prompt).strip()
    if not raw.isdigit():
        return None
    value = int(raw)
    if value < 1 or value > count:
        return None
    return value - 1


def _run_filter_mode(metadata, filters, output_dir, tag):
    print("Filter mode")
    print("Output:   %s" % output_dir)
    print("Tag:      %s" % tag)
    print("Filters:")
    for key, val in filters:
        print("  %s:%s" % (key, val))
    print("")

    all_series = flatten_series(metadata)
    matches = match_series(all_series, filters)

    if not matches:
        print("Error: No series matched all filters.", file=sys.stderr)
        print("\nAvailable values for each filter key:", file=sys.stderr)
        for key, _val in filters:
            print("  %s:" % key, file=sys.stderr)
            for value in available_values(all_series, key):
                print("    %s" % value, file=sys.stderr)
        return 1

    if len(matches) == 1:
        selected = matches[0]
    else:
        print("Multiple series match (%d):\n" % len(matches))
        for i, series in enumerate(matches):
            print("  %d) %s" % (i + 1, format_series(series)))
        print("")
        idx = _prompt_choice("Select series [1-%d]: " % len(matches), len(matches))
        if idx is None:
            print("Error: Invalid selection.", file=sys.stderr)
            return 1
        selected = matches[idx]

    return convert_series(selected, output_dir, tag)


def _run_interactive_mode(metadata, output_dir, tag):
    print("Interactive mode")
    print("Output:   %s" % output_dir)
    print("Tag:      %s\n" % tag)

    studies = metadata.get("studies", [])
    if not studies:
        print("Error: No studies found in metadata.", file=sys.stderr)
        return 1

    if len(studies) > 1:
        print("Studies:")
        for i, study in enumerate(studies):
            name = study.get("patient_name") or "(unknown)"
            pid = study.get("patient_id") or ""
            label = "%s (%s)" % (name, pid) if pid else name
            desc = study.get("study_description") or "(none)"
            date = study.get("study_date") or "(unknown)"
            if len(date) == 8 and date.isdigit():
                date = "%s-%s-%s" % (date[:4], date[4:6], date[6:])
            print("  %d) %s — %s — %s  (%d series)"
                  % (i + 1, label, desc, date, len(study.get("series", []))))
        print("")
        idx = _prompt_choice("Select study [1-%d]: " % len(studies), len(studies))
        if idx is None:
            print("Error: Invalid selection.", file=sys.stderr)
            return 1
        study = studies[idx]
    else:
        study = studies[0]
        name = study.get("patient_name") or "(unknown)"
        pid = study.get("patient_id") or ""
        label = "%s (%s)" % (name, pid) if pid else name
        print("Study: %s — %s" % (label, study.get("study_description") or "(none)"))

    series = study.get("series", [])
    if not series:
        print("Error: No series in selected study.", file=sys.stderr)
        return 1

    print('\nSeries in "%s":' % (study.get("study_description") or "(none)"))
    for i, s in enumerate(series):
        print("  %d) %s" % (i + 1, format_series(s)))
    print("")
    idx = _prompt_choice("Select series [1-%d]: " % len(series), len(series))
    if idx is None:
        print("Error: Invalid selection.", file=sys.stderr)
        return 1

    return convert_series(series[idx], output_dir, tag)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a DICOM series (selected from survey JSON) to NIfTI.")
    parser.add_argument("metadata_json", help="JSON produced by survey_dicoms.py")
    parser.add_argument("output_dir", help="Directory for the NIfTI output")
    parser.add_argument("tag", help='Filename prefix for dcm2niix (e.g. "preop")')
    parser.add_argument("filters", nargs="*", metavar="key:value",
                        help="Series field filters; omit for interactive mode")
    args = parser.parse_args(argv)

    if shutil.which("dcm2niix") is None:
        print("Error: dcm2niix not found. Install it "
              "(e.g. conda install -c conda-forge dcm2niix).", file=sys.stderr)
        return 1
    if not os.path.isfile(args.metadata_json):
        print("Error: Metadata file not found: %s" % args.metadata_json, file=sys.stderr)
        return 1
    try:
        with open(args.metadata_json) as fh:
            metadata = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print("Error: Invalid JSON: %s (%s)" % (args.metadata_json, exc), file=sys.stderr)
        return 1

    try:
        filters = parse_filters(args.filters)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if filters:
        return _run_filter_mode(metadata, filters, args.output_dir, args.tag)
    return _run_interactive_mode(metadata, args.output_dir, args.tag)


if __name__ == "__main__":
    raise SystemExit(main())
