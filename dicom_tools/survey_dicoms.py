#!/usr/bin/env python3
"""Survey DICOM directories and output structured JSON metadata.

Python port of survey_dicoms.sh. Reads DICOM headers with **pydicom** instead of
shelling out to ``dcmdump``, and assembles the report with the standard-library
``json`` module instead of ``jq`` — so it needs no external CLI tools and runs
unchanged on Windows, macOS, and Linux.

Recursively scans directories for DICOM files, groups them by Study > Series,
and reports reconstruction protocols and image parameters as JSON. Designed to
identify MAR vs non-MAR reconstructions, extract dose info, and feed
``convert_to_nii.py`` for series selection and NIfTI conversion.

Usage::

    survey_dicoms.py [-o metadata.json] DIR [DIR ...]
    python survey_dicoms.py [-o metadata.json] DIR [DIR ...]   # Windows

Design notes (carried over from the original):
  - One representative file per series supplies the per-series metadata; series
    tags (kernel, kVp, spacing, ...) are constant across a series' slices, so
    the first file seen is as good as any. The slice count is the exact number
    of files grouped into the series.
  - DICOMDIR files and obvious non-image sidecars (README*, *.txt/.xml/.json/
    .csv/.pdf, .DS_Store) are skipped before reading.
  - MAR detection checks SeriesDescription and ConvolutionKernel for
    vendor-specific keywords (iMAR, OMAR, SEMAR, BONEPLUS, whole-word "MAR").
    ReconstructionAlgorithm (0018,9315) is intentionally NOT used.
  - pydicom reads the top-level dataset, so values come from the main IOD rather
    than nested functional-group sequences (the common case for CT series).

Output schema matches the original survey_dicoms.sh byte-for-byte at the JSON
level, so existing metadata.json files and convert_to_nii remain compatible.

Requires: pydicom (already in the screws310 environment).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import pydicom
from pydicom.multival import MultiValue

# ── DICOM tags read from each file, by (group, element) ───────────────────────
# Order is documentation only; access is by tag number so keyword spelling can't
# silently drift.
TAG = {
    "study_uid":              0x0020000D,  # StudyInstanceUID
    "series_uid":             0x0020000E,  # SeriesInstanceUID
    "patient_name":           0x00100010,  # PatientName
    "patient_id":             0x00100020,  # PatientID
    "study_date":             0x00080020,  # StudyDate
    "study_description":      0x00081030,  # StudyDescription
    "series_description":     0x0008103E,  # SeriesDescription
    "protocol":               0x00181030,  # ProtocolName
    "kernel":                 0x00181210,  # ConvolutionKernel
    "slice_thickness":        0x00180050,  # SliceThickness
    "pixel_spacing":          0x00280030,  # PixelSpacing
    "rows":                   0x00280010,  # Rows
    "columns":                0x00280011,  # Columns
    "kvp":                    0x00180060,  # KVP
    "tube_current":           0x00181151,  # XRayTubeCurrent
    "exposure":               0x00181152,  # Exposure (mAs)
    "ctdi_vol":               0x00189345,  # CTDIvol
    "manufacturer":           0x00080070,  # Manufacturer
    "model":                  0x00081090,  # ManufacturerModelName
    "recon_diameter":         0x00181100,  # ReconstructionDiameter
}
_SPECIFIC_TAGS = list(TAG.values())

# Fields that are emitted as "" when absent (always strings in the JSON), versus
# the numeric/parameter fields that are emitted as null when absent.
_STRING_FIELDS = (
    "patient_name", "patient_id", "study_date", "study_description",
    "manufacturer", "model", "series_description", "protocol", "kernel",
)

# Non-image sidecar files to skip before attempting a DICOM read.
_SKIP_NAMES = {"DICOMDIR", ".DS_Store"}
_SKIP_EXTS = {".txt", ".xml", ".json", ".csv", ".pdf"}

_MAR_VENDOR_RE = re.compile(r"iMAR|OMAR|O-MAR|SEMAR|BONEPLUS", re.IGNORECASE)
_MAR_WORD_RE = re.compile(r"\bMAR\b", re.IGNORECASE)
_BONEPLUS_RE = re.compile(r"BONEPLUS", re.IGNORECASE)


# ── Pure helpers (unit-tested without any DICOM data) ─────────────────────────

def should_scan(name):
    """Return True if a filename should be read as a candidate DICOM file."""
    if name in _SKIP_NAMES or name.startswith("README"):
        return False
    return os.path.splitext(name)[1].lower() not in _SKIP_EXTS


def detect_mar(series_description, kernel):
    """Classify a series as metal-artifact-reduced from its description/kernel.

    Mirrors survey_dicoms.sh: vendor keyword in the description, OR a whole-word
    "MAR" in the description, OR a BONEPLUS kernel.
    """
    desc = series_description or ""
    if _MAR_VENDOR_RE.search(desc) or _MAR_WORD_RE.search(desc):
        return True
    return bool(_BONEPLUS_RE.search(kernel or ""))


def _tag_str(value):
    """Render a pydicom element value as the survey's textual form, or None.

    Multi-valued elements (e.g. PixelSpacing) are joined with the DICOM value
    delimiter ``\\`` to match dcmdump's bracket content. Empty/absent -> None.
    """
    if value is None:
        return None
    if isinstance(value, MultiValue):
        s = "\\".join(str(v) for v in value)
    else:
        s = str(value)
    s = s.strip()
    return s or None


def extract_metadata(ds):
    """Pull the surveyed tags from a (pydicom) dataset into a flat dict.

    String fields collapse missing -> "" ; parameter fields collapse missing ->
    None. UIDs are returned separately by the caller.
    """
    out = {}
    for field, tag in TAG.items():
        if field in ("study_uid", "series_uid"):
            continue
        raw = ds[tag].value if tag in ds else None
        val = _tag_str(raw)
        out[field] = (val or "") if field in _STRING_FIELDS else val
    return out


def _series_object(series_uid, meta, files):
    return {
        "series_instance_uid": series_uid,
        "series_description": meta["series_description"],
        "protocol": meta["protocol"],
        "kernel": meta["kernel"],
        "mar": detect_mar(meta["series_description"], meta["kernel"]),
        "slice_thickness": meta["slice_thickness"],
        "pixel_spacing": meta["pixel_spacing"],
        "rows": meta["rows"],
        "columns": meta["columns"],
        "slice_count": len(files),
        "recon_diameter": meta["recon_diameter"],
        "kvp": meta["kvp"],
        "tube_current": meta["tube_current"],
        "exposure": meta["exposure"],
        "ctdi_vol": meta["ctdi_vol"],
        "files": files,
    }


def _study_object(study_uid, meta, series_list):
    return {
        "study_instance_uid": study_uid,
        "patient_name": meta["patient_name"],
        "patient_id": meta["patient_id"],
        "study_date": meta["study_date"],
        "study_description": meta["study_description"],
        "manufacturer": meta["manufacturer"],
        "model": meta["model"],
        "series": series_list,
    }


def group_into_studies(records, series_meta):
    """Group ``(study_uid, series_uid, path)`` records into the studies array.

    Studies and series are ordered by UID; files within a series by path. The
    study-level fields are taken from the study's first series (by UID).
    """
    studies = {}
    for study_uid, series_uid, path in records:
        studies.setdefault(study_uid, {}).setdefault(series_uid, []).append(path)

    out = []
    for study_uid in sorted(studies):
        series_map = studies[study_uid]
        series_list = [
            _series_object(series_uid, series_meta[(study_uid, series_uid)],
                           sorted(series_map[series_uid]))
            for series_uid in sorted(series_map)
        ]
        first_meta = series_meta[(study_uid, sorted(series_map)[0])]
        out.append(_study_object(study_uid, first_meta, series_list))
    return out


def build_metadata(directories, studies, total_files, dicom_count):
    """Assemble the top-level report dict from grouped studies."""
    all_series = [s for study in studies for s in study["series"]]
    kernels = sorted({s["kernel"] for s in all_series if s["kernel"]})
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "directories": list(directories),
        "total_files": total_files,
        "dicom_files": dicom_count,
        "studies": studies,
        "summary": {
            "study_count": len(studies),
            "series_count": len(all_series),
            "unique_kernels": kernels,
            "mar_series_count": sum(1 for s in all_series if s["mar"]),
        },
    }


# ── Scanning (touches the filesystem) ─────────────────────────────────────────

def _read_tags(path):
    """Read the surveyed tags from a file, or None if it is not readable DICOM."""
    try:
        return pydicom.dcmread(path, stop_before_pixels=True,
                               specific_tags=_SPECIFIC_TAGS, force=False)
    except Exception:
        # Any non-DICOM / unreadable file is silently skipped, as dcmdump did.
        return None


def survey(directories, log=None):
    """Scan ``directories`` and return the metadata report dict."""
    log = log or (lambda _msg: None)
    records = []                 # (study_uid, series_uid, path)
    series_meta = {}             # (study_uid, series_uid) -> metadata dict
    total_files = 0

    for directory in directories:
        if not os.path.isdir(directory):
            log("Warning: '%s' is not a directory, skipping." % directory)
            continue
        log("Scanning: %s" % directory)

        dir_files = 0
        dir_series_before = set(series_meta)
        dir_dicom = 0
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if not should_scan(name):
                    continue
                dir_files += 1
                total_files += 1
                path = os.path.join(root, name)
                ds = _read_tags(path)
                if ds is None:
                    continue
                study_uid = ds[TAG["study_uid"]].value if TAG["study_uid"] in ds else None
                series_uid = ds[TAG["series_uid"]].value if TAG["series_uid"] in ds else None
                if not study_uid or not series_uid:
                    continue
                study_uid, series_uid = str(study_uid), str(series_uid)
                records.append((study_uid, series_uid, path))
                dir_dicom += 1
                key = (study_uid, series_uid)
                if key not in series_meta:
                    series_meta[key] = extract_metadata(ds)

        n_new_series = len(set(series_meta) - dir_series_before)
        log("  Found %d series in %d DICOM files (of %d total)"
            % (n_new_series, dir_dicom, dir_files))

    log("Total: %d files, %d DICOM instances." % (total_files, len(records)))
    studies = group_into_studies(records, series_meta)
    return build_metadata(directories, studies, total_files, len(records))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Survey DICOM directories and report studies/series as JSON.")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write JSON to FILE (default: stdout)")
    parser.add_argument("dirs", nargs="+", metavar="DIR",
                        help="One or more directories to scan recursively")
    args = parser.parse_args(argv)

    metadata = survey(args.dirs, log=lambda m: print(m, file=sys.stderr))

    if metadata["dicom_files"] == 0:
        print("No DICOM files found. Check that the directories contain readable "
              "DICOM files.", file=sys.stderr)
        return 1

    text = json.dumps(metadata, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
        print("JSON written to: %s" % args.output, file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
