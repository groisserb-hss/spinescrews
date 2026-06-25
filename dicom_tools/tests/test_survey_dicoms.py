"""Tests for survey_dicoms.py (the pydicom port of survey_dicoms.sh).

Runnable two ways:
    python dicom_tools/tests/test_survey_dicoms.py      # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                            # if pytest installed

Tiers:
  0  pure logic + synthetic DICOMs written to a temp dir -- always runs (pydicom
     is a hard dependency, so no real data is needed)
  1  survey a real series                                -- needs SCREWS_TEST_DICOM_DIR

No DICOM data is committed; tier-0 files are generated in a system temp dir at
run time and tier-1 reads a series you point it at.
"""

import importlib.util
import json
import os
import tempfile
import shutil

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, CTImageStorage,
                         PYDICOM_IMPLEMENTATION_UID)

# ---- import the tool module by path (works under pytest and plain python) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.normpath(os.path.join(_HERE, "..", "survey_dicoms.py"))
_spec = importlib.util.spec_from_file_location("survey_dicoms", _TOOL)
survey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(survey)

# ---- skip mechanism compatible with pytest and the plain runner ----
try:
    import pytest
    _SKIP_EXC = pytest.skip.Exception

    def skip(msg):
        pytest.skip(msg)
except Exception:                       # pytest not installed
    class _SkipExc(Exception):
        pass
    _SKIP_EXC = _SkipExc

    def skip(msg):
        raise _SkipExc(msg)


def _meta(**overrides):
    """A metadata dict shaped like extract_metadata()'s output, with overrides."""
    base = {f: ("" if f in survey._STRING_FIELDS else None)
            for f in survey.TAG if f not in ("study_uid", "series_uid")}
    base.update(overrides)
    return base


def _write_dicom(path, study_uid, series_uid, sop_uid, **tags):
    """Write a minimal, standards-compliant CT DICOM file for testing."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = sop_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds = FileDataset(path, {}, file_meta=fm, preamble=b"\x00" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def _require_dicom():
    d = os.environ.get("SCREWS_TEST_DICOM_DIR")
    if not d:
        skip("set SCREWS_TEST_DICOM_DIR to a CT DICOM series folder to run DICOM tests")
    if not os.path.isdir(d):
        skip(f"SCREWS_TEST_DICOM_DIR is not a directory: {d}")
    return d


# ===========================================================================
# Tier 0 -- pure logic (always runs)
# ===========================================================================

def test_detect_mar():
    yes = [
        ("BONE iMAR", "STANDARD"),     # vendor keyword in description
        ("OMAR", ""),
        ("O-MAR recon", ""),
        ("SEMAR head", ""),
        ("BONEPLUS", ""),              # vendor keyword (also matches kernel rule)
        ("MAZOR BONE MAR", "BONE"),    # whole-word MAR in description
        ("STANDARD", "BONEPLUS"),      # BONEPLUS kernel
    ]
    no = [
        ("STANDARD", "STANDARD"),
        ("MARROW STD", ""),            # MAR is not a whole word in MARROW
        ("SMART", ""),                 # MAR is a substring but not a whole word
        ("", ""),
        (None, None),
    ]
    for desc, kern in yes:
        assert survey.detect_mar(desc, kern) is True, (desc, kern)
    for desc, kern in no:
        assert survey.detect_mar(desc, kern) is False, (desc, kern)


def test_should_scan():
    for name in ["IM0001", "1.2.3.dcm", "0001.IMA", "image"]:
        assert survey.should_scan(name) is True, name
    for name in ["DICOMDIR", ".DS_Store", "README", "README.md",
                 "notes.txt", "meta.json", "list.csv", "doc.PDF", "x.xml"]:
        assert survey.should_scan(name) is False, name


def test_extract_metadata_from_dataset():
    ds = pydicom.Dataset()
    ds.SeriesDescription = "BONE iMAR"
    ds.ConvolutionKernel = "BONEPLUS"
    ds.SliceThickness = "0.625"
    ds.Rows = 512
    ds.PixelSpacing = ["0.5", "0.5"]
    ds.Manufacturer = "GE"
    m = survey.extract_metadata(ds)
    assert m["series_description"] == "BONE iMAR"
    assert m["kernel"] == "BONEPLUS"
    assert m["slice_thickness"] == "0.625"
    assert m["rows"] == "512"                 # numeric -> string form
    assert m["pixel_spacing"] == "0.5\\0.5"   # multi-valued joined with backslash
    assert m["manufacturer"] == "GE"
    assert m["columns"] is None               # absent parameter field -> None
    assert m["kvp"] is None
    assert m["protocol"] == ""                # absent string field -> ""


def test_group_into_studies():
    records = [
        ("1.2.3.1", "1.2.3.1.1", "/a/3.dcm"),
        ("1.2.3.1", "1.2.3.1.1", "/a/1.dcm"),
        ("1.2.3.1", "1.2.3.1.1", "/a/2.dcm"),
        ("1.2.3.1", "1.2.3.1.2", "/a/x.dcm"),
        ("1.2.3.2", "1.2.3.2.1", "/b/y.dcm"),
    ]
    series_meta = {
        ("1.2.3.1", "1.2.3.1.1"): _meta(series_description="BONE iMAR", kernel="BONEPLUS"),
        ("1.2.3.1", "1.2.3.1.2"): _meta(series_description="STD", kernel="STANDARD"),
        ("1.2.3.2", "1.2.3.2.1"): _meta(series_description="MAZOR BONE MAR", kernel="BONE"),
    }
    studies = survey.group_into_studies(records, series_meta)
    assert [s["study_instance_uid"] for s in studies] == ["1.2.3.1", "1.2.3.2"]
    s1 = studies[0]["series"]
    assert len(s1) == 2
    se1 = next(s for s in s1 if s["series_instance_uid"] == "1.2.3.1.1")
    assert se1["slice_count"] == 3
    assert se1["files"] == ["/a/1.dcm", "/a/2.dcm", "/a/3.dcm"]   # sorted by path
    assert se1["mar"] is True
    assert studies[1]["series"][0]["mar"] is True                  # whole-word MAR


def test_build_metadata_summary():
    records = [
        ("1.2.3.1", "1.2.3.1.1", "/a/1.dcm"),
        ("1.2.3.1", "1.2.3.1.2", "/a/2.dcm"),
        ("1.2.3.2", "1.2.3.2.1", "/b/1.dcm"),
    ]
    series_meta = {
        ("1.2.3.1", "1.2.3.1.1"): _meta(series_description="BONE iMAR", kernel="BONEPLUS"),
        ("1.2.3.1", "1.2.3.1.2"): _meta(series_description="STD", kernel="STANDARD"),
        ("1.2.3.2", "1.2.3.2.1"): _meta(series_description="MAR", kernel="BONE"),
    }
    studies = survey.group_into_studies(records, series_meta)
    meta = survey.build_metadata(["/a", "/b"], studies, total_files=9, dicom_count=3)
    summary = meta["summary"]
    assert summary["study_count"] == 2
    assert summary["series_count"] == 3
    assert summary["mar_series_count"] == 2
    assert summary["unique_kernels"] == ["BONE", "BONEPLUS", "STANDARD"]
    assert meta["total_files"] == 9 and meta["dicom_files"] == 3
    json.loads(json.dumps(meta))   # must be JSON-serializable


def test_survey_synthetic_dir():
    tmp = tempfile.mkdtemp(prefix="survey_t0_")
    try:
        # Study A: 2 series (3 + 2 slices); Study B: 1 series (1 slice).
        os.makedirs(os.path.join(tmp, "A"))
        os.makedirs(os.path.join(tmp, "B"))
        for i in range(3):
            _write_dicom(os.path.join(tmp, "A", f"a1_{i}.dcm"),
                         "1.2.3.1", "1.2.3.1.1", f"1.2.3.1.1.{i}",
                         SeriesDescription="BONE iMAR", ConvolutionKernel="BONEPLUS",
                         SliceThickness="0.625", Rows=512, Columns=512,
                         PixelSpacing=["0.5", "0.5"], Manufacturer="GE")
        for i in range(2):
            _write_dicom(os.path.join(tmp, "A", f"a2_{i}.dcm"),
                         "1.2.3.1", "1.2.3.1.2", f"1.2.3.1.2.{i}",
                         SeriesDescription="STANDARD", ConvolutionKernel="STANDARD")
        _write_dicom(os.path.join(tmp, "B", "b1_0.dcm"),
                     "1.2.3.2", "1.2.3.2.1", "1.2.3.2.1.0",
                     SeriesDescription="MAZOR BONE MAR", ConvolutionKernel="BONE")
        # Sidecars that should be skipped (not counted), plus a non-DICOM binary
        # that is scanned but unreadable (counted in total_files, not dicom_files).
        with open(os.path.join(tmp, "README.txt"), "w") as f:
            f.write("notes\n")
        with open(os.path.join(tmp, "list.csv"), "w") as f:
            f.write("a,b\n")
        with open(os.path.join(tmp, "garbage.bin"), "wb") as f:
            f.write(b"not a dicom file")

        meta = survey.survey([tmp])

        assert meta["dicom_files"] == 6
        assert meta["total_files"] == 7           # 6 DICOMs + garbage.bin (sidecars skipped)
        assert meta["summary"]["study_count"] == 2
        assert meta["summary"]["series_count"] == 3
        assert meta["summary"]["mar_series_count"] == 2
        assert meta["summary"]["unique_kernels"] == ["BONE", "BONEPLUS", "STANDARD"]

        a1 = next(s for s in meta["studies"][0]["series"]
                  if s["series_description"] == "BONE iMAR")
        assert a1["slice_count"] == 3 and len(a1["files"]) == 3
        assert all(os.path.isfile(p) for p in a1["files"])
        assert a1["slice_thickness"] == "0.625"
        assert a1["rows"] == "512" and a1["pixel_spacing"] == "0.5\\0.5"
        assert a1["mar"] is True
        a2 = next(s for s in meta["studies"][0]["series"]
                  if s["series_description"] == "STANDARD")
        assert a2["mar"] is False
        json.loads(json.dumps(meta))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# Tier 1 -- survey a real series (needs SCREWS_TEST_DICOM_DIR)
# ===========================================================================

def test_survey_real_series():
    d = _require_dicom()
    meta = survey.survey([d])
    assert meta["dicom_files"] > 0, "no DICOM instances found in SCREWS_TEST_DICOM_DIR"
    assert meta["summary"]["series_count"] >= 1
    json.loads(json.dumps(meta))   # serializable
    for study in meta["studies"]:
        for s in study["series"]:
            assert s["slice_count"] == len(s["files"])
            assert all(os.path.isfile(p) for p in s["files"])


# ===========================================================================
# Plain-python runner
# ===========================================================================

def _main():
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    npass = nskip = nfail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            npass += 1
        except _SKIP_EXC as e:
            print(f"SKIP {name}: {e}")
            nskip += 1
        except Exception as e:
            import traceback
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
            nfail += 1
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
