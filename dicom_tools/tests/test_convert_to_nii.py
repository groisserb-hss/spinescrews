"""Tests for convert_to_nii.py (the stdlib/pydicom port of convert_to_nii.sh).

Runnable two ways:
    python dicom_tools/tests/test_convert_to_nii.py     # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                            # if pytest installed

Tiers:
  0  pure selection/staging logic   -- always runs (no DICOM, no dcm2niix)
  1  end-to-end survey + convert     -- needs SCREWS_TEST_DICOM_DIR and dcm2niix
"""

import importlib.util
import os
import shutil
import tempfile

# ---- import the tool module by path (works under pytest and plain python) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.normpath(os.path.join(_HERE, "..", "convert_to_nii.py"))
_spec = importlib.util.spec_from_file_location("convert_to_nii", _TOOL)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

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


_SERIES = [
    {"series_description": "MAZOR BONE", "kernel": "BONE", "slice_count": 200,
     "slice_thickness": "0.625", "rows": "512", "columns": "512", "mar": False,
     "files": ["/x/1.dcm", "/x/2.dcm"]},
    {"series_description": "STANDARD", "kernel": "STANDARD", "slice_count": 100,
     "mar": False, "files": []},
    {"series_description": "BONE iMAR", "kernel": "BONEPLUS", "slice_count": 200,
     "mar": True, "files": []},
]
_META = {"studies": [{"series": _SERIES[:2]}, {"series": _SERIES[2:]}]}


# ===========================================================================
# Tier 0 -- pure logic (always runs)
# ===========================================================================

def test_parse_filters():
    assert conv.parse_filters(["series_description:MAZOR BONE", "kernel:bone"]) == \
        [("series_description", "MAZOR BONE"), ("kernel", "bone")]
    assert conv.parse_filters(["kernel:"]) == [("kernel", "")]   # empty value is allowed
    for bad in ["nocolon", ":value", ""]:
        try:
            conv.parse_filters([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_match_series():
    assert len(conv.match_series(_SERIES, [("series_description", "mazor")])) == 1
    # "bone" is a substring of BONE and of BONEPLUS -> two matches
    assert len(conv.match_series(_SERIES, [("kernel", "bone")])) == 2
    # AND-joined filters narrow to one
    assert conv.match_series(_SERIES, [("series_description", "bone"),
                                       ("kernel", "plus")]) == [_SERIES[2]]
    assert conv.match_series(_SERIES, [("series_description", "zzz")]) == []
    # empty value matches everything
    assert len(conv.match_series(_SERIES, [("series_description", "")])) == 3


def test_flatten_series():
    assert conv.flatten_series(_META) == _SERIES
    assert conv.flatten_series({}) == []


def test_format_series():
    line = conv.format_series(_SERIES[0])
    assert "MAZOR BONE" in line and "200 slices" in line and "[MAR]" not in line
    assert "[MAR]" in conv.format_series(_SERIES[2])
    # missing optional fields must not raise
    conv.format_series({"series_description": "x" * 40})        # long desc -> truncated
    conv.format_series({})


def test_available_values():
    vals = conv.available_values(
        [{"kernel": "BONE"}, {"kernel": "BONE"}, {"kernel": ""}, {}], "kernel")
    assert vals == ["(empty)", "BONE"]


def test_stage_files():
    src = tempfile.mkdtemp(prefix="conv_src_")
    dst = tempfile.mkdtemp(prefix="conv_dst_")
    try:
        # Two real files with the SAME basename in different subdirs (collision),
        # plus one path that does not exist (missing).
        os.makedirs(os.path.join(src, "d1"))
        os.makedirs(os.path.join(src, "d2"))
        p1 = os.path.join(src, "d1", "IM001")
        p2 = os.path.join(src, "d2", "IM001")
        for p in (p1, p2):
            with open(p, "wb") as f:
                f.write(b"x")
        staged, missing = conv.stage_files(
            [p1, p2, os.path.join(src, "gone", "IM999")], dst)
        assert staged == 2 and missing == 1
        assert len(os.listdir(dst)) == 2     # collision uniquified, both present
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)


# ===========================================================================
# Tier 1 -- end-to-end survey + convert (needs SCREWS_TEST_DICOM_DIR + dcm2niix)
# ===========================================================================

def test_convert_real_series():
    d = os.environ.get("SCREWS_TEST_DICOM_DIR")
    if not d or not os.path.isdir(d):
        skip("set SCREWS_TEST_DICOM_DIR to a CT DICOM series folder")
    if shutil.which("dcm2niix") is None:
        skip("dcm2niix not on PATH")

    # survey the directory, then convert its first (largest) series
    survey_path = os.path.normpath(os.path.join(_HERE, "..", "survey_dicoms.py"))
    sspec = importlib.util.spec_from_file_location("survey_dicoms", survey_path)
    survey = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(survey)

    meta = survey.survey([d])
    series = max(conv.flatten_series(meta), key=lambda s: s["slice_count"])
    out = tempfile.mkdtemp(prefix="conv_t1_")
    try:
        rc = conv.convert_series(series, out, "preop", log=lambda *_a: None)
        assert rc == 0
        assert os.path.isfile(os.path.join(out, "preop.nii.gz"))
    finally:
        shutil.rmtree(out, ignore_errors=True)


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
