"""Validation harness for burn_screw_endpoints.py (SimpleITK -> pydicom migration).

Runnable two ways:
    python slicer_tools/tests/test_burn_screw_endpoints.py      # prints PASS/SKIP/FAIL
    pytest slicer_tools/tests/                                   # if pytest installed

Tiers:
  0  pure math, no DICOM           -- always runs
  1  pydicom write round-trip      -- needs SCREWS_TEST_DICOM_DIR=<a CT series folder>
  2  cross-backend equivalence     -- additionally needs SCREWS_TEST_SITK=1 (SimpleITK installed)

No DICOM data is committed; tiers 1-2 read a series you point them at and write only to a
system temp dir. Equivalence is defined at the re-read-HU + geometry + burned-voxel level
(the new pydicom backend intentionally emits standards-correct int16+rescale, which differs
on disk from SimpleITK's bytes but re-reads to identical HU).
"""

import importlib.util
import os
import shutil
import tempfile

import numpy as np

# ---- import the tool module by path (works under pytest and plain python) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.normpath(os.path.join(_HERE, "..", "burn_screw_endpoints.py"))
_spec = importlib.util.spec_from_file_location("burn_screw_endpoints", _TOOL)
burn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(burn)

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


RADIUS_MM = 2.0
BURN_HU = 3000


def _require_dicom():
    d = os.environ.get("SCREWS_TEST_DICOM_DIR")
    if not d:
        skip("set SCREWS_TEST_DICOM_DIR to a CT DICOM series folder to run DICOM tests")
    if not os.path.isdir(d):
        skip(f"SCREWS_TEST_DICOM_DIR is not a directory: {d}")
    return d


def _require_sitk():
    if os.environ.get("SCREWS_TEST_SITK") != "1":
        skip("set SCREWS_TEST_SITK=1 (with SimpleITK installed) to run cross-backend tests")
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        skip("SimpleITK not installed")


def _rotation(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _interior_points_lps(geom, fracs):
    nk, nj, ni = geom.shape_zyx
    return [geom.index_to_physical((fi * (ni - 1), fj * (nj - 1), fk * (nk - 1)))
            for (fi, fj, fk) in fracs]


def _rows_from_lps(points):
    """Pair points into entry/tip lines, expressed as RAS (the CSV's coordinate frame)."""
    rows = []
    for m in range(0, len(points) - 1, 2):
        e = burn.ras_to_lps(points[m])      # ras_to_lps is an involution: LPS<->RAS
        t = burn.ras_to_lps(points[m + 1])
        rows.append({"line_name": f"T{m//2 + 1}L", "entry_ras": e, "tip_ras": t})
    return rows


# ===========================================================================
# Tier 0 -- pure math (always runs)
# ===========================================================================

def test_affine_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        geom = burn.Geometry(
            origin=rng.standard_normal(3) * 100,
            spacing=rng.uniform(0.2, 2.0, 3),
            direction=_rotation(rng),
            shape_zyx=(30, 40, 50),
        )
        for ijk in [(0, 0, 0), (49, 0, 0), (0, 39, 0), (0, 0, 29), (13, 21, 7), (49, 39, 29)]:
            p = geom.index_to_physical(ijk)
            back = geom.physical_to_continuous_index(p)
            assert np.allclose(back, ijk, atol=1e-7), (ijk, back)


def test_ij_axis_pairing():
    # Anisotropic in-plane spacing + non-square shape: a swapped i/j (or swapped
    # PixelSpacing) pairing fails here. spacing=(col, row, slice)=(0.5, 0.7, 1.25).
    geom = burn.Geometry(
        origin=np.array([10.0, -5.0, 3.0]),
        spacing=np.array([0.5, 0.7, 1.25]),
        direction=np.eye(3),
        shape_zyx=(20, 64, 48),
    )
    o = geom.index_to_physical((0, 0, 0))
    assert np.allclose(geom.index_to_physical((1, 0, 0)) - o, [0.5, 0, 0])   # i -> X * spacing[0]
    assert np.allclose(geom.index_to_physical((0, 1, 0)) - o, [0, 0.7, 0])   # j -> Y * spacing[1]
    assert np.allclose(geom.index_to_physical((0, 0, 1)) - o, [0, 0, 1.25])  # k -> normal * spacing[2]


def test_sphere_burn_matches_analytic():
    geom = burn.Geometry(
        origin=np.array([-12.3, 7.1, 40.0]),
        spacing=np.array([0.5, 0.7, 1.25]),
        direction=_rotation(np.random.default_rng(3)),
        shape_zyx=(24, 40, 30),
    )
    hu = np.zeros(geom.shape_zyx, dtype=np.int32)
    center = geom.index_to_physical((15.0, 20.0, 12.0))   # interior, non-grid
    r = 3.0
    touched = burn.burn_sphere_into_array(hu, geom, center, r, BURN_HU)

    nk, nj, ni = geom.shape_zyx
    kk, jj, ii = np.mgrid[0:nk, 0:nj, 0:ni]
    pts = (geom.origin[:, None, None, None]
           + np.tensordot(geom.direction, np.stack([ii, jj, kk]) * geom.spacing[:, None, None, None], axes=([1], [0])))
    d2 = ((pts - np.asarray(center)[:, None, None, None]) ** 2).sum(0)
    analytic = d2 <= r * r

    assert touched == int(analytic.sum())
    assert np.array_equal(hu == BURN_HU, analytic)


def test_edge_cases_no_crash():
    geom = burn.Geometry(
        origin=np.zeros(3), spacing=np.array([0.5, 0.5, 1.0]),
        direction=np.eye(3), shape_zyx=(10, 20, 20),
    )
    base = np.zeros(geom.shape_zyx, dtype=np.int32)

    # point well outside -> 0 voxels, array untouched
    hu = base.copy()
    assert burn.burn_sphere_into_array(hu, geom, [1e4, 1e4, 1e4], 2.0, BURN_HU) == 0
    assert not hu.any()

    # corner + sub-voxel radius -> no out-of-bounds, no exception
    hu = base.copy()
    corner = geom.index_to_physical((0, 0, 0))
    n = burn.burn_sphere_into_array(hu, geom, corner, 0.1, BURN_HU)
    assert n >= 0 and hu[hu == BURN_HU].size == n

    # giant radius clamped to volume, no exception
    hu = base.copy()
    n = burn.burn_sphere_into_array(hu, geom, geom.index_to_physical((10, 10, 5)), 500.0, BURN_HU)
    assert n == int((hu == BURN_HU).sum()) and n <= hu.size


# ===========================================================================
# Tier 1 -- pydicom write round-trip (needs SCREWS_TEST_DICOM_DIR)
# ===========================================================================

def test_pydicom_roundtrip():
    d = _require_dicom()
    be = burn.PydicomBackend()
    sid = be.choose_series_id(d, None)
    vol = be.read_series(d, sid)

    pts = _interior_points_lps(vol.geometry, [(0.5, 0.5, 0.5), (0.35, 0.45, 0.6),
                                              (0.6, 0.55, 0.4), (0.45, 0.3, 0.7),
                                              (0.3, 0.6, 0.35), (0.65, 0.4, 0.55)])
    rows = _rows_from_lps(pts)

    # also exercise parse_points_csv end-to-end
    tmp = tempfile.mkdtemp(prefix="burn_t1_")
    try:
        csv_path = os.path.join(tmp, "plan.csv")
        with open(csv_path, "w", newline="") as f:
            f.write("line_name,entry_ras_x,entry_ras_y,entry_ras_z,tip_ras_x,tip_ras_y,tip_ras_z\n")
            for r in rows:
                e, t = r["entry_ras"], r["tip_ras"]
                f.write(f"{r['line_name']},{e[0]},{e[1]},{e[2]},{t[0]},{t[1]},{t[2]}\n")
        parsed = burn.parse_points_csv(csv_path)
        assert len(parsed) == len(rows)

        burned, summary, total = burn.burn_all_points(vol, parsed, RADIUS_MM, BURN_HU)
        assert total > 0

        out = os.path.join(tmp, "out")
        be.write_series(burned, out)

        reread = be.read_series(out, be.choose_series_id(out, None))

        assert reread.hu_zyx.shape == burned.hu_zyx.shape
        assert np.array_equal(reread.hu_zyx, burned.hu_zyx), "re-read HU != burned HU"
        assert np.allclose(reread.geometry.origin, burned.geometry.origin, atol=1e-4)
        assert np.allclose(reread.geometry.spacing, burned.geometry.spacing, atol=1e-6)
        assert np.allclose(reread.geometry.direction, burned.geometry.direction, atol=1e-6)
        assert int((reread.hu_zyx == BURN_HU).sum()) == int((burned.hu_zyx == BURN_HU).sum())

        # DICOM identity/metadata checks on the written slices
        slices = reread.source_slices
        src_series = {str(s.SeriesInstanceUID) for s in vol.source_slices}
        sops, series_uids = set(), set()
        for k, ds in enumerate(slices):
            assert list(ds.ImageType[:2]) == ["DERIVED", "SECONDARY"]
            assert int(ds.InstanceNumber) == k + 1
            series_uids.add(str(ds.SeriesInstanceUID))
            sops.add(str(ds.SOPInstanceUID))
            expect_ipp = reread.geometry.index_to_physical((0, 0, k))
            assert np.allclose(np.asarray(ds.ImagePositionPatient, float), expect_ipp, atol=1e-4)
        assert len(series_uids) == 1 and series_uids.isdisjoint(src_series)
        assert len(sops) == len(slices), "SOPInstanceUIDs not unique"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# Tier 2 -- cross-backend equivalence vs current SimpleITK behavior
# ===========================================================================

def test_cross_backend_equivalence():
    d = _require_dicom()
    _require_sitk()
    import SimpleITK as sitk

    pyd, sk = burn.PydicomBackend(), burn.SimpleitkBackend()
    sid_p = pyd.choose_series_id(d, None)
    sid_s = sk.choose_series_id(d, None)
    vol_p = pyd.read_series(d, sid_p)
    vol_s = sk.read_series(d, sid_s)

    # (a) series file selection matches sitk (guards the 794-vs-793 drop)
    pyd_files = {os.path.basename(s.filename) for s in vol_p.source_slices}
    sitk_files = {os.path.basename(f) for f in vol_s.source_slices[1]}
    assert pyd_files == sitk_files, (
        f"file-set mismatch: pydicom-only={sorted(pyd_files - sitk_files)}, "
        f"sitk-only={sorted(sitk_files - pyd_files)}"
    )

    # (b) geometry matches
    assert vol_p.geometry.shape_zyx == vol_s.geometry.shape_zyx
    assert np.allclose(vol_p.geometry.origin, vol_s.geometry.origin, atol=1e-4)
    assert np.allclose(vol_p.geometry.spacing, vol_s.geometry.spacing, atol=1e-6)
    assert np.allclose(vol_p.geometry.direction, vol_s.geometry.direction, atol=1e-6)

    # (c) index->physical matches the actual sitk transform at corners + random
    img = sitk.ReadImage(vol_s.source_slices[1])
    nk, nj, ni = vol_p.geometry.shape_zyx
    rng = np.random.default_rng(7)
    idxs = [(0, 0, 0), (ni - 1, 0, 0), (0, nj - 1, 0), (0, 0, nk - 1), (ni - 1, nj - 1, nk - 1)]
    idxs += [tuple(int(v) for v in rng.integers([ni, nj, nk])) for _ in range(50)]
    max_err = max(np.linalg.norm(np.asarray(img.TransformIndexToPhysicalPoint(ijk))
                                 - vol_p.geometry.index_to_physical(ijk)) for ijk in idxs)
    assert max_err < 1e-4, f"index->physical max error {max_err} mm"

    # (d) in-memory HU identical
    assert np.array_equal(vol_p.hu_zyx, vol_s.hu_zyx), "in-memory HU differs from SimpleITK"

    # (e) identical burn -> identical burned-voxel set + per-endpoint counts
    pts = _interior_points_lps(vol_p.geometry, [(0.5, 0.5, 0.5), (0.35, 0.45, 0.6),
                                                (0.6, 0.55, 0.4), (0.45, 0.3, 0.7)])
    pts.append(np.array([1e5, 1e5, 1e5]))   # degenerate (outside) -> 0 both backends
    pts.append(vol_p.geometry.index_to_physical((0.3 * ni, 0.6 * nj, 0.35 * nk)))
    rows = _rows_from_lps(pts)

    bp, sp, _ = burn.burn_all_points(vol_p, rows, RADIUS_MM, BURN_HU)
    bs, ss, _ = burn.burn_all_points(vol_s, rows, RADIUS_MM, BURN_HU)
    assert np.array_equal(bp.hu_zyx == BURN_HU, bs.hu_zyx == BURN_HU), "burned-voxel sets differ"
    assert [x["voxels_burned"] for x in sp] == [x["voxels_burned"] for x in ss]

    # (f) re-read HU of both written outputs identical (bytes differ by design)
    tmp = tempfile.mkdtemp(prefix="burn_t2_")
    try:
        out_p, out_s = os.path.join(tmp, "pyd"), os.path.join(tmp, "sitk")
        pyd.write_series(bp, out_p)
        sk.write_series(bs, out_s)
        rp = pyd.read_series(out_p, pyd.choose_series_id(out_p, None))
        rs = pyd.read_series(out_s, pyd.choose_series_id(out_s, None))
        assert np.array_equal(rp.hu_zyx, rs.hu_zyx), "re-read HU of pydicom vs sitk output differ"
        assert np.allclose(rp.geometry.origin, rs.geometry.origin, atol=1e-4)
        assert np.allclose(rp.geometry.spacing, rs.geometry.spacing, atol=1e-6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
