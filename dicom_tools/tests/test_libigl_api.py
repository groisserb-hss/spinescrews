"""Drift guard: the installed libigl must expose the 2.5.x API the pipeline uses.

spinescrews/pyproject.toml and bg3dtools/pyproject.toml pin ``libigl==2.5.1``
specifically because the 2.6.x line changed this surface in ways that break the
pipeline — and some breaks are SILENT (changed return arity consumed positionally
via ``[1:3]``). Concretely, against a unit tetrahedron:

  - igl.extract_manifold_patches  was REMOVED in 2.6.x        (vertebrae.py)
  - igl.qslim                     5-tuple (2.5) -> 4-tuple    (qslim(...)[1:3])
  - igl.decimate                  5-tuple (2.5) -> 4-tuple    (decimate(...)[1:3])
  - igl.signed_distance           3-tuple (2.5) -> 4-tuple    (s,_,_ = ...)

This test fails loudly if libigl is upgraded past the supported API, pointing at
the pin. It is the executable form of why the pin exists.

Runnable two ways:
    python dicom_tools/tests/test_libigl_api.py     # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                        # if pytest installed

Runs in the screws310 env (where libigl is installed); needs no data. Skips if
libigl is not importable in the current interpreter.
"""

import numpy as np

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


# Closed manifold (tetrahedron) + interior/exterior query points.
_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
_Q = np.array([[0.25, 0.25, 0.25], [2.0, 2.0, 2.0]], dtype=np.float64)

_PIN_HINT = ("libigl has drifted from the 2.5.x API spinescrews targets; the pin "
             "is `libigl==2.5.1` in spinescrews/ and bg3dtools/ pyproject.toml")


def _igl():
    try:
        import igl
    except Exception as e:
        skip(f"libigl not importable in this interpreter: {e}")
    return igl


def test_required_functions_present():
    igl = _igl()
    for name in ["read_triangle_mesh", "write_triangle_mesh", "barycenter",
                 "signed_distance", "point_mesh_squared_distance", "winding_number",
                 "qslim", "decimate", "doublearea", "extract_manifold_patches",
                 "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER"]:
        assert hasattr(igl, name), f"igl.{name} missing — {_PIN_HINT}"


def test_return_arities_match_pipeline_expectations():
    igl = _igl()
    # qslim / decimate: pipeline unpacks 5 values and indexes [1:3] for (V, F)
    assert len(igl.qslim(_V, _F, 2)) == 5, f"qslim arity changed — {_PIN_HINT}"
    assert len(igl.decimate(_V, _F, 2)) == 5, f"decimate arity changed — {_PIN_HINT}"
    # signed_distance: pipeline unpacks exactly `s, _, _`
    sd = igl.signed_distance(_Q, _V, _F,
                             sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
    assert len(sd) == 3, f"signed_distance arity changed — {_PIN_HINT}"
    # point_mesh_squared_distance: pipeline reads [0] and unpacks (d2, _, surfpts)
    assert len(igl.point_mesh_squared_distance(_Q, _V, _F)) == 3, \
        f"point_mesh_squared_distance arity changed — {_PIN_HINT}"


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
