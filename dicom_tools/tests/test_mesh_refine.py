"""Geometry tests for spinescrews.tools.mesh_refine.split_long_edges (the erosion unwind's edge refinement).

The unwind relies on four properties: splitting never changes the genus (the genus-1 lock is the point of the
erosion), the mesh stays closed and consistently oriented, the surface does not move (the next nonrigid ICP call
resumes from exactly where the last stopped), and the rest map places every vertex on the starting mesh with the
original vertices untouched (align_preop's correspondence chain is built on it).

Runnable two ways:
    python dicom_tools/tests/test_mesh_refine.py     # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                         # if pytest installed

Runs in the screws310 env (where spinescrews is installed); needs no data.
"""

import numpy as np
from scipy import sparse

from spinescrews.tools.mesh_refine import mesh_edges, euler_characteristic, split_long_edges

# ---- skip mechanism compatible with pytest and the plain runner ----
try:
    import pytest
    _SKIP_EXC = pytest.skip.Exception
except Exception:                       # pytest not installed
    class _SkipExc(Exception):
        pass
    _SKIP_EXC = _SkipExc


def _torus(nu=24, nv=12, R=2.0, r=1.0):
    """A closed, consistently oriented genus-1 triangle mesh."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False),
                       np.linspace(0, 2 * np.pi, nv, endpoint=False), indexing='ij')
    V = np.column_stack([((R + r * np.cos(v)) * np.cos(u)).ravel(), ((R + r * np.cos(v)) * np.sin(u)).ravel(),
                         (r * np.sin(v)).ravel()])
    idx = np.arange(nu * nv).reshape(nu, nv)
    i0, j0 = idx, np.roll(idx, -1, 0)
    i1, j1 = np.roll(idx, -1, 1), np.roll(np.roll(idx, -1, 0), -1, 1)
    F = np.vstack([np.column_stack([i0.ravel(), j0.ravel(), j1.ravel()]),
                   np.column_stack([i0.ravel(), j1.ravel(), i1.ravel()])])
    return V, F


def _split_stretched(max_len=0.6):
    V0, F0 = _torus()
    Vs = V0 * np.array([6.0, 1.0, 1.0])            # stretched along x: those edges ~6x longer
    V, F, R, n = split_long_edges(Vs, F0, sparse.identity(len(V0), format='csr'), max_len)
    return V0, F0, Vs, V, F, R, n


def test_genus_and_manifold_kept():
    _, F0, _, V, F, _, n = _split_stretched()
    assert n > 0
    assert euler_characteristic(F) == euler_characteristic(F0) == 0, 'genus changed'
    _, fe = mesh_edges(F)
    assert np.all(np.bincount(fe.ravel()) == 2), 'mesh is no longer closed and manifold'


def test_orientation_consistent():
    _, _, _, _, F, _, _ = _split_stretched()
    half_edges = np.stack([F, np.roll(F, -1, axis=1)], axis=2).reshape(-1, 2)
    assert len(np.unique(half_edges, axis=0)) == len(half_edges), 'a directed edge repeats: orientation broke'


def test_no_long_edge_left():
    _, _, _, V, F, _, _ = _split_stretched(max_len=0.6)
    E, _ = mesh_edges(F)
    assert np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).max() <= 0.6


def test_surface_does_not_move():
    """Every new vertex lies on the pre-split surface: the midpoint of an edge of the stretched mesh."""
    V0, _, Vs, V, _, R, _ = _split_stretched()
    assert np.allclose(V[:len(V0)], Vs), 'original vertices moved'
    # R is affine-exact for the linear stretch, so R @ Vs reproduces every (recursively midpointed) vertex
    assert np.allclose(R @ Vs, V)


def test_rest_map():
    V0, _, _, V, _, R, _ = _split_stretched()
    assert R.shape == (len(V), len(V0))
    assert np.allclose(R[:len(V0)].toarray(), np.eye(len(V0))), 'original rows are not the identity'
    assert np.allclose(np.asarray(R.sum(axis=1)).ravel(), 1.0), 'rows are not convex combinations'


def test_nothing_to_split_is_a_no_op():
    V0, F0 = _torus()
    V, F, R, n = split_long_edges(V0, F0, sparse.identity(len(V0), format='csr'), max_len=10.0)
    assert n == 0 and np.array_equal(F, F0) and np.allclose(V, V0) and R.shape == (len(V0), len(V0))


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
