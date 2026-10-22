"""Tests for Vertebra.load_unwound / load_measurement_surface: how steps 03 and 07 read step 02's erosion-unwound
mesh and carry the template labels onto it.

Step 07 measures breach against the unwound mesh (its wall is the segmented cortex), while the template labels
that locate the pedicle were matched to the eroded, decimated preop_gen1.ply. The chain
template2bone (small) -> small2med (intermediate) -> inflated2med (unwound, refined) must therefore compose to a
per-vertex label map on the measurement surface, with the identity fallback for step-02 outputs written before the
unwind refined the mesh, and loud failures for anything inconsistent.

Runnable two ways:
    python dicom_tools/tests/test_measurement_surface.py     # prints PASS/SKIP/FAIL
    pytest dicom_tools/tests/                                 # if pytest installed

Runs in the screws310 env (where spinescrews is installed); needs no data (a synthetic level directory is written
to a temporary folder).
"""

import os
import tempfile
from os.path import join

import igl
import numpy as np
from scipy import sparse

from spinescrews.tools.mesh_refine import split_long_edges
from spinescrews.tools.vertebrae import Vertebra
from test_mesh_refine import _torus

# ---- skip mechanism compatible with pytest and the plain runner ----
try:
    import pytest
    _SKIP_EXC = pytest.skip.Exception
except Exception:                       # pytest not installed
    class _SkipExc(Exception):
        pass
    _SKIP_EXC = _SkipExc


def _row_stochastic(rng, n_rows, n_cols, k=3):
    """Sparse (n_rows x n_cols) matrix whose rows are convex combinations of k random columns."""
    rows = np.repeat(np.arange(n_rows), k)
    cols = np.concatenate([rng.choice(n_cols, size=k, replace=False) for _ in range(n_rows)])
    w = rng.random(n_rows * k).reshape(n_rows, k)
    w /= w.sum(axis=1, keepdims=True)
    return sparse.csr_matrix((w.ravel(), (rows, cols)), shape=(n_rows, n_cols))


def _write_level(level_dir, refined=True, with_map=True, seed=0):
    """Write a synthetic 02_preop/{LEVEL}/ directory and return what was written."""
    rng = np.random.default_rng(seed)
    small_v, small_f = _torus(6, 4)        # preop_gen1.ply (final decimation)
    med_v, med_f = _torus(10, 6)           # intermediate decimation
    n_small, n_med = len(small_v), len(med_v)
    if refined:
        # the unwind stretches and then splits the intermediate mesh, appending vertices
        inflated_v, inflated_f, inflated2med, n_split = split_long_edges(
            med_v * np.array([4.0, 1.0, 1.0]), med_f, sparse.identity(n_med, format='csr'), max_len=0.9)
        assert n_split > 0
    else:
        inflated_v, inflated_f, inflated2med = med_v.copy(), med_f.copy(), sparse.identity(n_med, format='csr')
    small2med = _row_stochastic(rng, n_med, n_small)
    med2small = _row_stochastic(rng, n_small, n_med)

    os.makedirs(level_dir, exist_ok=True)
    np.save(join(level_dir, 'preop_affine.npy'), np.eye(4))
    igl.write_triangle_mesh(join(level_dir, 'preop_gen1.ply'), small_v, small_f)
    igl.write_triangle_mesh(join(level_dir, 'preop_gen1-inflated.ply'), inflated_v, inflated_f)
    sparse.save_npz(join(level_dir, 'small2med.npz'), small2med)
    sparse.save_npz(join(level_dir, 'med2small.npz'), med2small)
    if with_map:
        sparse.save_npz(join(level_dir, 'inflated2med.npz'), inflated2med)
    return dict(n_small=n_small, n_med=n_med, n_inflated=len(inflated_v), inflated2med=inflated2med,
                small2med=small2med, med2small=med2small)


def _dense(m):
    return np.asarray(sparse.csr_matrix(m).todense())


def test_loads_refined_mesh_and_maps():
    with tempfile.TemporaryDirectory() as tmp:
        w = _write_level(tmp, refined=True, with_map=True)
        v, f, small2med, med2small, inflated2med = Vertebra.load_unwound(tmp)
    assert len(v) == w['n_inflated'] > w['n_med'], 'the refined mesh was not loaded'
    assert small2med.shape == (w['n_med'], w['n_small']) and med2small.shape == (w['n_small'], w['n_med'])
    assert inflated2med.shape == (w['n_inflated'], w['n_med'])
    assert np.allclose(_dense(inflated2med), _dense(w['inflated2med']))
    assert np.allclose(_dense(inflated2med)[:w['n_med']], np.eye(w['n_med'])), 'original vertices must map to themselves'


def test_identity_fallback_for_unrefined_legacy_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        w = _write_level(tmp, refined=False, with_map=False)
        v, _, _, _, inflated2med = Vertebra.load_unwound(tmp)
    assert len(v) == w['n_med']
    assert np.allclose(_dense(inflated2med), np.eye(w['n_med']))


def test_refined_mesh_without_rest_map_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _write_level(tmp, refined=True, with_map=False)
        try:
            Vertebra.load_unwound(tmp)
        except FileNotFoundError as e:
            assert 'inflated2med' in str(e)
        else:
            raise AssertionError('a refined mesh without inflated2med.npz must not load')


def test_missing_step02_output_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _write_level(tmp)
        os.remove(join(tmp, 'small2med.npz'))
        try:
            Vertebra.load_unwound(tmp)
        except FileNotFoundError as e:
            assert 'small2med' in str(e)
        else:
            raise AssertionError('a level missing small2med.npz must not load')


def test_rest_map_shape_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        w = _write_level(tmp, refined=True, with_map=False)
        sparse.save_npz(join(tmp, 'inflated2med.npz'), sparse.identity(w['n_med'], format='csr'))
        try:
            Vertebra.load_unwound(tmp)
        except ValueError as e:
            assert 'expected' in str(e)
        else:
            raise AssertionError('a rest map of the wrong shape must not load')


def test_labels_carried_onto_measurement_surface():
    rng = np.random.default_rng(1)
    n_template = 15
    with tempfile.TemporaryDirectory() as tmp:
        w = _write_level(tmp, refined=True, with_map=True)
        template2bone = _row_stochastic(rng, w['n_small'], n_template)
        v, f, template2surface = Vertebra.load_measurement_surface(tmp, template2bone)
    n_med = w['n_med']
    assert template2surface.shape == (w['n_inflated'], n_template)
    T = _dense(template2surface)
    assert np.allclose(T.sum(axis=1), 1.0), 'label weights must stay convex combinations'
    on_med = _dense(w['small2med'] @ template2bone)
    assert np.allclose(T[:n_med], on_med), 'the intermediate mesh vertices must carry their own labels'
    expected_new = _dense(w['inflated2med'])[n_med:] @ on_med
    assert np.allclose(T[n_med:], expected_new), 'appended vertices must blend their parents\' labels'
    label = (rng.random(n_template) < 0.3).astype(float)   # one template label, e.g. a pedicle
    weights = template2surface @ label
    assert weights.shape == (w['n_inflated'],) and weights.min() >= 0 and weights.max() <= 1


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
