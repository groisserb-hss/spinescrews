"""Figure generation for each pipeline step.

- seg_overlay: 3-panel segmentation overlay (axial/coronal/sagittal)
- preop_orientation: Grid of oriented vertebrae with CT slices and meshes
- correspondence_preprocess: 4-panel geodesic distance visualization
- correspondence_match: 2x2 label and gradient correspondence QC
- orient_refinement: Bar charts of rotation/translation corrections
- spine_template: Anterior + lateral views of seg and template meshes
- detection_screws: Multi-angle MIP with detected screw lines
- detection_plan_vs_detected: Multi-angle MIP comparing planned vs detected screws
- CT_visualization: 4-panel CT visualization per vertebral level
- visualize_breach: 3-panel screw-aligned breach visualization
- group_statistics: Cross-specimen error analysis
"""

import functools
import logging
import sys

log = logging.getLogger(__name__)


def safe_figure(fn, *args, **kwargs):
    """Run a figure-generating callable, swallowing any error it raises.

    Figures are diagnostic-only: a failure (a dead Open3D backend raising
    ``RenderUnavailable``, a matplotlib glitch, missing optional inputs, ...)
    must never block a pipeline step from writing its data products or its
    ``summary.json`` gate. This is the in-process counterpart to
    ``bg3dtools.render.run_isolated`` (which isolates Open3D in a subprocess);
    use ``safe_figure`` for the matplotlib / in-process figure calls.

    Returns True on success, False if *fn* raised (logged with full traceback).
    """
    try:
        fn(*args, **kwargs)
        return True
    except Exception:
        log.warning('figure %s failed; continuing without it',
                    getattr(fn, '__name__', repr(fn)), exc_info=True)
        return False


@functools.lru_cache(maxsize=1)
def probe_render_backends() -> frozenset:
    """Probe the Open3D render backends ONCE in this process; return the names found unusable.

    On a headless host (e.g. Windows without EGL) Open3D's offscreen/legacy engines print a native
    "EGL Headless is not supported" error to stderr *before* raising. ``bg3dtools``'s dispatcher
    memoizes a dead backend so it prints at most once per process -- but every ``run_isolated``
    render subprocess is a fresh process that re-probes. Probing once here in the parent and handing
    the result to those subprocesses (see :func:`seed_dead_backends`) collapses a whole pipeline run
    down to a single probe.

    No-op on macOS: offscreen isn't used there, and probing would pull the legacy Visualizer into the
    main process -- exactly what ``run_isolated`` exists to avoid. Any failure is swallowed; a probe
    must never affect the pipeline.
    """
    if sys.platform == 'darwin':
        return frozenset()
    try:
        import numpy as np
        import bg3dtools.render.scan as scan
        from bg3dtools.render import render_frame, CameraParams, Mesh
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # non-degenerate (3D extent)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
        cam = CameraParams(lookat=np.array([0.25, 0.25, 0.25], np.float32),
                           eye=np.array([2, 2, 2], np.float32),
                           up=np.array([0, 0, 1], np.float32), fov=45.0)
        try:
            render_frame([Mesh(vertices=v, faces=f)], cam, width=16, height=16)
        except Exception:
            pass  # a dead backend is the point of the probe; scan._dead_backends now records it
        return frozenset(scan._dead_backends)
    except Exception:
        return frozenset()


def seed_dead_backends(names) -> None:
    """Pre-mark render backends the parent already found dead (from :func:`probe_render_backends`).

    Call at the top of a ``run_isolated`` child so it skips those backends instead of re-probing --
    avoiding a repeat of Open3D's native stderr error in each render subprocess.
    """
    if not names:
        return
    try:
        import bg3dtools.render.scan as scan
        scan._dead_backends.update(names)
    except Exception:
        pass
