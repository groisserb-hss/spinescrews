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


def _probe_child(q) -> None:
    """Render a tiny scene in a spawned subprocess and report the dead render tiers via ``q``.

    Runs isolated on purpose: Open3D's offscreen Filament engine fails *catchably* on a headless
    host, but the legacy Visualizer it falls back to can then *hard-abort* (native C++ crash, not a
    Python exception) -- doing this in the main process would take the whole pipeline down. If this
    child survives, it reports which tiers ``bg3dtools`` marked dead; if it aborts outright, the
    parent gets no result and assumes every Open3D tier is dead.
    """
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
            pass  # a dead backend is the point of the probe; scan._dead_backends records it
        q.put(list(scan._dead_backends))
    except Exception:
        pass  # setup failed; a missing result tells the parent to assume Open3D is unusable


# bg3dtools' Open3D render tiers, named as scan.py's backend function __name__s.
_OPEN3D_TIERS = ('_exec_offscreen', '_exec_legacy')


@functools.lru_cache(maxsize=1)
def probe_render_backends() -> frozenset:
    """Determine ONCE which Open3D render tiers are unusable, WITHOUT risking the main process.

    On a headless host (e.g. a Windows box with no EGL and no usable GL context) Open3D's offscreen
    Filament engine fails catchably, but the legacy Visualizer it then tries can *hard-abort*
    (native, uncatchable) -- so a probe that renders in the main process crashes the whole pipeline.
    That is exactly what killed ``spinescrews-align`` at the end of step 02 on headless Windows.

    So the probe render runs in a spawned subprocess: if it survives it reports the dead tiers; if it
    aborts we pessimistically treat every Open3D tier as dead. Either way the result is seeded into
    THIS process too (see :func:`seed_dead_backends`), so in-process figures (the correspondence QC
    panels) and the ``run_isolated`` figure subprocesses alike skip the crashing tiers and fall
    through to the matplotlib renderer instead of aborting.

    No-op on macOS: offscreen isn't used there and the legacy Visualizer is a visible window handled
    by ``run_isolated``. Any failure is swallowed; a probe must never affect the pipeline.
    """
    if sys.platform == 'darwin':
        return frozenset()
    dead = _OPEN3D_TIERS  # pessimistic default: assume Open3D is unusable unless the child proves otherwise
    reason = 'no probe result'
    try:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_probe_child, args=(q,))
        p.start()
        p.join(timeout=60)
        if p.is_alive():                     # hung (deadlocked GL init) -> kill, keep pessimistic default
            p.kill(); p.join()
            reason = 'probe subprocess hung and was killed after 60s'
        elif p.exitcode == 0:                # child survived -> trust the tiers it actually found dead
            try:
                dead = tuple(q.get(timeout=5))
                reason = 'probe subprocess completed'
            except Exception:
                reason = 'probe subprocess exited cleanly but returned no result'
        else:                                # child hard-aborted: the headless native crash we can't catch
            reason = 'probe subprocess crashed (exit code %s) -- Open3D unusable in-process' % p.exitcode
    except Exception as exc:
        log.warning('render-backend probe could not run (%r); each figure will fall back on its own', exc)
        return frozenset()

    result = frozenset(dead)
    seed_dead_backends(result)               # also skip the dead tiers for in-process render_frame calls here
    if set(_OPEN3D_TIERS) <= result:
        log.warning('render probe: %s. Open3D offscreen and legacy are both unavailable on this host, so 3D '
                    'QC figures fall back to the crude matplotlib renderer. Any "[Open3D Error] ..." printed '
                    'above is expected here and is NOT a pipeline failure.', reason)
    elif result:
        log.info('render probe: %s. Unusable tier(s): %s; 3D figures use the legacy Open3D Visualizer.',
                 reason, ', '.join(sorted(result)))
    else:
        log.info('render probe: %s. Open3D offscreen renderer available.', reason)
    return result


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
