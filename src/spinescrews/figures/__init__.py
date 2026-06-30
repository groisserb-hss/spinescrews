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

import logging

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
