"""Detection plan-vs-detected figure: 2x2 multi-angle comparison of planned and detected screws."""

import os
import logging
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from spinescrews.tools.paths import detection_dir
from spinescrews.figures._detection_common import (
    VIEWS, load_postop_geometry, compute_rotated_shape,
    world_to_mip_pixel, load_screw_yamls, build_preop_to_postop,
)

log = logging.getLogger(__name__)


def generate_detection_plan_vs_detected(analysis_dir, specimen_dir, threshold=None):
    """Generate 2x2 multi-angle figure with planned (green) + detected (red) screws.

    No CT background -- just screw shafts on black.

    Parameters
    ----------
    analysis_dir : str
        Path to analysis/ directory.
    specimen_dir : str
        Path to specimen root (contains postop.nii.gz).
    threshold : int or None
        Unused (kept for API compatibility with pipeline caller).
    """
    affine, vol_shape = load_postop_geometry(specimen_dir)
    screws = load_screw_yamls(analysis_dir)
    if not screws:
        log.warning('No screw YAMLs found -- skipping plan-vs-detected figure')
        return

    det_dir = detection_dir(analysis_dir)
    preop_to_postop = build_preop_to_postop(analysis_dir, det_dir)
    aspect = abs(affine[2, 2] / affine[0, 0])

    # -- Collect all screw endpoints in postop world space ----------------
    planned_world = []   # list of (2, 3) arrays
    detected_world = []  # list of (2, 3) arrays
    screw_levels = []    # parallel to planned_world

    for screw in screws:
        name = screw.get('name', '')
        level = name[:-1]

        pe = screw.get('planned_entry')
        pt = screw.get('planned_tip')
        if pe is not None and pt is not None:
            pts = np.array([pe, pt])
            if level in preop_to_postop:
                pts_h = np.c_[pts, np.ones(len(pts))]
                pts = (preop_to_postop[level] @ pts_h.T).T[:, :3]
            planned_world.append(pts)
        else:
            planned_world.append(None)
        screw_levels.append(level)

        de = screw.get('detected_entry')
        dt = screw.get('detected_tip')
        if de is not None and dt is not None:
            detected_world.append(np.array([de, dt]))
        else:
            detected_world.append(None)

    # -- Compute per-panel pixel coords + bounding boxes ------------------
    PAD_FRAC = 0.10
    all_y = []  # S pixel coords across all panels (for shared y-limits)
    panel_data = []

    for label, angle in VIEWS:
        rot_shape = compute_rotated_shape(vol_shape, angle)
        all_x = []
        panel_planned_px = []
        panel_detected_px = []

        for pw, dw in zip(planned_world, detected_world):
            if pw is not None:
                px = world_to_mip_pixel(pw, affine, angle, vol_shape, rot_shape)
                panel_planned_px.append(px)
                all_x.extend(px[:, 0])
                all_y.extend(px[:, 1])
            else:
                panel_planned_px.append(None)

            if dw is not None:
                px = world_to_mip_pixel(dw, affine, angle, vol_shape, rot_shape)
                panel_detected_px.append(px)
                all_x.extend(px[:, 0])
                all_y.extend(px[:, 1])
            else:
                panel_detected_px.append(None)

        x_min, x_max = min(all_x), max(all_x)
        x_pad = max((x_max - x_min) * PAD_FRAC, 5)
        panel_data.append({
            'label': label,
            'planned_px': panel_planned_px,
            'detected_px': panel_detected_px,
            'xlim': (x_min - x_pad, x_max + x_pad),
        })

    y_min, y_max = min(all_y), max(all_y)
    y_pad = max((y_max - y_min) * PAD_FRAC, 5)
    ylim = (y_min - y_pad, y_max + y_pad)

    # -- Draw panels ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor='black')

    for ax, pd in zip(axes.flat, panel_data):
        ax.set_xlim(pd['xlim'])
        ax.set_ylim(ylim)
        ax.set_facecolor('black')
        ax.set_aspect(aspect)

        for ppx in pd['planned_px']:
            if ppx is not None:
                ax.plot(ppx[:, 0], ppx[:, 1], 'g-', linewidth=1.5, alpha=0.8)
                ax.plot(ppx[0, 0], ppx[0, 1], 'g.', markersize=4)

        for dpx in pd['detected_px']:
            if dpx is not None:
                ax.plot(dpx[:, 0], dpx[:, 1], 'r-', linewidth=1.5, alpha=0.8)
                ax.plot(dpx[0, 0], dpx[0, 1], 'r.', markersize=4)

        ax.set_title(pd['label'], color='white')
        ax.axis('off')

    legend_elements = [
        Line2D([0], [0], color='g', linewidth=1.5, label='Planned'),
        Line2D([0], [0], color='r', linewidth=1.5, label='Detected'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=False, labelcolor='white')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(det_dir, 'detection_plan-vs-detected.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    log.info('Saved plan-vs-detected figure to %s' % out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate multi-angle plan-vs-detected figure.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--threshold', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    generate_detection_plan_vs_detected(os.path.join(specimen, 'analysis'), specimen,
                                        threshold=args.threshold)
