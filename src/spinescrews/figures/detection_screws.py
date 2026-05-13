"""Detection screws figure: 2x2 multi-angle MIP with detected screw lines."""

import os
import logging
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spinescrews.tools.paths import detection_dir
from spinescrews.figures._detection_common import (
    VIEWS, load_postop_metal, compute_mip, world_to_mip_pixel, load_screw_yamls,
)

log = logging.getLogger(__name__)


def render_mip_with_screws(metal_data, affine, screw_endpoints, output_path):
    """Render 2x2 MIP figure with screw lines overlaid.

    Parameters
    ----------
    metal_data : ndarray
        3D array with original HU values where metal, 0 elsewhere.
    affine : ndarray
        4x4 voxel-to-world affine.
    screw_endpoints : list of (entry_3d, tip_3d)
        Each element is a tuple of two (3,) arrays in world coordinates.
    output_path : str
        Path to save the figure.
    """
    aspect = abs(affine[2, 2] / affine[0, 0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 14))

    for ax, (label, angle) in zip(axes.flat, VIEWS):
        mip, orig_shape, rot_shape = compute_mip(metal_data, angle)

        ax.imshow(np.clip(mip.T / 4000, 0, 1), cmap='gray',
                  origin='lower', aspect=aspect)

        for entry, tip in screw_endpoints:
            pts = np.array([entry, tip])
            px = world_to_mip_pixel(pts, affine, angle, orig_shape, rot_shape)
            ax.plot(px[:, 0], px[:, 1], 'r-', linewidth=1.5, alpha=0.8)
            ax.plot(px[0, 0], px[0, 1], 'r.', markersize=4)

        ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.debug('Saved MIP figure to %s', output_path)


def generate_detection_screws(analysis_dir, specimen_dir, threshold=None):
    """Generate 2x2 multi-angle MIP of post-op metal with detected screw lines (red).

    Parameters
    ----------
    analysis_dir : str
        Path to analysis/ directory.
    specimen_dir : str
        Path to specimen root (contains postop.nii.gz).
    threshold : int or None
        HU threshold for metal voxels. None = adaptive (Otsu).
    """
    metal_data, affine = load_postop_metal(specimen_dir, threshold)
    screws = load_screw_yamls(analysis_dir)
    if not screws:
        log.warning('No screw YAMLs found -- skipping detection_screws figure')
        return

    screw_endpoints = []
    for screw in screws:
        de = screw.get('detected_entry')
        dt = screw.get('detected_tip')
        if de is not None and dt is not None:
            screw_endpoints.append((np.asarray(de), np.asarray(dt)))

    det_dir = detection_dir(analysis_dir)
    out_path = os.path.join(det_dir, 'detection_screws.png')
    render_mip_with_screws(metal_data, affine, screw_endpoints, out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate multi-angle detected screws figure.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--threshold', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    generate_detection_screws(os.path.join(specimen, 'analysis'), specimen,
                              threshold=args.threshold)
