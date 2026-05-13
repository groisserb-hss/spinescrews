"""Step 04 QC figure: grid of oriented vertebrae (CT slices + genus-1 mesh)."""

import os
import logging
import numpy as np
import nibabel as nib
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spinescrews.tools import possible_levels
from spinescrews.tools.paths import orient_dir, orient_level_dir, preop_level_dir

log = logging.getLogger(__name__)


def generate_orientation_summary(analysis_dir):
    """Grid of oriented vertebrae. Saves to 04_orient/preop_orientation.png."""

    step_dir = orient_dir(analysis_dir)

    # discover levels that have preop.nii.gz in 04_orient/
    levels = []
    for level in possible_levels:
        level_dir = orient_level_dir(analysis_dir, level)
        if os.path.isfile(os.path.join(level_dir, 'preop.nii.gz')):
            levels.append(level)

    if not levels:
        log.warning('No oriented vertebrae found in %s', step_dir)
        return

    n_levels = len(levels)
    fig, axs = plt.subplots(n_levels, 4, figsize=(14, 3 * n_levels))
    if n_levels == 1:
        axs = axs[np.newaxis, :]

    col_titles = ['Axial (S=120)', 'Coronal (A=100)', 'Sagittal (R=100)', 'Mesh (coronal proj.)']

    for row, level in enumerate(levels):
        level_dir = orient_level_dir(analysis_dir, level)

        ct_img = nib.load(os.path.join(level_dir, 'preop.nii.gz'))
        ct_data = ct_img.get_fdata()
        ct_norm = np.clip((ct_data + 200) / 1500, 0, 1)

        seg_img = nib.load(os.path.join(level_dir, 'preop_seg.nii.gz'))
        seg_data = seg_img.get_fdata()
        seg_mask = seg_data > 0

        # Slice indices for normalized 200x200x200 volume
        # World origin (0,0,0) = voxel (100, 100, 120)
        slices = [
            (2, 120),  # axial: S=120
            (1, 100),  # coronal: A=100
            (0, 100),  # sagittal: R=100
        ]

        for col, (dim, idx) in enumerate(slices):
            ax = axs[row, col]
            idx = min(idx, ct_norm.shape[dim] - 1)
            ct_slice = np.take(ct_norm, idx, axis=dim)
            seg_slice = np.take(seg_mask, idx, axis=dim).astype(float)

            # build RGB from grayscale CT
            rgb = np.stack([ct_slice] * 3, axis=-1)

            # blue overlay where seg > 0
            overlay = np.zeros_like(rgb)
            overlay[..., 2] = seg_slice  # blue channel
            alpha = 0.3
            rgb = rgb * (1 - alpha * seg_slice[..., None]) + overlay * alpha

            # orient for display (coronal, sagittal need rotation)
            if dim in (0, 1):
                rgb = np.flipud(np.rot90(rgb, k=1))

            ax.imshow(np.clip(rgb, 0, 1), origin='lower' if dim in (0, 1) else 'upper')
            ax.axis('off')
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

        # Column 4: mesh scatter (gen1 from step 02, transformed to refined frame)
        ax_mesh = axs[row, 3]
        gen1_file = os.path.join(preop_level_dir(analysis_dir, level), 'preop_gen1.ply')
        if os.path.isfile(gen1_file):
            import igl
            v, _ = igl.read_triangle_mesh(gen1_file)
            # Transform gen1 from original frame to refined frame
            orig_aff_file = os.path.join(preop_level_dir(analysis_dir, level), 'preop_affine.npy')
            refined_aff_file = os.path.join(level_dir, 'preop_affine-refined.npy')
            if os.path.isfile(orig_aff_file) and os.path.isfile(refined_aff_file):
                orig_aff = np.load(orig_aff_file)
                refined_aff = np.load(refined_aff_file)
                delta = np.linalg.inv(refined_aff) @ orig_aff
                v = (delta[:3, :3] @ v.T).T + delta[:3, 3]
            ax_mesh.scatter(v[:, 0], v[:, 2], c=v[:, 1], s=0.3, cmap='viridis',
                            rasterized=True)
            ax_mesh.set_aspect('equal')
            ax_mesh.axis('off')
            if row == 0:
                ax_mesh.set_title(col_titles[3], fontsize=10)
        else:
            ax_mesh.text(0.5, 0.5, 'N/A', ha='center', va='center',
                         transform=ax_mesh.transAxes, fontsize=14, color='gray')
            ax_mesh.axis('off')
            if row == 0:
                ax_mesh.set_title(col_titles[3], fontsize=10)

        # Row label
        axs[row, 0].text(-0.15, 0.5, level, ha='center', va='center',
                         transform=axs[row, 0].transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(step_dir, 'preop_orientation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info('Saved preop orientation figure to %s', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate preop orientation QC figure.')
    parser.add_argument('specimen_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    generate_orientation_summary(os.path.join(specimen, 'analysis'))
