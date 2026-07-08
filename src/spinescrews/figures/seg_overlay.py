"""Segmentation overlay figure: 3-panel (axial/coronal/sagittal) with colored labels."""

import os
import logging
import numpy as np
import nibabel as nib
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spinescrews.tools import val_seg
from spinescrews.tools.paths import segmentation_dir, segmentation_file

log = logging.getLogger(__name__)

# color per label — use a fixed colormap so labels are consistent across specimens
_LABEL_CMAP = plt.cm.tab20


def generate_seg_overlay(analysis_dir, specimen_dir):
    """Generate 3-panel segmentation overlay and save to 01_segmentation/seg_overlay.png.

    Parameters
    ----------
    analysis_dir : str
        Path to analysis/ directory.
    specimen_dir : str
        Path to specimen root (contains preop.nii.gz and preop_seg.nii.gz).
    """

    ct_path = os.path.join(specimen_dir, 'preop.nii.gz')
    seg_path = segmentation_file(analysis_dir)

    if not os.path.isfile(ct_path) or not os.path.isfile(seg_path):
        log.warning('Cannot generate seg overlay: missing preop.nii.gz or preop_seg.nii.gz')
        return

    ct_img = nib.as_closest_canonical(nib.load(ct_path), True)
    seg_img = nib.as_closest_canonical(nib.load(seg_path), True)

    ct_data = ct_img.get_fdata()
    seg_data = seg_img.get_fdata().astype(np.int32)

    # normalize CT to [0, 1] bone window
    ct_norm = np.clip((ct_data + 200) / 1500, 0, 1)

    # find center of mass of segmentation for slice selection
    nonzero = np.argwhere(seg_data > 0)
    if len(nonzero) == 0:
        log.warning('Segmentation is empty, skipping overlay')
        return
    center = np.median(nonzero, axis=0).astype(int)

    # Build label overlay (RGBA)
    labels = sorted(set(np.unique(seg_data)) - {0})
    n_labels = len(labels)
    label_to_color = {}
    for lab in labels:
        # key the color to the label integer (not its rank among present labels)
        # so a given vertebra keeps the same color across specimens regardless of
        # which levels happen to be segmented
        label_to_color[lab] = _LABEL_CMAP(int(lab) % _LABEL_CMAP.N)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (dim, title) in zip(axs, [(2, 'Axial'), (1, 'Coronal'), (0, 'Sagittal')]):
        idx = center[dim]
        ct_slice = np.take(ct_norm, idx, axis=dim)
        seg_slice = np.take(seg_data, idx, axis=dim)

        # build RGB image
        rgb = np.stack([ct_slice] * 3, axis=-1)

        # overlay colored segmentation labels
        overlay = np.zeros((*seg_slice.shape, 4))
        for lab, color in label_to_color.items():
            mask = seg_slice == lab
            overlay[mask] = color

        # blend: where overlay exists, mix 60% CT + 40% label color
        alpha = overlay[..., 3:4]
        rgb = rgb * (1 - 0.4 * alpha) + overlay[..., :3] * 0.4 * alpha

        # orient for display
        if dim in (0, 1):
            rgb = np.flipud(np.rot90(rgb, k=1))

        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(title)
        ax.axis('off')

    # legend
    from matplotlib.patches import Patch
    legend_patches = []
    for lab in labels:
        name = val_seg.get(lab, str(lab))
        legend_patches.append(Patch(color=label_to_color[lab], label=name))
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(8, n_labels),
               fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    seg_dir = segmentation_dir(analysis_dir)
    os.makedirs(seg_dir, exist_ok=True)
    out_path = os.path.join(seg_dir, 'seg_overlay.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info('Saved segmentation overlay to %s' % out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate segmentation overlay figure.')
    parser.add_argument('specimen_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    generate_seg_overlay(os.path.join(specimen, 'analysis'), specimen)
