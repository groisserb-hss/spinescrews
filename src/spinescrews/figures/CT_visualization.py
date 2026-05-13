import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import yaml
import logging
from argparse import ArgumentParser
from bg3dtools.transforms_unified import transform_points_inverse

from spinescrews.tools.paths import registration_level_dir, orient_level_dir, detection_dir, registration_dir

log = logging.getLogger(__name__)


def generate_ct_figures(analysis_dir, postop_volumes=None):
    """Generate CT visualization figures for all levels with registered post-op data.

    Outputs {LEVEL}.png to 06_registration/.

    Parameters
    ----------
    postop_volumes : dict or None
        {level_name: nib.Nifti1Image} from the registration save loop.
        When provided, figures can be generated without postop-reg.nii.gz on disk.
    """
    reg_dir = registration_dir(analysis_dir)
    # find available levels by scanning for postop-reg_affine.npy (always written)
    reg_files = glob(os.path.join(reg_dir, '*', 'postop-reg_affine.npy'))
    levels = [os.path.basename(os.path.dirname(f)) for f in reg_files]

    if postop_volumes is None:
        postop_volumes = {}

    for level in levels:
        try:
            _generate_one(analysis_dir, level, postop_volumes.get(level))
        except Exception as e:
            log.warning('CT figure for %s failed: %s' % (level, e))


def _generate_one(analysis_dir, level, postop_img=None):
    """Generate a single 4-panel CT figure for one vertebral level, saved to 06_registration/.

    Parameters
    ----------
    postop_img : nib.Nifti1Image or None
        In-memory postop volume. Falls back to loading postop-reg.nii.gz from disk.
    """
    reg_level = registration_level_dir(analysis_dir, level)
    orient_level = orient_level_dir(analysis_dir, level)
    det_dir = detection_dir(analysis_dir)

    if postop_img is not None:
        CT = _crop_200(postop_img.get_fdata(), postop_img.affine)
    else:
        CT = load_200(os.path.join(reg_level, 'postop-reg.nii.gz'))

    label_file = os.path.join(orient_level, 'preop_seg.nii.gz')
    label = load_200(label_file)

    planned, detected = load_screws(orient_level, reg_level, det_dir, level)

    h = draw_error(CT, label, planned, detected)
    plt.suptitle(level)

    out_dir = registration_dir(analysis_dir)
    figname = os.path.join(out_dir, '%s.png' % level)
    plt.savefig(figname, dpi=150)
    plt.close()
    log.debug('Saved %s' % figname)


def _crop_200(data, affine):
    """Pad and crop a 200x200x200 voxel box centered on the vertebra origin."""
    data = np.pad(data, ((50, 50), (50, 50), (50, 50)), mode='constant')

    assert np.allclose(np.diag(affine)[:3], [0.5, 0.5, 0.5])

    midpt = -2 * affine[:3, 3] + 50
    assert np.all(midpt - np.array([99, 99, 119]) > 0)
    assert np.all(midpt + np.array([100, 100, 80]) <= np.array(data.shape))

    data = data[int(midpt[0] - 99):int(midpt[0] + 100),
           int(midpt[1] - 99):int(midpt[1] + 100),
           int(midpt[2] - 119):int(midpt[2] + 80)]

    return data


def load_200(img_file):
    """Load a NIfTI and crop a 200x200x200 voxel box centered on the vertebra origin."""
    img = nib.load(img_file)
    return _crop_200(img.get_fdata(), img.affine)


def load_screws(preop_level_dir, reg_level_dir, det_dir, level):
    """Load screw endpoints and transform to voxel coordinates."""
    # Affine transformation matrix (normalized → voxel)
    vox2world = np.array([[0.5, 0, 0, -49.75],
                          [0, 0.5, 0, -49.75],
                          [0, 0, 0.5, -59.75],
                          [0, 0, 0, 1]])

    # load screws from 05_detection/
    with open(os.path.join(det_dir, level + 'L_screw.yml'), 'r') as file:
        left_screw = yaml.safe_load(file)
    with open(os.path.join(det_dir, level + 'R_screw.yml'), 'r') as file:
        right_screw = yaml.safe_load(file)

    planned = np.row_stack([left_screw['planned_entry'], left_screw['planned_tip'],
                            right_screw['planned_entry'], right_screw['planned_tip']])

    detected = np.row_stack([left_screw['detected_entry'], left_screw['detected_tip'],
                             right_screw['detected_entry'], right_screw['detected_tip']])

    # load and apply transformation matrices
    pre_affine = np.load(os.path.join(preop_level_dir, 'preop_affine-refined.npy'))
    planned = transform_points_inverse(pre_affine, planned)
    planned = transform_points_inverse(vox2world, planned)

    post_affine = np.load(os.path.join(reg_level_dir, 'postop-reg_affine.npy'))
    detected = transform_points_inverse(post_affine, detected)
    detected = transform_points_inverse(vox2world, detected)

    return planned, detected

def draw_error(CT, segmentation, planned=None, detected=None):
    """Draw 4-panel (axial, coronal, sagittal L/R) CT with planned (green) and detected (red) screw lines."""
    mask = np.maximum(CT > 2000, segmentation)
    masked = CT * ((0.3 + mask) / 1.3)
    mark = True
    segmentation = mask > 0

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Axial slice
    axs[0, 0].imshow(pseudo_slice(masked, segmentation, detected, dim=2), cmap="gray")
    axs[0, 0].set_title('Axial')
    if mark and planned is not None and detected is not None:
        plot_points(axs[0, 0], planned, detected, dim_order=[1, 0])

    # Coronal slice
    axs[0, 1].imshow(pseudo_slice(masked, segmentation, detected, dim=1), cmap="gray")
    axs[0, 1].set_title('Coronal')
    if mark and planned is not None and detected is not None:
        plot_points(axs[0, 1], planned, detected, dim_order=[0, 2], invert_y=True)

    # Sagittal slice left
    axs[1, 0].imshow(pseudo_slice(masked, segmentation, detected[:2], dim=0), cmap="gray")
    axs[1, 0].set_title('Left')
    if mark and planned is not None and detected is not None:
        plot_points(axs[1, 0], planned[:2], detected[:2], dim_order=[1, 2], invert_y=True)

    # Sagittal slice right
    axs[1, 1].imshow(pseudo_slice(masked, segmentation, detected[2:], dim=0), cmap="gray")
    axs[1, 1].set_title('Right')
    if mark and planned is not None and detected is not None:
        plot_points(axs[1, 1], planned[2:], detected[2:], dim_order=[1, 2], invert_y=True)

    return fig


def plot_points(ax, planned, detected, dim_order, invert_y=False):
    """Overlay planned (green) and detected (red) screw endpoint lines on an axis."""
    ax.plot(planned[:2, dim_order[0]], planned[:2, dim_order[1]], 'g-', linewidth=2)
    ax.plot(planned[2:, dim_order[0]], planned[2:, dim_order[1]], 'g-', linewidth=2)
    ax.plot(detected[:2, dim_order[0]], detected[:2, dim_order[1]], 'r-', linewidth=2)
    ax.plot(detected[2:, dim_order[0]], detected[2:, dim_order[1]], 'r-', linewidth=2)
    if invert_y:
        ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")


def pseudo_slice(masked, segmentation, detected, dim):
    """Average a slab around detected screws along dim, returning an RGB image with blue seg highlight."""
    N = masked.shape[dim]
    if np.all(np.isfinite(detected)):
        aa = max(1, int(np.round(np.nanmin(detected[:, dim]) - 2)))
        zz = min(N, int(np.round(np.nanmax(detected[:, dim]) + 2)))
    else:
        aa = int(N // 3)
        zz = int(2 * N // 3)

    mm = int(np.round((aa + zz) / 2))

    if dim == 0:
        img = masked[aa:zz, :, :]
        img = np.mean(img, axis=dim)
        mask = np.any(segmentation[round((aa + mm) / 2):round((zz + mm) / 2), :, :], axis=dim)
        img = np.fliplr(np.rot90(img, k=3))
        mask = np.fliplr(np.rot90(mask, k=3))
    elif dim == 1:
        img = masked[:, aa:zz, :]
        img = np.mean(img, axis=dim)
        mask = np.any(segmentation[:, round((aa + mm) / 2):round((zz + mm) / 2), :], axis=dim)
        img = np.fliplr(np.rot90(img, k=3))
        mask = np.fliplr(np.rot90(mask, k=3))
    else:
        img = masked[:, :, aa:zz]
        img = np.mean(img, axis=dim)
        mask = np.any(segmentation[:, :, round((aa + mm) / 2):round((zz + mm) / 2)], axis=dim)

    # blue highlight for segmentation mask
    img = np.clip((img + 50) / 750, 0, 1)
    highlighted = img.copy()
    highlighted[mask] = (highlighted[mask] + 1) / 2
    highlighted = np.stack((img * 0.8, img * 0.8, highlighted), axis=-1)

    return highlighted


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate CT visualization figures.')
    parser.add_argument("specimen_dir", type=str)
    args = parser.parse_args()

    specimen_dir = os.path.expanduser(args.specimen_dir)
    analysis_dir = os.path.join(specimen_dir, 'analysis')
    generate_ct_figures(analysis_dir)
