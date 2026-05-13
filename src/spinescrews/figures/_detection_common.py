"""Shared helpers for multi-angle detection figures."""

import os
import logging
import numpy as np
import nibabel as nib
from glob import glob
import yaml

from spinescrews.tools.paths import detection_dir, orient_level_dir, preop_level_dir

log = logging.getLogger(__name__)

# 2x2 panel layout: (label, angle_deg)
VIEWS = [
    ('Coronal', 0),
    ('Oblique +45\u00b0', 45),
    ('Oblique \u221245\u00b0', -45),
    ('Sagittal', 90),
]


def load_postop_metal(specimen_dir, threshold=None):
    """Load postop CT, return (metal_data, affine).

    metal_data has original HU values where > threshold, 0 elsewhere.
    When threshold is None, uses adaptive Otsu-based threshold.
    """
    postop_path = os.path.join(specimen_dir, 'postop.nii.gz')
    if not os.path.isfile(postop_path):
        raise FileNotFoundError('Missing postop.nii.gz in %s' % specimen_dir)

    img = nib.as_closest_canonical(nib.load(postop_path), True)
    data = img.get_fdata()
    affine = img.affine

    if threshold is None:
        from spinescrews.tools.nifti_utils import compute_metal_threshold
        threshold = compute_metal_threshold(data)

    metal_data = np.where(data > threshold, data, 0.0)
    return metal_data, affine


def compute_mip(metal_data, angle_deg):
    """Compute MIP by rotating sparse metal voxels, then projecting.

    For 0 deg (coronal): projects along A-axis (axis 1).
    For nonzero angles: rotates sparse coordinates in the R-A plane,
    then bins into a 2D grid using np.maximum.at for MIP projection.

    Returns (mip_2d, orig_shape, rotated_shape).
    """
    orig_shape = metal_data.shape

    if angle_deg == 0:
        mip = np.max(metal_data, axis=1)
        return mip, orig_shape, orig_shape

    rot_shape = compute_rotated_shape(orig_shape, angle_deg)

    # Extract sparse non-zero voxels
    coords = np.argwhere(metal_data != 0)  # (N, 3) — (r, a, s)
    values = metal_data[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Rotate in R-A plane around volume center
    c_r = (orig_shape[0] - 1) / 2.0
    c_a = (orig_shape[1] - 1) / 2.0
    c_r_rot = (rot_shape[0] - 1) / 2.0
    c_a_rot = (rot_shape[1] - 1) / 2.0

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dr = coords[:, 0] - c_r
    da = coords[:, 1] - c_a
    r_rot = cos_t * dr - sin_t * da + c_r_rot
    a_rot = sin_t * dr + cos_t * da + c_a_rot

    # Bin into 2D MIP (project away A-axis)
    r_idx = np.clip(np.round(r_rot).astype(int), 0, rot_shape[0] - 1)
    s_idx = coords[:, 2]  # S-axis unchanged by R-A rotation

    mip = np.zeros((rot_shape[0], rot_shape[2]), dtype=metal_data.dtype)
    np.maximum.at(mip, (r_idx, s_idx), values)

    return mip, orig_shape, rot_shape


def world_to_mip_pixel(pts_world, affine, angle_deg, orig_shape, rotated_shape):
    """Convert Nx3 world-RAS points to Nx2 MIP pixel coordinates (x, y).

    x = horizontal (rotated R direction), y = vertical (S direction).
    The same rotation that ndimage.rotate applies to the volume is applied
    to the voxel coordinates in the R-A plane; the A component is then
    discarded (projected away by the MIP along axis 1).
    """
    pts = np.atleast_2d(pts_world)
    inv_aff = np.linalg.inv(affine)

    # World -> voxel
    pts_h = np.c_[pts, np.ones(len(pts))]
    vox = (inv_aff @ pts_h.T).T[:, :3]  # (N, 3) -- (v_r, v_a, v_s)

    # Rotate in R-A plane around volume center
    c_r = (orig_shape[0] - 1) / 2.0
    c_a = (orig_shape[1] - 1) / 2.0
    c_r_rot = (rotated_shape[0] - 1) / 2.0

    theta = np.radians(angle_deg)
    dr = vox[:, 0] - c_r
    da = vox[:, 1] - c_a
    r_rot = np.cos(theta) * dr - np.sin(theta) * da + c_r_rot

    return np.column_stack([r_rot, vox[:, 2]])


def build_preop_to_postop(analysis_dir, det_dir):
    """Build per-level 4x4 transforms mapping preop world -> postop world.

    Uses spine_tforms_initial.npz (postop vertebral affines from detection)
    and preop affines (step 04 refined, fallback step 02) to compose:
        preop_to_postop[level] = spine_tform[level] @ inv(preop_aff[level])
    """
    tforms_path = os.path.join(det_dir, 'spine_tforms_initial.npz')
    if not os.path.isfile(tforms_path):
        return {}

    preop_to_postop = {}
    spine_tforms = dict(np.load(tforms_path))
    for level, postop_aff in spine_tforms.items():
        refined = os.path.join(orient_level_dir(analysis_dir, level),
                               'preop_affine-refined.npy')
        original = os.path.join(preop_level_dir(analysis_dir, level),
                                'preop_affine.npy')
        if os.path.isfile(refined):
            preop_aff = np.load(refined)
        elif os.path.isfile(original):
            preop_aff = np.load(original)
        else:
            continue
        preop_to_postop[level] = postop_aff @ np.linalg.inv(preop_aff)

    return preop_to_postop


def load_postop_geometry(specimen_dir):
    """Load postop CT geometry (affine + shape) without reading voxel data."""
    postop_path = os.path.join(specimen_dir, 'postop.nii.gz')
    if not os.path.isfile(postop_path):
        raise FileNotFoundError('Missing postop.nii.gz in %s' % specimen_dir)
    img = nib.as_closest_canonical(nib.load(postop_path), True)
    return img.affine, img.shape


def compute_rotated_shape(orig_shape, angle_deg):
    """Compute output shape of ndimage.rotate(..., axes=(0,1), reshape=True).

    Matches scipy's bounding-box calculation so that world_to_mip_pixel
    coordinates are consistent whether or not the actual rotation is performed.
    """
    if angle_deg == 0:
        return orig_shape
    theta = np.radians(angle_deg)
    n0, n1 = orig_shape[0], orig_shape[1]
    rot_0 = int(np.ceil(abs(n0 * np.cos(theta)) + abs(n1 * np.sin(theta))))
    rot_1 = int(np.ceil(abs(n0 * np.sin(theta)) + abs(n1 * np.cos(theta))))
    return (rot_0, rot_1) + orig_shape[2:]


def load_screw_yamls(analysis_dir):
    """Load all screw YAML files from detection dir. Returns list of dicts."""
    det_dir = detection_dir(analysis_dir)
    screw_files = sorted(glob(os.path.join(det_dir, '*_screw.yml')))
    screws = []
    for yml_path in screw_files:
        with open(yml_path, 'r') as f:
            screws.append(yaml.safe_load(f))
    return screws
