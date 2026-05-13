import logging
import os
import numpy as np
import nibabel as nib
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import igl
import trimesh
import yaml
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from bg3dtools.transforms_unified import transform_points_inverse

from spinescrews.tools.paths import preop_level_dir, orient_level_dir, registration_level_dir, detection_dir, accuracy_dir

log = logging.getLogger(__name__)


def generate_breach_figure(analysis_dir, level, side, extent_mm=15, output=None, preop=False):
    """Generate a 3-panel breach visualization figure.

    Parameters
    ----------
    analysis_dir : str
        Path to the analysis/ directory.
    level : str
        Vertebral level (e.g. 'T11').
    side : str
        'L' or 'R'.
    extent_mm : float
        Extent of the resampled volume in mm.
    output : str or None
        Custom output path. If None, writes to 07_accuracy/breach_{LEVEL}{SIDE}.png.
    preop : bool
        If True, use preop CT instead of registered postop CT.
    """
    screw_name = '%s%s' % (level, side)

    # 1. Load all inputs
    (ct_data, ct_affine, mesh_v, mesh_f, screw_data,
     pre_affine, post_affine, screw_pt_world, ped_pt_world,
     breach_dist, breach_angle) = _load_inputs(analysis_dir, level, side, preop=preop)

    # 2. Transform to normalized space
    detected_norm, planned_norm, closest_norm, shaft_rad = \
        _transform_to_normalized(screw_data, pre_affine, post_affine,
                                 screw_pt_world, ped_pt_world)

    detected_entry, detected_tip = detected_norm[0], detected_norm[1]
    screw_pt, ped_pt = closest_norm[0], closest_norm[1]

    # 3. Build screw-aligned frame
    R = build_screw_frame(detected_entry, detected_tip, screw_pt, side=side)
    origin = screw_pt

    # 4. Resample CT
    volume, coords_1d = resample_ct(ct_data, ct_affine, R, origin, extent_mm)

    # 5. Slice definitions
    # Volume axes: volume[i_long, j_med, k_sup]
    e_long = R[:, 0]
    e_med = R[:, 1]
    e_sup = R[:, 2]

    mid = len(coords_1d) // 2

    # origin = screw_pt is on the screw surface, not the axis.
    # Shift axial/sagittal slices to pass through the screw centerline.
    axis_dir = detected_tip - detected_entry
    t = np.dot(origin - detected_entry, axis_dir) / np.dot(axis_dir, axis_dir)
    center_on_axis = detected_entry + t * axis_dir
    center_offset = center_on_axis - origin
    j_center_mm = np.dot(center_offset, e_med)
    k_center_mm = np.dot(center_offset, e_sup)
    j_center_idx = np.argmin(np.abs(coords_1d - j_center_mm))
    k_center_idx = np.argmin(np.abs(coords_1d - k_center_mm))

    slices = [
        {
            'title': 'Coronal',
            'normal': e_long,
            'horiz': e_med,
            'vert': e_sup,
            'img': volume[mid, :, :],
            'screw_type': 'circle',
            'compass': ('M', 'L', 'S', 'I'),
            'plane_origin': origin,
        },
        {
            'title': 'Axial',
            'normal': e_sup,
            'horiz': e_med,
            'vert': e_long,
            'img': volume[:, :, k_center_idx].T,
            'screw_type': 'rect',
            'compass': ('M', 'L', 'A', 'P'),
            'plane_origin': origin + k_center_mm * e_sup,
        },
        {
            'title': 'Sagittal',
            'normal': e_med,
            'horiz': e_long,
            'vert': e_sup,
            'img': volume[:, j_center_idx, :],
            'screw_type': 'rect',
            'compass': ('A', 'P', 'S', 'I'),
            'plane_origin': origin + j_center_mm * e_med,
        },
    ]

    # 6. Draw figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, sl in zip(axs, slices):
        contours = mesh_plane_contour(
            mesh_v, mesh_f, sl['plane_origin'], sl['normal'],
            sl['horiz'], sl['vert'], origin)

        if sl['screw_type'] == 'circle':
            cu, cv = project_to_2d(detected_entry, origin, sl['horiz'], sl['vert'])
            screw_ann = ('circle', (cu, cv, shaft_rad))
        elif sl['screw_type'] == 'rect':
            xs, ys = project_line_to_2d(detected_entry, detected_tip,
                                        origin, sl['horiz'], sl['vert'])
            screw_ann = ('rect', (xs, ys, shaft_rad))

        su, sv = project_to_2d(screw_pt, origin, sl['horiz'], sl['vert'])
        pu, pv = project_to_2d(ped_pt, origin, sl['horiz'], sl['vert'])
        closest = ((su, sv), (pu, pv))

        draw_slice(ax, sl['img'], coords_1d, contours, screw_ann,
                   closest, sl['title'], extent_mm, compass=sl['compass'])

    fig.suptitle('%s  breach=%.1f mm  angle=%.0f%s' % (screw_name, breach_dist, breach_angle, '\u00b0'),
                 fontsize=14)
    plt.tight_layout()

    # 7. Save
    if output:
        out_path = output
    else:
        acc_dir = accuracy_dir(analysis_dir)
        os.makedirs(acc_dir, exist_ok=True)
        out_path = os.path.join(acc_dir, 'breach_%s.png' % screw_name)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    log.debug('Saved to %s' % out_path)
    plt.close()


def _load_inputs(analysis_dir, level, side, preop=False):
    """Load CT, mesh, screw YAML, affines, and closest points needed for breach visualization."""
    o_dir = orient_level_dir(analysis_dir, level)
    preop_ldir = preop_level_dir(analysis_dir, level)
    reg_dir = registration_level_dir(analysis_dir, level)
    det_dir = detection_dir(analysis_dir)
    acc_dir = accuracy_dir(analysis_dir)

    # CT volume
    if preop:
        ct_img = nib.load(os.path.join(o_dir, 'preop.nii.gz'))
    else:
        postop_file = os.path.join(reg_dir, 'postop-reg.nii.gz')
        if os.path.isfile(postop_file):
            ct_img = nib.load(postop_file)
        else:
            log.warning('%s: postop-reg.nii.gz missing, falling back to preop CT', level)
            ct_img = nib.load(os.path.join(o_dir, 'preop.nii.gz'))
    ct_data = ct_img.get_fdata()
    ct_affine = ct_img.affine

    # Bone mesh (genus-1 from step 02, transformed to refined frame)
    mesh_v, mesh_f = igl.read_triangle_mesh(os.path.join(preop_ldir, 'preop_gen1.ply'))
    orig_aff = np.load(os.path.join(preop_ldir, 'preop_affine.npy'))
    refined_aff = np.load(os.path.join(o_dir, 'preop_affine-refined.npy'))
    delta = np.linalg.inv(refined_aff) @ orig_aff
    mesh_v = (delta[:3, :3] @ mesh_v.T).T + delta[:3, 3]

    # Screw YAML
    screw_name = '%s%s' % (level, side)
    with open(os.path.join(det_dir, '%s_screw.yml' % screw_name), 'r') as f:
        screw_data = yaml.safe_load(f)

    # Affines
    pre_affine = np.load(os.path.join(o_dir, 'preop_affine-refined.npy'))
    post_affine = np.load(os.path.join(reg_dir, 'postop-reg_affine.npy'))

    # Closest points from results.csv
    results = pd.read_csv(os.path.join(acc_dir, 'results.csv'), index_col=0)
    row = results.loc[screw_name]
    screw_pt_world = np.array([row['screw_pt_x'], row['screw_pt_y'], row['screw_pt_z']])
    ped_pt_world = np.array([row['ped_pt_x'], row['ped_pt_y'], row['ped_pt_z']])
    breach_dist = row['breach_dist']
    breach_angle = row.get('breach_angle', float('nan'))

    return (ct_data, ct_affine, mesh_v, mesh_f, screw_data,
            pre_affine, post_affine, screw_pt_world, ped_pt_world,
            breach_dist, breach_angle)


def _transform_to_normalized(screw_data, pre_affine, post_affine,
                             screw_pt_world, ped_pt_world):
    """Transform screw endpoints and closest points from world to normalized space."""
    detected_pts = np.array([screw_data['detected_entry'], screw_data['detected_tip']])
    detected_norm = transform_points_inverse(post_affine, detected_pts)

    planned_pts = np.array([screw_data['planned_entry'], screw_data['planned_tip']])
    planned_norm = transform_points_inverse(pre_affine, planned_pts)

    closest_pts = np.array([screw_pt_world, ped_pt_world])
    closest_norm = transform_points_inverse(pre_affine, closest_pts)

    shaft_rad = screw_data['shaft_rad']

    return detected_norm, planned_norm, closest_norm, shaft_rad


def build_screw_frame(detected_entry, detected_tip, origin, side='L'):
    """Build screw-aligned coordinate frame.

    Columns of R are [e_long, e_med, e_sup] where:
    - e_long points entry→tip (approximately anterior)
    - e_sup  points approximately superior (seeded from S axis)
    - e_med  points medial (toward midline) for both L and R screws
    """
    e_long = detected_tip - detected_entry
    e_long = e_long / np.linalg.norm(e_long)

    # Seed from superior axis [0,0,1], orthogonalize to e_long
    sup = np.array([0.0, 0.0, 1.0])
    e_sup = sup - np.dot(sup, e_long) * e_long
    norm = np.linalg.norm(e_sup)
    if norm < 1e-6:
        sup = np.array([0.0, 1.0, 0.0])
        e_sup = sup - np.dot(sup, e_long) * e_long
        norm = np.linalg.norm(e_sup)
    e_sup = e_sup / norm

    # e_med = cross(e_long, e_sup) → +R for L screws (medial)
    e_med = np.cross(e_long, e_sup)
    if side == 'R':
        e_med = -e_med

    R = np.column_stack([e_long, e_med, e_sup])
    return R


def resample_ct(ct_data, ct_affine, R, origin, extent_mm=15, spacing=0.5):
    """Resample CT onto screw-aligned grid."""
    n = int(2 * extent_mm / spacing) + 1
    coords_1d = np.linspace(-extent_mm, extent_mm, n)

    gi, gj, gk = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    grid_pts = np.column_stack([gi.ravel(), gj.ravel(), gk.ravel()])

    norm_pts = grid_pts @ R.T + origin

    inv_ct_aff = np.linalg.inv(ct_affine)
    vox_pts = norm_pts @ inv_ct_aff[:3, :3].T + inv_ct_aff[:3, 3]

    axes = [np.arange(s) for s in ct_data.shape]
    interp = RegularGridInterpolator(axes, ct_data, method='linear',
                                     bounds_error=False, fill_value=0)

    values = interp(vox_pts)
    volume = values.reshape(n, n, n)

    return volume, coords_1d


def mesh_plane_contour(mesh_v, mesh_f, plane_origin, plane_normal,
                       axis_horiz, axis_vert, origin_2d):
    """Intersect mesh with plane and project to 2D."""
    mesh = trimesh.Trimesh(mesh_v, mesh_f)
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return []

    contours_2d = []
    for entity in section.entities:
        pts_3d = section.vertices[entity.points]
        u = (pts_3d - origin_2d) @ axis_horiz
        v = (pts_3d - origin_2d) @ axis_vert
        contours_2d.append(np.column_stack([u, v]))
    return contours_2d


def project_to_2d(pt_3d, origin_3d, axis_horiz, axis_vert):
    """Project a 3D point onto 2D slice axes."""
    d = pt_3d - origin_3d
    return np.dot(d, axis_horiz), np.dot(d, axis_vert)


def project_line_to_2d(entry, tip, origin_3d, axis_horiz, axis_vert):
    """Project screw entry/tip line onto 2D."""
    u0, v0 = project_to_2d(entry, origin_3d, axis_horiz, axis_vert)
    u1, v1 = project_to_2d(tip, origin_3d, axis_horiz, axis_vert)
    return [u0, u1], [v0, v1]


def draw_slice(ax, img, coords_1d, contours, screw_annotation,
               closest_line, title, extent_mm, compass=None):
    """Draw a single slice panel."""
    extent = [-extent_mm, extent_mm, -extent_mm, extent_mm]

    ax.imshow(img.T, cmap='gray', vmin=-350, vmax=1150,
              origin='lower', extent=extent, aspect='equal')

    for contour in contours:
        closed = np.vstack([contour, contour[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'b-', linewidth=1.5)

    ann_type, ann_data = screw_annotation
    if ann_type == 'circle':
        cx, cy, r = ann_data
        circle = patches.Circle((cx, cy), r, fill=True, color='red', alpha=0.6)
        ax.add_patch(circle)
    elif ann_type == 'rect':
        xs, ys, r = ann_data
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        length = np.hypot(dx, dy)
        if length > 1e-6:
            px, py = -dy / length * r, dx / length * r
        else:
            px, py = r, 0
        corners = np.array([
            [xs[0] + px, ys[0] + py],
            [xs[1] + px, ys[1] + py],
            [xs[1] - px, ys[1] - py],
            [xs[0] - px, ys[0] - py],
        ])
        poly = Polygon(corners, closed=True, facecolor='red', alpha=0.6, edgecolor='none')
        ax.add_patch(poly)

    if closest_line is not None:
        (su, sv), (pu, pv) = closest_line
        ax.plot([su, pu], [sv, pv], 'g-', linewidth=2)
        ax.plot(su, sv, 'go', markersize=5)
        ax.plot(pu, pv, 'go', markersize=5)

    # Compass rose
    if compass is not None:
        right, left, up, down = compass
        offset = 0.925 * extent_mm
        kw = dict(fontsize=18, color='#4488ff', fontweight='bold',
                  ha='center', va='center')
        ax.text(offset, 0, right, **kw)
        ax.text(-offset, 0, left, **kw)
        ax.text(0, offset, up, **kw)
        ax.text(0, -offset, down, **kw)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-extent_mm, extent_mm)
    ax.set_ylim(-extent_mm, extent_mm)
    ax.axis('off')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize pedicle breach on CT slices')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--level', type=str, required=True)
    parser.add_argument('--side', type=str, required=True, choices=['L', 'R'])
    parser.add_argument('--extent_mm', type=float, default=15)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--preop', action='store_true',
                        help='Use preop CT instead of registered postop CT')
    args = parser.parse_args()

    specimen_dir = os.path.expanduser(args.specimen_dir)
    analysis_dir = os.path.join(specimen_dir, 'analysis')

    generate_breach_figure(analysis_dir, args.level, args.side,
                           extent_mm=args.extent_mm, output=args.output,
                           preop=args.preop)
