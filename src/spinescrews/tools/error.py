import logging
import os

import igl
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from bg3dtools.transforms_unified import transform_points_forward, transform_points_inverse
from bg3dtools.mesh.utils import submesh, join_meshes
from bg3dtools.pointclouds.fitting import align_axes

from spinescrews.tools.screw_models import Screw
from spinescrews.tools import ScrewMeasures, BreachMeasures, MeshLabels, dimR, dimA, dimS

log = logging.getLogger(__name__)


def signed_distance_to_mesh(point, mesh_v, mesh_f):
    """Signed Euclidean distance from a 3D point to a closed mesh.

    Negative = inside the mesh, positive = outside.
    """
    pt = np.atleast_2d(point).astype(np.float64)
    s, _, _ = igl.signed_distance(pt, mesh_v, mesh_f,
                                  sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
    return float(s)


def align_to_screw(pt0, pt1, v3=np.array([0., 0., 1.])):
    """Build a 4x4 frame aligned to a screw axis (pt0→pt1), used by measure_screw_error and normalize_to_left."""
    pt0 = pt0.reshape([1, 3])
    pt1 = pt1.reshape([1, 3])
    v3 = v3.reshape([1, 3])

    v2 = pt1 - pt0
    v2 /= np.sqrt(np.sum(v2 ** 2))
    v3 -= (v3 @ v2.T) * v2
    v3 /= np.sqrt(np.sum(v3 ** 2))
    v1 = np.cross(v2, v3)

    tform = np.row_stack((v1, v2, v3)).T
    tform = np.column_stack((tform, pt0.T))
    tform = np.row_stack((tform, [0, 0, 0, 1]))

    return tform


def angle_error(line_a, line_b):
    """Total 3D angle (degrees) between two screw lines given as (2,3) endpoint arrays."""
    from math import pi
    rad2deg = 180 / pi

    vec_a = line_a[1] - line_a[0]
    vec_b = line_b[1] - line_b[0]

    a = vec_a / np.sqrt(np.sum(vec_a**2))
    b = vec_b / np.sqrt(np.sum(vec_b**2))
    return rad2deg * np.arccos(np.dot(a, b))


def projected_angle_error(line_a, line_b):
    """Angular error projected onto sagittal, coronal, and axial planes (degrees)."""
    from math import pi
    rad2deg = 180 / pi

    vec_1 = line_a[1] - line_a[0]
    vec_2 = line_b[1] - line_b[0]
    # sagittal plane
    intended_a = np.arctan2(vec_1[dimA], vec_1[dimS])
    actual_a = np.arctan2(vec_2[dimA], vec_2[dimS])
    alpha = rad2deg * (intended_a - actual_a)
    # coronal plane
    intended_b = np.arctan2(vec_1[dimR], vec_1[dimS])
    actual_b = np.arctan2(vec_2[dimR], vec_2[dimS])
    beta = rad2deg * (intended_b - actual_b)
    # axial plane
    intended_c = np.arctan2(vec_1[dimR], vec_1[dimA])
    actual_c = np.arctan2(vec_2[dimR], vec_2[dimA])
    gamma = rad2deg * (intended_c - actual_c)

    return alpha, beta, gamma


def measure_screw_error(screw, ped_y, use_anatomic_axis=False):
    """
    :param ped_y: distance along screw shaft to pedicle measurement plane
    :param use_anatomic_axis:
    :return:
    """
    SCREW_AXIS = not use_anatomic_axis
    l_y = screw.shaft_len

    planned_pts = np.row_stack([screw.planned_entry, screw.planned_tip])
    detected_pts = np.row_stack([screw.detected_entry, screw.detected_tip])

    if screw.name[-1] == 'R':
        planned_pts[:, dimR] *= -1
        detected_pts[:, dimR] *= -1

    theta = angle_error(planned_pts, detected_pts)
    if SCREW_AXIS:
        ## For alignment with planned screw axis
        tform = align_to_screw(planned_pts[0], planned_pts[1])
        aligned_pts = transform_points_inverse(tform, detected_pts)  # puts x as AP, y as SI, z as ML
        e_x, e_y, e_z = aligned_pts[0]  # entry points
        t_x, t_y, t_z = aligned_pts[1]  # tip points

        # Entry Error
        entry_y = e_y  # AP offset is simply y coordinate
        # trigonometric ratio of points coplanar with entry
        r = (t_y / (t_y + e_y))
        entry_x = t_x + r * (e_x - t_x)
        entry_z = t_z + r * (e_z - t_z)

        # Pedicle Error
        r = (ped_y - e_y) / (t_y - e_y)
        ped_x = e_x + r * (t_x - e_x)
        ped_z = e_z + r * (t_z - e_z)

        # Tip Error
        tip_y = t_y - l_y
        # trigonometric ratio of points coplanar with tip
        r = (l_y - e_y) / (t_y - e_y)
        tip_x = e_x + r * (t_x - e_x)
        tip_z = e_z + r * (t_z - e_z)

        # angular error
        theta_s, theta_c, theta_a = projected_angle_error(np.array([[0, 0, 0], [0, 1, 0]]), aligned_pts)

    else:
        entry_x, entry_y, entry_z = detected_pts[0] - planned_pts[0]
        tip_x, tip_y, tip_z = detected_pts[1] - detected_pts[1]
        t = ped_y / l_y
        ped_x = t * tip_x + (1-t) * entry_x
        ped_z = t * tip_z + (1-t) * entry_z

        theta_s, theta_c, theta_a = projected_angle_error(planned_pts, detected_pts)


    log.debug('pedicle error: %.3f, %.3f' % (ped_x, ped_z))
    return ScrewMeasures(entry_x=entry_x, entry_y=entry_y, entry_z=entry_z,
                         ped_x=ped_x, ped_z=ped_z,
                         tip_x=tip_x, tip_y=tip_y, tip_z=tip_z,
                         theta=theta, theta_s=theta_s, theta_c=theta_c, theta_a=theta_a)


def _bone_contour_normals(path_2d, tform, reference_pt_3d):
    """Extract the pedicle-region bone contour and compute outward normal angles.

    Selects the closed entity closest to the screw axis, transforms it to 3D,
    projects onto the dimR-dimS plane, and computes outward normals there.
    This ensures the normal angles use the same atan2(x, z) convention as the
    rest of the medial/lateral classification code.

    Parameters
    ----------
    path_2d : trimesh.path.Path2D
        Cross-section path from mesh_multiplane (may contain multiple bodies).
    tform : ndarray (4, 4)
        Transform from section 2D coords to 3D space (from mesh_multiplane).
    reference_pt_3d : ndarray (3,)
        Point on the screw axis at this slice height — closest loop wins.

    Returns
    -------
    contour_xz : ndarray (N, 2) or None
        Contour vertices projected to (dimR, dimS).  None if no closed entity.
    angles : ndarray (N,) or None
        Outward normal angles in [0, 360) via atan2(n_R, n_S).  < 180 = medial.
    """
    # --- pick the closed entity whose 3D centroid is closest to screw axis ---
    best_dist = np.inf
    best_pts_3d = None
    for entity in path_2d.entities:
        if not entity.closed:
            continue
        pts_2d = path_2d.vertices[entity.points]
        if len(pts_2d) > 1 and np.allclose(pts_2d[0], pts_2d[-1], atol=1e-8):
            pts_2d = pts_2d[:-1]
        if len(pts_2d) < 3:
            continue
        pts_3d = transform_points_forward(tform, np.c_[pts_2d, np.zeros(len(pts_2d))])
        centroid = pts_3d.mean(axis=0)
        d = np.linalg.norm(centroid - reference_pt_3d)
        if d < best_dist:
            best_dist = d
            best_pts_3d = pts_3d
    if best_pts_3d is None:
        return None, None

    # --- project to the dimR-dimS (x-z) plane and compute normals there ---
    contour_xz = best_pts_3d[:, [dimR, dimS]]

    n = len(contour_xz)
    x, z = contour_xz[:, 0], contour_xz[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)

    tangents = np.roll(contour_xz, -1, axis=0) - np.roll(contour_xz, 1, axis=0)
    if signed_area > 0:  # CCW → outward = rotate CW: (t_z, -t_x)
        normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    else:  # CW → outward = rotate CCW: (-t_z, t_x)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)

    angles = np.degrees(np.arctan2(normals[:, 0], normals[:, 1])) % 360
    return contour_xz, angles


def distance_to_pedicle(bone_v: np.ndarray, bone_f: np.ndarray,
                        screw_pts: np.ndarray, shaft_rad: float, canal_v, canal_f,
                        mesh_output_dir=None,
                        head_rad: float = None, head_len: float = None):
    """
    Compute medial breach distance from screw shaft to pedicle canal wall.

    Only medial-direction breach candidates are considered. Direction is
    determined by the bone mesh cross-section normals: a breach/clearance
    point is medial if the nearest bone contour normal faces medially
    (angle < 180° in the screw-aligned frame after normalize_to_left).

    Parameters
    ----------
    bone_v, bone_f : ndarray
        Full bone mesh vertices and faces.
    screw_pts : ndarray
        [2x3] array of screw entry and tip.
    shaft_rad : float
        Screw shaft radius.
    canal_v, canal_f : ndarray
        Canal mesh vertices and faces.
    mesh_output_dir : str or None
        If provided, save breach meshes (bone, canal, screw, line PLYs) to this directory.
    head_rad : float or None
        Head radius for mesh export (defaults to 1.5 * shaft_rad).
    head_len : float or None
        Head length for mesh export (defaults to 12mm).

    Returns
    -------
    dist : float
        Medial breach distance (positive = breached, negative = clearance).
    ped_pt : ndarray
        Closest point on pedicle wall.
    screw_pt : ndarray
        Closest point on screw surface.
    """

    # construct trimesh objects to compute boolean difference
    shaft_v, shaft_f = Screw.build_cylinder(shaft_rad, screw_pts[0], screw_pts[1], N=64)
    shaft_mesh = trimesh.Trimesh(shaft_v, shaft_f)

    # find sections of screw that are in the canal
    canal_mesh = trimesh.Trimesh(canal_v, canal_f)
    breach = shaft_mesh.intersection(canal_mesh, check_volume=False)
    if isinstance(breach, trimesh.Scene):
        breach = breach.dump(concatenate=True)
    breach_v, breach_f = breach.vertices, breach.faces

    dist = 0
    ped_pt = screw_pt = None

    if breach_v.size > 0:
        log.info('    breach intersection: %d verts, %d faces', len(breach_v), len(breach_f))
        # find distance from breach faces to bone surface
        bc = igl.barycenter(breach_v, breach_f)
        d2 = igl.point_mesh_squared_distance(bc, canal_v, canal_f)[0]

        wall_mask = d2 < 0.0001
        log.info('    wall faces: %d / %d  (screw faces: %d)',
                 int(wall_mask.sum()), len(wall_mask), int((~wall_mask).sum()))

        wall_v, wall_f = submesh(breach_v, breach_f, wall_mask, return_indices=False)
        if not np.all(wall_mask):
            screw_v, screw_f = submesh(breach_v, breach_f, np.logical_not(wall_mask), return_indices=False)

            dist, ped_pt, screw_pt = breached_distance(
                screw_v, screw_f, wall_v, wall_f,
                screw_entry=screw_pts[0], screw_tip=screw_pts[1],
                bone_v=bone_v, bone_f=bone_f)

    # No medial-interior breach — compute medial-side clearance as negative
    # distance.  Applies when: no canal intersection, all-wall tangent, or
    # breach is entirely lateral (breached_distance returned 0).
    if dist <= 0:
        if breach_v.size > 0:
            log.info('    no medial breach despite canal intersection — computing medial clearance')
        else:
            log.info('    no breach (screw does not intersect canal)')

        # Slice bone mesh at coarse (1mm) intervals along the shaft range to
        # determine which shaft vertices face medially via bone contour normals.
        origin = np.zeros(3)
        normal = np.array([0, 1, 0])
        clearance_res = 1.0  # mm
        shaft_a = screw_pts[0, dimA] + 0.00001
        shaft_z = screw_pts[1, dimA] - 0.00001
        if shaft_z > shaft_a:
            n_clear = max(3, int((shaft_z - shaft_a) / clearance_res))
            clear_cuts = np.linspace(shaft_a, shaft_z, n_clear)
            bone_mesh_tm = trimesh.Trimesh(bone_v, bone_f)
            bone_clr_cuts, bone_clr_tforms, _ = trimesh.intersections.mesh_multiplane(
                bone_mesh_tm, origin, normal, clear_cuts)

            # Pre-compute bone contours + normals per slice
            bone_contours = []  # list of (contour_xz, angles) or None
            for ii in range(n_clear):
                if len(bone_clr_cuts[ii]) == 0:
                    bone_contours.append(None)
                    continue
                path_2d = trimesh.load_path(bone_clr_cuts[ii])
                ref_3d = np.array([0., clear_cuts[ii], 0.])
                contour_xz, angles = _bone_contour_normals(path_2d, bone_clr_tforms[ii], ref_3d)
                if contour_xz is None:
                    bone_contours.append(None)
                    continue
                bone_contours.append((contour_xz, angles))

            # For each shaft vertex, find its nearest cut, look up nearest bone
            # contour point, check if normal is medial-facing.
            shaft_y = shaft_v[:, dimA]
            cut_idx = np.clip(np.searchsorted(clear_cuts, shaft_y) - 1, 0, n_clear - 1)
            medial_mask = np.zeros(len(shaft_v), dtype=bool)
            for vi in range(len(shaft_v)):
                ci = cut_idx[vi]
                bc = bone_contours[ci]
                if bc is None:
                    # fallback: use shaft-angle filter
                    axis_vec = screw_pts[1] - screw_pts[0]
                    axis_len2 = np.dot(axis_vec, axis_vec)
                    t_param = np.dot(shaft_v[vi] - screw_pts[0], axis_vec) / axis_len2
                    center = screw_pts[0] + t_param * axis_vec
                    rad = shaft_v[vi] - center
                    ang = np.degrees(np.arctan2(rad[dimR], rad[dimS])) % 360
                    medial_mask[vi] = ang < 180
                    continue
                contour_xz, normal_angles = bc
                pt_xz = shaft_v[vi, [dimR, dimS]].reshape(1, -1)
                dists = np.sum((contour_xz - pt_xz) ** 2, axis=1)
                nearest = np.argmin(dists)
                medial_mask[vi] = normal_angles[nearest] < 180
        else:
            medial_mask = np.zeros(len(shaft_v), dtype=bool)

        if medial_mask.any():
            medial_v = shaft_v[medial_mask]
            d2, _, surfpts = igl.point_mesh_squared_distance(medial_v, canal_v, canal_f)

            ii = np.argmin(d2)
            dist = -np.sqrt(d2[ii])
            ped_pt = surfpts[ii]
            screw_pt = medial_v[ii]
            log.info('    medial clearance: %.3f mm (%d/%d medial)',
                     dist, int(medial_mask.sum()), len(shaft_v))
        else:
            log.warning('    no medial shaft vertices — cannot compute clearance')
            ped_pt = np.mean(shaft_v, axis=0)
            screw_pt = ped_pt.copy()

    if not (screw_pts[0, dimA] + 0.001 < screw_pt[dimA] < screw_pts[1, dimA] - 0.001):
        log.warning('breach point is at screw extremity; geometric assumptions invalid')

    if mesh_output_dir is not None:
        # save out meshes for offline review (Blender/MeshLab)
        os.makedirs(mesh_output_dir, exist_ok=True)
        screw_axis = (screw_pts[1] - screw_pts[0]) / np.linalg.norm(screw_pts[1] - screw_pts[0])
        hr = head_rad if head_rad is not None else 1.5 * shaft_rad
        hl = head_len if head_len is not None else 12.
        head_v, head_f = Screw.build_cylinder(hr, screw_pts[0], screw_pts[0] - hl * screw_axis, N=64)
        full_screw_v, full_screw_f = join_meshes(shaft_v, shaft_f, head_v, head_f)
        line_v, line_f = Screw.build_cylinder(0.15, ped_pt, screw_pt, N=64)
        igl.write_triangle_mesh(os.path.join(mesh_output_dir, 'bone.ply'), bone_v, bone_f)
        igl.write_triangle_mesh(os.path.join(mesh_output_dir, 'canal.ply'), canal_v, canal_f)
        igl.write_triangle_mesh(os.path.join(mesh_output_dir, 'screw.ply'), full_screw_v, full_screw_f)
        igl.write_triangle_mesh(os.path.join(mesh_output_dir, 'line.ply'), line_v, line_f)
        log.debug('Saved breach meshes to %s, dist=%.3f' % (mesh_output_dir, dist))

    return dist, ped_pt, screw_pt


def breached_distance(screw_v, screw_f, ped_v, ped_f,
                      screw_entry, screw_tip,
                      bone_v, bone_f,
                      resolution=0.05):
    """Max medial-interior breach distance between screw surface and pedicle wall.

    Two filters are applied to each screw-point / wall-point pair:

    1. **Medial filter** — the screw surface point must be on the medial half of
       the shaft cross-section (radial angle < 180° after projecting onto the
       detected screw axis).
    2. **Bone-normal filter** — the matched pedicle wall point's nearest bone
       contour normal must face medially (angle < 180°), ensuring it is on
       the canal-facing side of the pedicle rather than the lateral side.

    Returns (0, centroid, centroid) when no valid candidates exist.
    """
    # Precompute screw axis direction for per-slice center projection.
    axis_vec = screw_tip - screw_entry
    axis_len2 = np.dot(axis_vec, axis_vec)

    # cut screw and pedicle meshes with planes perpendicular to (planned) screw axis
    origin = np.zeros(3)
    normal = np.array([0, 1, 0])
    a = max(np.min(ped_v[:, dimA]), np.min(screw_v[:, dimA])) + 0.00001
    z = min(np.max(ped_v[:, dimA]), np.max(screw_v[:, dimA])) - 0.00001
    if z <= a:
        raise RuntimeError('breach geometry invalid: z (%.3f) must be > a (%.3f)' % (z, a))
    num_cuts = max(3, int((z-a) / resolution))  # cut at 0.1mm resolution
    cuts = np.linspace(a, z, num_cuts)

    ped_mesh = trimesh.Trimesh(ped_v, ped_f)
    ped_cuts, ped_tforms, _ = trimesh.intersections.mesh_multiplane(ped_mesh, origin, normal, cuts)
    screw_mesh = trimesh.Trimesh(screw_v, screw_f)
    screw_cuts, screw_tforms, _ = trimesh.intersections.mesh_multiplane(screw_mesh, origin, normal, cuts)

    # Slice bone mesh at the same cut planes for normal-based medial/lateral classification
    bone_mesh_tm = trimesh.Trimesh(bone_v, bone_f)
    bone_cuts, bone_tforms, _ = trimesh.intersections.mesh_multiplane(bone_mesh_tm, origin, normal, cuts)

    # Pre-compute bone contours + normal angles for each cut
    bone_contour_data = []  # list of (contour_xz, angles) or None
    for ii in range(num_cuts):
        if len(bone_cuts[ii]) == 0:
            bone_contour_data.append(None)
            continue
        path_2d = trimesh.load_path(bone_cuts[ii])
        ref_3d = np.array([0., cuts[ii], 0.])
        contour_xz, angles = _bone_contour_normals(path_2d, bone_tforms[ii], ref_3d)
        if contour_xz is None:
            bone_contour_data.append(None)
            continue
        bone_contour_data.append((contour_xz, angles))

    max_d, screw_pt, ped_pt = 0, np.mean(screw_v, axis=0), np.mean(ped_v, axis=0)
    n_medial, n_lateral, n_no_contour = 0, 0, 0
    for nn in range(num_cuts):
        ped_cut, screw_cut = ped_cuts[nn], screw_cuts[nn]
        ped_tform, screw_tform = ped_tforms[nn], screw_tforms[nn]
        if (len(ped_cut) == 0) or (len(screw_cut) == 0):
            continue

        ped_pts = trimesh.load_path(ped_cut).vertices
        ped_pts = sample_path(ped_pts, resolution)
        ped_pts = np.c_[ped_pts, np.zeros(len(ped_pts))]
        ped_pts = transform_points_forward(ped_tform, ped_pts)

        screw_pts = trimesh.load_path(screw_cut).vertices
        screw_pts = sample_path(screw_pts, resolution)
        screw_pts = np.c_[screw_pts, np.zeros(len(screw_pts))]
        screw_pts = transform_points_forward(screw_tform, screw_pts)

        D = cdist(screw_pts, ped_pts)
        min_dists = np.min(D, axis=1)        # per-screw-point distance to nearest wall
        closest_idx = np.argmin(D, axis=1)   # index of nearest wall point

        # Filter: bone-normal based medial classification.
        bc_data = bone_contour_data[nn]
        if bc_data is not None:
            contour_xz, normal_angles = bc_data
            tree = cKDTree(contour_xz)
            # Check pedicle wall points: is the nearest bone contour normal medial?
            ped_xz = ped_pts[:, [dimR, dimS]]
            _, bone_idx = tree.query(ped_xz)
            medial_ped = normal_angles[bone_idx] < 180
            # Apply per screw-to-ped pair
            valid_mask = medial_ped[closest_idx]
        else:
            # Fallback to shaft-angle filter when no bone contour available
            n_no_contour += len(screw_pts)
            t = np.dot(screw_pts - screw_entry, axis_vec) / axis_len2
            axis_centers = screw_entry + t[:, None] * axis_vec
            radial = screw_pts - axis_centers
            angles = np.arctan2(radial[:, 0], radial[:, 2]) * 180 / np.pi % 360
            valid_mask = angles < 180

        n_medial += int(valid_mask.sum())
        n_lateral += int((~valid_mask).sum())

        if not valid_mask.any():
            continue

        valid_dists = min_dists[valid_mask]
        best = np.argmax(valid_dists)
        ss = np.where(valid_mask)[0][best]
        pp = closest_idx[ss]
        if valid_dists[best] > max_d:
            max_d = valid_dists[best]
            screw_pt = screw_pts[ss]
            ped_pt = ped_pts[pp]

    log.info('    breached_distance: max=%.3f mm  (%d medial, %d lateral, %d no-contour-fallback)',
             max_d, n_medial, n_lateral, n_no_contour)
    return max_d, ped_pt, screw_pt


def sample_path(nodes, resolution):
    """
    Sample points on a path defined by sequential nodes in N-dimensional space.

    Parameters:
    - nodes: (M, N) numpy array representing M nodes in N-dimensional space.
    - num_samples: Number of points to sample along the path.

    Returns:
    - sampled_points: (num_samples, N) numpy array of points sampled along the path.
    """
    if resolution <= 0.00001:
        raise ValueError('resolution must be greater than 0.00001')
    # Calculate the distances between consecutive nodes
    segments = np.diff(nodes, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)

    # Normalize segment lengths to get the proportion of each segment
    cumulative_length = np.r_[0, np.cumsum(segment_lengths)]
    total_length = np.sum(segment_lengths)
    if total_length < resolution:
        return np.mean(nodes, axis=0, keepdims=True)

    samples = np.arange(0.00001, total_length, resolution)
    num_samples = len(samples)

    # Find the segment each sample belongs to
    segment_indices = np.searchsorted(cumulative_length, samples) - 1

    # Interpolate sampled points within their respective segments
    sampled_points = np.zeros((num_samples, nodes.shape[1]))
    for i in range(num_samples):
        segment_index = segment_indices[i]
        t = (samples[i] - cumulative_length[segment_index]) / segment_lengths[segment_index]
        start_node = nodes[segment_index]
        end_node = nodes[segment_index + 1]
        sampled_points[i] = start_node + t * (end_node - start_node)

    return sampled_points



