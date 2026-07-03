"""The `Vertebra` class: per-level geometry extracted from a segmented CT.

Builds a genus-1 (canal-threaded) surface mesh for a single vertebra and provides skeleton-based
canal-loop detection, orientation normalization, affine transforms between CT and canonical
frames, and volume cropping used throughout alignment and accuracy measurement.
"""

from os.path import join
import logging
import warnings

log = logging.getLogger(__name__)
from collections import namedtuple

import igl
import numpy as np
import nibabel as nib
from dipy.align._public import AffineMap
from skimage.measure import marching_cubes
import skimage.morphology as image_morph
from scipy import ndimage
import scipy.spatial as spatial
import igraph as ig
from scipy.stats import mode
from scipy import sparse


from bg3dtools.mesh.utils import per_vertex_smoothing, per_vertex_normals, get_genus, surface_sample, submesh, mesh_volume, as_igl_faces
from bg3dtools.mesh.barycentric import bc2sparse
from bg3dtools.mesh.mesh_io import read_triangle_mesh, read_colored_plyfile
from bg3dtools.mesh.registration import nonrigid_ICP, surface_match
from bg3dtools.pointclouds.fitting import project_to_plane, fit_plane_to_points
from bg3dtools.utils import ConvergenceScheduler
from bg3dtools.transforms_unified import transform_points_forward, transform_points_inverse
from bg3dtools.pointclouds.quantize import convert_to_points, voxelize
from bg3dtools.graphs import skeleton_to_graph, redistribute_evenly, smooth_loop, get_longest_path_fast
from spinescrews.tools.nifti_utils import nonzero_box


Template = namedtuple('Template', ['raw', 'processed', 'labels'])
VertebralLabels = namedtuple('VertebralLabels', ['canal', 'ped_right', 'ped_left'])


def _unwind_erosion(current_v, faces, target_stack, model_weight=0.0002, n_samples=30000):
    """Reverse-walk erosion stack to recover anatomy lost during genus-1 enforcement.

    Deforms the genus-1 mesh toward each earlier erosion step via nonrigid ICP.
    Faces are unchanged so genus stays locked. Skips iter 0 (too noisy).
    """
    _nricp_log = logging.getLogger('nonrigid_reg')
    _prev_level = _nricp_log.level
    _nricp_log.setLevel(logging.WARNING)
    for target_v, target_f in reversed(target_stack[1:]):
        sample_map = surface_sample(target_v, target_f, N=n_samples)[0]
        pts = sample_map @ target_v
        nrm = sample_map @ per_vertex_normals(target_v, target_f)
        current_v = nonrigid_ICP(pts, faces, current_v,
                                 pt_normals=nrm, model_weight=model_weight,
                                 rad=0.001, converge_thresh=0.1)[0]

    _nricp_log.setLevel(_prev_level)
    return current_v


def _trim_border(mask):
    """Zero out all six faces of a 3-D binary mask (in-place)."""
    mask[0] = 0
    mask[-1] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[:, :, 0] = 0
    mask[:, :, -1] = 0
    return mask


def _compute_distance_field(seg_vol):
    """Signed distance field (negative inside) from binary segmentation."""
    seg_bin = seg_vol.get_fdata() > 0
    pitch = np.diag(seg_vol.affine[:3, :3]).tolist()
    trans = seg_vol.affine[:3, 3].reshape([1, 3])
    sdf = (ndimage.distance_transform_edt(~seg_bin, pitch)
           - ndimage.distance_transform_edt(seg_bin, pitch))
    return sdf, pitch, trans


def _prune_non_canal_nodes(canal_verts, cycles, coords, graph):
    """Remove skeleton nodes belonging to non-canal cycles.

    Canal loop nodes are always kept.

    Returns (support_pts, n_pruned).
    """
    canal_nodes = set(canal_verts)
    prune_nodes = set()
    for cycle in cycles:
        prune_nodes.update(n for n in cycle if n not in canal_nodes)

    keep_nodes = sorted(set(range(len(coords))) - prune_nodes)

    # Remove floating points: keep only the component connected to the canal
    sub = graph.induced_subgraph(keep_nodes)
    components = sub.connected_components()
    keep_set = {v: i for i, v in enumerate(keep_nodes)}
    canal_sub_idx = keep_set[canal_verts[0]]
    main_comp = components.membership[canal_sub_idx]
    keep_nodes = [keep_nodes[i] for i, m in enumerate(components.membership)
                  if m == main_comp]

    support_pts = coords[keep_nodes]
    return support_pts, len(coords) - len(keep_nodes)


def _mcb_to_ordered_vertices(graph, edge_indices):
    """Convert igraph MCB edge-index cycle to ordered vertex list."""
    e0 = graph.es[edge_indices[0]]
    if len(edge_indices) == 1:
        return [e0.source, e0.target]
    e1 = graph.es[edge_indices[1]]
    if e0.target in (e1.source, e1.target):
        verts = [e0.source, e0.target]
    else:
        verts = [e0.target, e0.source]
    for eid in edge_indices[1:]:
        e = graph.es[eid]
        verts.append(e.target if e.source == verts[-1] else e.source)
    if verts[-1] == verts[0]:
        verts.pop()
    return verts


def _penalty(x, thresh, k=10):
    """Soft penalty: ~0 when x < thresh, ~1 when x > thresh."""
    return 1 / (1 + np.exp(-k * (x / thresh - 1)))


def _assess_canal_loop(loop_pts, orig_pts, skel_thresh, iter_idx):
    """Geometric quality score for a candidate canal loop.

    Returns a continuous cost (lower = better, 0 = perfect).
    Each criterion contributes ~0 when satisfied, ~1 when violated.
    """
    median_dist = np.median(spatial.cKDTree(loop_pts).query(orig_pts, k=1)[0])
    c_interior = _penalty(median_dist, skel_thresh)

    centroid = loop_pts.mean(axis=0)
    centroid_offset = np.linalg.norm(centroid - np.mean(orig_pts, axis=0))
    c_centered = _penalty(centroid_offset, 2 * skel_thresh)
    radii = np.linalg.norm(loop_pts - centroid, axis=1)
    mean_radius = radii.mean()

    plane = fit_plane_to_points(loop_pts)
    residuals = loop_pts @ plane[:3] + plane[3]
    rms_residual = np.sqrt(np.mean(residuals ** 2))
    c_planar = _penalty(rms_residual, mean_radius / 3)
    c_circular = _penalty(radii.std(), mean_radius / 3)

    bone_extent = np.percentile(orig_pts, 95, axis=0) - np.percentile(orig_pts, 5, axis=0)
    bone_size = np.linalg.norm(bone_extent)
    diameter_ratio = 2 * mean_radius / bone_size
    c_size = _penalty(0.2, diameter_ratio) + _penalty(diameter_ratio, 0.75)

    cost = c_interior + c_centered + c_planar + c_circular + c_size

    log.debug('  iter %d: cost=%.3f  interior=%.2f (med=%.2f thresh=%.2f) '
              'centered=%.2f (dist=%.2f) '
              'planar=%.2f (%.3f) circular=%.2f (%.3f) size=%.2f (r=%.2f bone=%.2f)',
              iter_idx, cost, c_interior, median_dist, skel_thresh,
              c_centered, centroid_offset,
              c_planar, rms_residual / mean_radius,
              c_circular, radii.std() / mean_radius,
              c_size, mean_radius, bone_size)

    return cost


def _find_body_center(label_mask, affine):
    """Locate vertebral body center via smoothed distance-field peak.

    Returns body_pts (N, 3) ndarray of core voxel positions.
    """
    sigma = 1 / np.abs(np.diag(affine[:3, :3]))
    smoothed = ndimage.gaussian_filter(label_mask, 8 * sigma)
    core = smoothed > np.percentile(smoothed[smoothed > 0.05], 99.5)
    return convert_to_points(core, affine)


def _build_anatomical_frame(plane, midpoint, body_pts, z_up):
    """Construct a 4x4 RAS anatomical frame from canal plane and body center.

    z = canal plane normal (cranial), y = toward body, x = right (cross product).
    """
    if (z_up and plane[2] < 0) or (not z_up and plane[2] > 0):
        plane = plane * -1

    z_vec = plane[:3]
    body_projected = project_to_plane(plane, body_pts)[0]
    y_vec = np.mean(body_projected, axis=0) - midpoint
    y_vec /= np.sqrt(np.sum(y_vec ** 2))
    x_vec = np.cross(y_vec, z_vec)

    tform = np.eye(4)
    tform[:3, :3] = np.vstack((x_vec, y_vec, z_vec)).T
    tform[:3, 3] = midpoint
    return tform


def _extract_surface(seg_mask, pitch, trans):
    """Marching cubes -> largest manifold patch -> winding fix -> smooth -> genus."""
    verts, faces, _, _ = marching_cubes(seg_mask, spacing=pitch)
    faces = as_igl_faces(faces)  # marching_cubes gives int32 faces on Windows; keep igl calls int64
    p = igl.extract_manifold_patches(faces)
    verts, faces, f_idx, v_idx = submesh(verts, faces, p[1] == mode(p[1])[0])
    verts += trans
    faces = faces[:, [0, 2, 1]]
    verts = per_vertex_smoothing(verts, faces)
    genus = get_genus(verts, faces)
    if not isinstance(genus, int):
        raise RuntimeError('genus is not an integer; malformed mesh')
    return verts, faces, genus


class Vertebra:
    def __init__(self, name: str, affine=None):
        """Initialize a vertebra with a level name and optional world-to-normalized affine."""
        self.name = name

        self.img_normalized = nib.Nifti1Image(np.array([]), np.eye(4))
        self.seg_normalized = nib.Nifti1Image(np.array([]), np.eye(4))
        self.affine = np.eye(4) if affine is None else affine

        self.verts_ = None  # in normalized coordinates
        self.faces_ = None

        self.template2mesh = None
        self.mesh2template = None

    def import_volumes(self, raw_img, label_vol=None):
        """Resample raw CT (and optional segmentation) into the vertebra's normalized frame."""
        self.img_normalized = self.rotate_and_crop(raw_img, self.affine)
        if label_vol is not None:
            self.seg_normalized = self.rotate_and_crop(label_vol, self.affine, sampling='nearest')

    @staticmethod
    def get_mesh(seg_vol: nib.Nifti1Image, smoothness=2, offset=0.0):
        """Single-pass genus-agnostic mesh extraction.

        Returns (verts, faces).
        """
        if np.prod(seg_vol.shape) == 0:
            raise ValueError('segmentation volume is empty')
        if not (0 <= smoothness <= 10):
            raise ValueError('smoothness should be in range [0, 10], got %s' % smoothness)

        inside_dist, pitch, trans = _compute_distance_field(seg_vol)

        seg_mask = (inside_dist < offset)
        seg_mask = ndimage.binary_opening(seg_mask, iterations=2)
        seg_smooth = ndimage.gaussian_filter(seg_mask.astype(np.float32), smoothness / np.array(pitch))
        seg_mask = _trim_border(seg_smooth > 0.5)

        if not np.any(seg_mask):
            raise ValueError('get_mesh: mask empty after thresholding')

        verts, faces, genus = _extract_surface(seg_mask, pitch, trans)
        log.debug('get_mesh: genus = %d', genus)

        scale = np.linalg.norm(np.std(verts, axis=0))
        sa2 = np.sum(igl.doublearea(verts / scale, faces))
        final_target = int(400 * sa2)
        log.debug('qslim to %d faces', final_target)
        verts, faces = igl.qslim(verts, faces, final_target)[1:3]

        return verts, faces

    @staticmethod
    def get_mesh_genus1(seg_vol: nib.Nifti1Image, support_pts, smoothness=1.0, offset=0.0):
        """Iterative erosion targeting genus=1 with skeleton support field.

        Returns (verts, faces, inflated_v, f_inflated).
        """
        import time as _time
        _t0 = _time.perf_counter()

        if np.prod(seg_vol.shape) == 0:
            raise ValueError('segmentation volume is empty')
        if not (0 <= smoothness <= 10):
            raise ValueError('smoothness should be in range [0, 10], got %s' % smoothness)

        inside_dist, pitch, trans = _compute_distance_field(seg_vol)

        # Skeleton support field: gaussian-smoothed voxelized skeleton
        vox_pts = transform_points_inverse(seg_vol.affine, support_pts)
        support_mask = voxelize(vox_pts, seg_vol.shape).astype(np.float32)
        sigma_mm = 2.0  # support spread radius in mm
        support_field = ndimage.gaussian_filter(1000 * support_mask, sigma_mm / np.array(pitch))
        # Scale: peak reinforcement = mean depth at skeleton locations
        depth_at_skeleton = np.mean(inside_dist[support_mask > 0])
        supported_dist = inside_dist + support_field * (depth_at_skeleton / support_field.max())  # negative values deepen skeleton region

        # Iterative erosion targeting genus=1
        max_genus_iters = 12
        completed, v0, f0 = False, None, None
        best_verts, best_faces, best_genus = None, None, None
        mesh_stack = []
        final_iter = 0

        for ii in range(max_genus_iters):
            dynamic_thresh = offset - ii * np.mean(pitch)/2

            seg_mask = (supported_dist < dynamic_thresh)
            # Skip opening — it destroys thin bridges (pedicles/laminae)
            seg_smooth = ndimage.gaussian_filter(seg_mask.astype(np.float32), smoothness / np.array(pitch))
            seg_mask = _trim_border(seg_smooth > 0.5)

            if not np.any(seg_mask):
                log.warning('get_mesh_genus1: mask empty at iter %d (thresh=%.3f), stopping',
                            ii, dynamic_thresh)
                break

            verts, faces, genus = _extract_surface(seg_mask, pitch, trans)
            log.debug('iter %d: thresh = %.3f  genus = %d', ii, dynamic_thresh, genus)

            if ii == 0:
                v0, f0 = verts, faces
            if genus >= 1 and (best_genus is None or genus < best_genus):
                best_verts, best_faces, best_genus = verts.copy(), faces.copy(), genus
                final_iter = ii
            mesh_stack.append((verts.copy(), faces.copy()))

            completed = genus == 1
            if genus < 2:
                break

        # Fallback selection
        if not completed:
            if best_verts is not None:
                log.warning('get_mesh_genus1: genus=1 not reached after %d iters, '
                            'using best mesh (genus=%d)', len(mesh_stack), best_genus)
                verts, faces = best_verts, best_faces
            elif v0 is not None:
                log.warning('get_mesh_genus1: genus=1 not reached, '
                            'using iter-0 mesh (genus=%d)', get_genus(v0, f0))
                verts, faces = v0, f0

        # Compute surface-area-based face targets
        scale = np.linalg.norm(np.std(verts, axis=0))
        sa2 = np.sum(igl.doublearea(verts / scale, faces))
        intermediate_target = int(1200 * sa2)
        final_target = int(400 * sa2)

        # Step 1: decimate halfway (shortest-edge removal → good element quality)
        log.debug('decimate to %d faces (intermediate)', intermediate_target)
        v_med, f_med = igl.decimate(verts, faces, intermediate_target)[1:3]

        # Step 2: unwind erosion to recover lost anatomy
        v_inflated, f_inflated = v_med.copy(), f_med.copy()
        if completed and len(mesh_stack) > 2:
            v_inflated = _unwind_erosion(v_med.copy(), f_med, mesh_stack[:-1])
            vol_eroded = abs(mesh_volume(v_med, f_med))
            vol_inflated = abs(mesh_volume(v_inflated, f_inflated))
            log.info('unwind: volume %.0f → %.0f mm³ (×%.2f)',
                     vol_eroded, vol_inflated, vol_inflated / vol_eroded)

        # Step 3: decimate to final count
        log.debug('decimate to %d faces (final)', final_target)
        v_small, f_small = igl.decimate(v_med, f_med, final_target)[1:3]

        # Step 4: small↔inflated correspondence
        _, fidx, bc = surface_match(v_med, v_small, f_small)
        small2med = bc2sparse(f_small, fidx, bc, nV=len(v_small))   # (n_med, n_small)

        _, fidx, bc = surface_match(v_small, v_med, f_med)
        med2small = bc2sparse(f_med, fidx, bc, nV=len(v_med))       # (n_small, n_med)

        log.info('genus1 done in %.1fs: genus=%d at iter %d, %d verts %d faces',
                 _time.perf_counter() - _t0, best_genus or get_genus(verts, faces),
                 final_iter, len(v_small), len(f_small))
        return v_small, f_small, v_inflated, f_inflated, small2med, med2small

    def set_mesh(self, v, f):
        """Directly assign mesh vertices and faces (e.g. after loading from disk)."""
        self.verts_ = v
        self.faces_ = f

    @property
    def verts(self):
        """Mesh vertices in normalized coordinates; extracted on first access if not set."""
        if self.verts_ is None:
            self.verts_, self.faces_ = self.get_mesh(self.seg_normalized)
        return self.verts_

    @property
    def faces(self):
        """Mesh faces; extracted on first access if not set."""
        if self.faces_ is None:
            self.verts_, self.faces_ = self.get_mesh(self.seg_normalized)
        return self.faces_

    def save(self, folder, descriptor, save_volume=True):
        """Save vertebra state (affine, volumes, mesh, correspondence matrices) to disk."""
        if self.img_normalized is None:
            raise RuntimeError('Vertebra not initialized')

        basename = join(folder, descriptor)
        if save_volume:
            ct = np.clip(np.round(self.img_normalized.get_fdata()),
                         -32768, 32767).astype(np.int16)
            nib.save(nib.Nifti1Image(ct, self.img_normalized.affine),
                     basename + '.nii.gz')
        np.save(str(basename + '_affine.npy'), self.affine)

        if self.seg_normalized is not None and np.prod(self.seg_normalized.shape) > 0:
            nib.save(self.seg_normalized, basename + '_seg.nii.gz')

        if self.verts_ is not None and self.faces_ is not None:
            igl.write_triangle_mesh(basename + '_gen1.ply', self.verts_, self.faces_)

        if self.template2mesh is not None and self.mesh2template is not None:
            sparse.save_npz(join(folder, 'template2bone.npz'), self.template2mesh)
            sparse.save_npz(join(folder, 'bone2template.npz'), self.mesh2template)

    @staticmethod
    def load(folder, level_name, descriptor):
        """Load a vertebra from disk (affine, volumes, mesh, correspondence if present)."""
        vert = Vertebra(level_name)
        basename = join(folder, descriptor)

        vert.affine = np.load(basename + '_affine.npy')

        try:
            vert.img_normalized = nib.load(basename + '.nii.gz')
        except FileNotFoundError:
            pass

        try:
            vert.seg_normalized = nib.load(basename + '_seg.nii.gz')
        except FileNotFoundError:
            pass

        try:
            v, f = read_triangle_mesh(str(basename + '_gen1.ply'))
            vert.set_mesh(v, f)
        except (FileNotFoundError, ValueError):
            pass

        try:
            vert.template2mesh = sparse.load_npz(join(folder, 'template2bone.npz')).tocsr()
            vert.mesh2template = sparse.load_npz(join(folder, 'bone2template.npz')).tocsr()
        except FileNotFoundError:
            pass

        return vert

    @staticmethod
    def binarize_label(seg_vol, val):
        """Extract a single vertebral label as a binary NIfTI mask."""
        mask = (seg_vol.get_fdata() == val).astype(np.uint8)
        label = nib.Nifti1Image(mask, affine=seg_vol.affine)
        return label

    @staticmethod
    def rotate_and_crop(image, affine, dimensions=(200, 200, 200), output_aff=None, sampling='linear'):
        """
        Wrapper for AffineMap (from dipy.align._public).
        :param image: input image object with 3D data and affine transform to world coordinates
        :param tform: mapping from input to output location (in world coordinates)
        :param dimensions: (200, 200, 200) output dimensions
        :param output_aff: (optional) transform from output image to world coordinates
        :param sampling: 'nearest' or 'linear' (default linear)
        :return: nib.Nifti1Image object with resampled data
        """
        if output_aff is None:
            output_aff = np.eye(4) / 2
            output_aff[:, 3] = [-50, -50, -60, 1]

        aff_map = AffineMap(affine, domain_grid_shape=dimensions, domain_grid2world=output_aff)
        label_crop = aff_map.transform(image.get_fdata(), interpolation=sampling, image_grid2world=image.affine)
        return nib.Nifti1Image(label_crop, output_aff)

    @staticmethod
    def normalize_orientation(label: nib.Nifti1Image, z_up=True) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        :param label: binary mask of a single vertebra
        :param z_up: True if z-axis points cranially, False if z-axis points caudally
        :return: affine transformation to being vertebra into normalized coordinate system
        """
        subvol = nonzero_box(label)

        log.debug('find center of vertebral body')
        label_mask = (subvol.get_fdata() > 0).astype(np.float32)
        body_pts = _find_body_center(label_mask, subvol.affine)

        initial_loop, support_skeleton = Vertebra.initialize_canal_loop(label_mask, subvol.affine)

        verts, faces, inflated_v, inflated_f, small2med, med2small = \
            Vertebra.get_mesh_genus1(subvol, support_skeleton)
        plane, midpoint, loop_pts = fit_canal(initial_loop, verts, faces)

        tform = _build_anatomical_frame(plane, midpoint, body_pts, z_up)
        verts = transform_points_inverse(tform, verts)
        if inflated_v is not None:
            inflated_v = transform_points_inverse(tform, inflated_v)
        return tform, verts, faces, inflated_v, inflated_f, small2med, med2small

    @staticmethod
    def initialize_canal_loop(bin_mask, affine):
        """Find canal loop and pruned skeleton support points.

        Evaluates both the largest loop and longest path at each iteration,
        tracking the best candidate by continuous cost. Pruning runs once
        after the best candidate is selected.

        Returns
        -------
        canal_loop : (N, 3) ndarray
            Points around the spinal canal (for fit_canal).
        support_pts : (M, 3) ndarray
            Skeleton points with non-canal loop nodes removed (for get_mesh_genus1).
        """
        import time as _time
        _t0 = _time.perf_counter()
        log.debug('initialize spinal canal loop')
        orig_pts = convert_to_points(bin_mask, affine)
        bin_mask = bin_mask.astype(np.float32)

        # distance to surface
        pitch = np.diag(affine[:3, :3]).tolist()
        _t1 = _time.perf_counter()
        inside_dist = ndimage.distance_transform_edt(bin_mask == 1, pitch)
        outside_dist = ndimage.distance_transform_edt(bin_mask == 0, pitch)
        log.debug('  distance transforms done in %.2fs (vol shape %s)', _time.perf_counter() - _t1, bin_mask.shape)

        skel_thresh = 3 * np.percentile(inside_dist[bin_mask > 0], 90)

        best_cost = float('inf')
        best_iter = -1
        best_loop_pts = None
        best_loop_verts = None   # vertex indices of best canal cycle
        best_graph = None
        best_cycles = None       # all MCB cycles (for pruning)
        best_coords = None

        # path candidate tracked separately (fallback for bad segmentations)
        best_path_cost = float('inf')
        best_path_pts = None
        best_path_graph = None

        for ii in range(10):
            # dynamic thresholding; raise threshold until the spinal canal is enclosed
            dynamic_thresh = (ii + 0.001) * np.mean(pitch)
            expanded_mask = outside_dist < dynamic_thresh

            _t1 = _time.perf_counter()
            skeleton = image_morph.skeletonize(expanded_mask)
            graph = skeleton_to_graph(skeleton, affine)
            log.debug('  iter %d: skeleton_to_graph done in %.2fs (V=%d E=%d)', ii, _time.perf_counter() - _t1, len(graph.vs), len(graph.es))

            # minimum cycle basis: elementary cycles only (no composites)
            mcb_raw = graph.minimum_cycle_basis()
            coords = np.vstack(graph.vs.get_attribute_values('coord'))

            if mcb_raw:
                all_cycles = [_mcb_to_ordered_vertices(graph, eidxs) for eidxs in mcb_raw]
                cycles = [c for c in all_cycles if len(c) >= 20]
                if not cycles:
                    log.debug('  iter %d: %d MCB cycles but none >= 20 verts', ii, len(all_cycles))
                    continue
                canal_idx = max(range(len(cycles)), key=lambda i: len(cycles[i]))
                canal_cycle = cycles[canal_idx]
                cand_pts = coords[canal_cycle]
                cost = _assess_canal_loop(cand_pts, orig_pts, skel_thresh, ii)

                log.debug('  iter %d: %d MCB cycles, lengths=[%s], largest=%d cost=%.3f',
                          ii, len(cycles),
                          ', '.join(str(len(c)) for c in cycles),
                          len(canal_cycle), cost)

                if cost < best_cost:
                    best_cost, best_loop_pts, best_iter = cost, cand_pts, ii
                    best_loop_verts = canal_cycle
                    best_graph, best_cycles, best_coords = graph, cycles, coords

                if best_cost < 0.25:
                    break
            else:
                log.debug('  iter %d: no cycles found', ii)

            # --- Candidate 2: longest path (separate track) ---
            path_pts = get_longest_path_fast(graph)
            if len(path_pts) > 2:
                distances = np.linalg.norm(path_pts - np.roll(path_pts, -1, axis=0), axis=1)
                N = int(np.sum(distances) / np.mean(distances[1:-1]))
                path_pts = redistribute_evenly(path_pts, N)
                path_cost = _assess_canal_loop(path_pts, orig_pts, skel_thresh, ii)
                log.debug('  iter %d: path cost=%.3f (%d pts)', ii, path_cost, len(path_pts))
                if path_cost < best_path_cost:
                    best_path_cost = path_cost
                    best_path_pts = path_pts
                    best_path_graph = graph
        else:
            log.warning('no excellent canal loop after all iterations')

        # path can still beat cycle pool (bad-segmentation fallback)
        if best_path_cost < best_cost:
            best_cost = best_path_cost
            best_loop_pts = best_path_pts
            best_graph = best_path_graph
            best_cycles = None
            log.info('path candidate beat cycle pool: cost=%.3f', best_cost)

        # --- Prune support points ---
        if best_cycles:
            support_pts, n_pruned = _prune_non_canal_nodes(
                best_loop_verts, best_cycles, best_coords, best_graph)
            log.debug('pruned %d non-canal loop nodes, %d support pts remain', n_pruned, len(support_pts))
        else:
            support_pts = np.vstack(best_graph.vs.get_attribute_values('coord'))
            log.debug('no cycles to prune; using full skeleton (%d pts)', len(support_pts))

        log.info('canal_loop done in %.1fs: cost=%.3f at iter %d (%d pts, %d support)',
                 _time.perf_counter() - _t0, best_cost, best_iter,
                 len(best_loop_pts), len(support_pts))
        return best_loop_pts, support_pts


def fit_canal(loop_pts: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Iteratively move loop points inward to fit the spinal canal.

    Uses a ConvergenceScheduler per step-size phase to detect oscillation
    plateaus.  When movement stalls above tolerance, step is halved and a
    fresh scheduler starts.  Stops when movement drops below tolerance or
    the total iteration budget (200) is exhausted.
    """
    import time as _time
    _log = logging.getLogger(__name__)
    _log.debug('fit_canal: %d loop pts, mesh %d verts %d faces', len(loop_pts), len(verts), len(faces))

    edges = np.diff(np.roll(loop_pts, 1, axis=0), axis=0)
    scale = np.mean(np.linalg.norm(edges, axis=1))
    tol = 0.01 * scale

    inside = np.ones(len(loop_pts), dtype=bool)
    step = scale
    max_iters = 500
    total_iters = 0
    movement = np.inf
    _fc_t0 = _time.perf_counter()

    while total_iters < max_iters:
        scheduler = ConvergenceScheduler(thresh=0.05, window=5)

        while not scheduler.complete and total_iters < max_iters:
            init_pts = loop_pts.copy()
            # points in bone take a step towards the center
            vec = np.mean(loop_pts, axis=0) - loop_pts[inside]
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            loop_pts[inside] += vec * step

            # smooth loop; this will also contract the loop
            loop_pts = smooth_loop(loop_pts, 10)
            loop_pts = redistribute_evenly(loop_pts)

            # points outside the bone get projected back onto the surface
            w = np.round(igl.winding_number(verts, faces, loop_pts)).astype(np.int32)
            outside = w % 2 == 0
            inside = ~outside
            if np.any(outside):
                out_pts = loop_pts[outside]
                d2, _, projected = igl.point_mesh_squared_distance(out_pts, verts, faces)
                vec = out_pts - projected
                vec /= np.linalg.norm(vec, axis=1, keepdims=True)
                loop_pts[outside] = projected + vec * step

            displacement = np.linalg.norm(loop_pts - init_pts, axis=1)
            movement = np.percentile(displacement, 95)
            total_iters += 1
            scheduler.push(movement)

        if movement <= tol:
            _log.info('fit_canal: converged in %d iters (%.2fs), movement=%.4f',
                      total_iters, _time.perf_counter() - _fc_t0, movement)
            break

        if total_iters >= max_iters:
            break

        step *= 0.5
        _log.debug('fit_canal: stall at iter %d (movement=%.4f), step -> %.4f',
                   total_iters, movement, step)

        if step < tol:
            _log.info('fit_canal: step < tol at iter %d (%.2fs), movement=%.4f',
                      total_iters, _time.perf_counter() - _fc_t0, movement)
            break
    else:
        _log.warning('fit_canal: did NOT converge after %d iters (movement=%.4f threshold=%.4f)',
                     total_iters, movement, tol)

    plane = fit_plane_to_points(loop_pts)
    midpoint = np.mean(loop_pts, axis=0)
    return plane, midpoint, loop_pts
