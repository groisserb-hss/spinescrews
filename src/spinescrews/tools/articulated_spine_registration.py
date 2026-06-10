import os
import numpy as np
from numpy import pi
import logging
import igl
import maxflow
from bg3dtools.transforms_unified import transform_points_forward, transform_points_inverse, make_aff, rigid_reg, extract_params, aff_to_rel_params, inverse_rigid
from spinescrews.tools.articulated_models.spine import Spine
import nibabel as nib
from scipy.spatial import KDTree
from scipy.optimize import approx_fprime, least_squares
from scipy.signal import savgol_filter
import scipy.ndimage as ndi

from bg3dtools.pointclouds.quantize import convert_to_points, sparse_quantize
from bg3dtools.mesh.utils import per_vertex_normals
from bg3dtools.mesh.mesh_io import write_colored_plyfile
from bg3dtools.render.colors import default_colors
from spinescrews.tools.screw_models import Screw
from spinescrews.tools.nifti_utils import compute_metal_threshold
from spinescrews.tools.vertebrae import Vertebra
from spinescrews.tools import possible_levels
from spinescrews.tools.paths import timed
from joblib import Parallel, delayed


log = logging.getLogger(__name__)

_CMAES_DEBUG_DIR = '/tmp/cmaes_debug'


def _save_cmaes_inputs(spine, postop_pts, icp_affs, ratios, kdtrees,
                       postop_img, artifact_mask, metal_thresh,
                       level_names, iso_res, initial_radius):
    """Dump refit inputs to disk for offline debugging."""
    import json
    d = _CMAES_DEBUG_DIR
    os.makedirs(d, exist_ok=True)

    # Spine: verts, landmarks, default_aff (faces not used by _cmaes_refit)
    np.savez(os.path.join(d, 'spine.npz'),
             **{f'verts_{i}': v for i, v in enumerate(spine.verts)},
             **{f'landmarks_{i}': lm for i, lm in enumerate(spine.landmarks)},
             default_aff=spine.default_aff)

    np.save(os.path.join(d, 'postop_pts.npy'), postop_pts)
    np.save(os.path.join(d, 'icp_affs.npy'), icp_affs)
    np.save(os.path.join(d, 'ratios.npy'), ratios)

    if postop_img is not None:
        nib.save(postop_img, os.path.join(d, 'postop.nii.gz'))
    if artifact_mask is not None:
        np.save(os.path.join(d, 'artifact_mask.npy'), artifact_mask)

    with open(os.path.join(d, 'params.json'), 'w') as f:
        json.dump({'metal_thresh': metal_thresh, 'level_names': level_names,
                   'iso_res': iso_res, 'initial_radius': initial_radius,
                   'nJ': spine.nJ}, f)

    log.info('Saved CMA-ES debug inputs to %s', d)


def align_spine_to_CT(preop_verts: dict[str, Vertebra], postop_img: nib.Nifti1Image,
                      screws: list[Screw], initial_affs: dict,
                      iso_res=1.5, initial_radius=12., ratio_thresh=0.75,
                      output_dir=None) -> tuple[dict, dict, np.ndarray]:
    """
    Aligns a spine model to a post-operative CT scan using an articulated variant of ICP.

    Returns
    -------
    icp_affs : dict
        {level_name: 4x4 ndarray} ICP-refined transforms.
    metrics : dict
        Structured quality metrics for summary.json.
    artifact_mask : np.ndarray
        Boolean mask (same shape as postop volume). True = artifact voxel.
    """
    nJ = len(preop_verts)
    timings = {}

    level_names = list(preop_verts.keys())
    level_names = [name for name in possible_levels if name in level_names] # sort from bottom to top
    screws = [s for s in screws if s.type != 'skip']
    initial_affs = np.stack([initial_affs[name].copy() for name in level_names], axis=0)

    ## Step 1: Prepare pre-op data
    with timed('extract_preop_pts', timings):
        v_list, f_list, pt_list, thresh_list = extract_preop_pts([preop_verts[name] for name in level_names], iso_res=iso_res)
    num_preop = np.sum([len(p) for p in pt_list])
    log.info('Extracted %d preop cortical points (average %dmm^3 per vertebra)' %
             (num_preop, int(num_preop * (iso_res**3) / len(level_names)) ))

    # Construct spine model
    preop_affs = np.stack([preop_verts[name].affine for name in level_names], axis=0)
    spine = Spine(v_list, default_aff=preop_affs, faces=f_list, landmarks=pt_list)

    ## Step 2: Prepare post-op data
    thresh_dict = {name: thresh for name, thresh in zip(level_names, thresh_list)}
    thresh_list =[thresh_dict[s.level] for s in screws]
    init_bonepts = np.row_stack(spine.build_landmarks(initial_affs))
    metal_thresh = compute_metal_threshold(postop_img.get_fdata())
    with timed('build_artifact_mask', timings):
        artifact_mask = _build_artifact_mask_fast(postop_img, screws, metal_threshold=metal_thresh)
    with timed('extract_postop_pts', timings):
        postop_pts, postop_hu = extract_postop_pts(postop_img, screws, thresh_list, init_bonepts,
                                                    iso_res=iso_res, initial_radius=initial_radius,
                                                    artifact_mask=artifact_mask,
                                                    metal_thresh=metal_thresh)

    ## Step 3: Run articulated ICP on full spine simultaneously
    # HU-threshold downsampling: keep the densest cortical points for ICP
    ICP_TARGET = 80_000
    if len(postop_pts) > ICP_TARGET:
        cutoff = np.percentile(postop_hu, 100 * (1 - ICP_TARGET / len(postop_pts)))
        icp_mask = postop_hu > cutoff
        icp_pts = postop_pts[icp_mask]
        log.info('HU-threshold downsampled %d -> %d postop pts for ICP (cutoff %.1f)',
                 len(postop_pts), len(icp_pts), cutoff)
    else:
        icp_pts = postop_pts
    with timed('articulated_registration', timings):
        icp_affs, kdtrees, initial_loss, final_loss = articulated_registration(spine, icp_pts, initial_affs,
                                                                                initial_radius=initial_radius)

    # calculate goodness of fit for each bone
    ratios = np.zeros(spine.nJ)
    for jj in range(spine.nJ):
        postop_inv = transform_points_inverse(icp_affs[jj], postop_pts)
        d = kdtrees[jj].query(postop_inv)[0]
        ratios[jj] = np.sum(d < iso_res) / len(spine.landmarks[jj])
    log.info('Computed ratios: %s' % ', '.join(['%s: %.2f' % (name, r) for name, r in zip(level_names, ratios)]))

    ## Save inputs for offline debugging (enable with env var DEBUG_CMAES=1)
    if os.environ.get('DEBUG_CMAES'):
        _save_cmaes_inputs(spine, postop_pts, icp_affs, ratios, kdtrees,
                           postop_img, artifact_mask, metal_thresh,
                           level_names, iso_res, initial_radius)

    ## Step 4: Per-level refit via particle belief propagation
    screw_levels = set(s.level for s in screws)
    with timed('perlevel_refit', timings):
        icp_affs, ratios = _particle_bp_refit(spine, postop_pts, icp_affs, ratios, kdtrees,
                                               level_names=level_names, iso_res=iso_res,
                                               screw_levels=screw_levels)
    per_level_ratios = {name: float(r) for name, r in zip(level_names, ratios)}

    metrics = {
        'n_preop_cortical_pts': int(num_preop),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'per_level_ratios': per_level_ratios,
        'perlevel_refit': True,
        'timings': timings,
    }

    if output_dir is not None:
        empty_faces = np.zeros((0, 3), dtype=np.int32)
        write_colored_plyfile(os.path.join(output_dir, 'icp_postop.ply'),
                              postop_pts, empty_faces)
        model_pts, model_rgb = [], []
        for jj in range(nJ):
            posed = transform_points_forward(icp_affs[jj], spine.landmarks[jj])
            model_pts.append(posed)
            c = (np.array(default_colors[jj % len(default_colors)]) * 255).astype(np.uint8)
            model_rgb.append(np.tile(c, (len(posed), 1)))
        write_colored_plyfile(os.path.join(output_dir, 'icp_model.ply'),
                              np.row_stack(model_pts), empty_faces,
                              v_rgb=np.row_stack(model_rgb))
        log.info('Saved ICP point clouds to %s', output_dir)

    return {name: aff for name, aff in zip(level_names, icp_affs)}, metrics, artifact_mask


def extract_preop_pts(vertebrae: list[Vertebra], iso_res=1.5) -> (list[np.ndarray], list[float]):
    """
    Downsamples meshes
    :param vertebrae:
    :return:
    """
    v_list, f_list = [], []
    thresh_list, pt_list = [], []
    for vertebra in vertebrae:
        v, f = vertebra.verts, vertebra.faces
        _, v, f, _, _ = igl.qslim(v, f, 5000)
        n = per_vertex_normals(v, f)
        v = v + n  # inflate by 1mm
        v_list.append(np.ascontiguousarray(v))
        f_list.append(np.ascontiguousarray(f))

        data = vertebra.img_normalized.get_fdata()
        noise = np.std(ndi.laplace(data))
        sigma = ( 4.5 * noise / 10000) + .1
        pitch = np.diag(vertebra.img_normalized.affine)[:3]
        smoothed = ndi.gaussian_filter(data, sigma / pitch)
        seg_mask = vertebra.seg_normalized.get_fdata().astype(bool)

        p = max(min(.75, 1 - 6000 / (np.sum(seg_mask) * (0.5 ** 3))), 0)  # bad segmentations might have low volume
        t = np.percentile(smoothed[seg_mask], 100*p)
        intensity_mask = smoothed > t
        pts = convert_to_points(intensity_mask & seg_mask, vertebra.img_normalized.affine)
        pts = sparse_quantize(pts / iso_res).astype(float) * iso_res
        pt_list.append(pts)
        thresh_list.append(t)
        log.debug('Vertebra %s: %.2fmm^3 at threshold %f; sigma=%.2f for final noise=%.2f' % (
            vertebra.name, len(pts) * (iso_res ** 3), t, sigma, np.std(ndi.laplace(smoothed))))

    return v_list, f_list, pt_list, thresh_list


def extract_postop_pts(postop_img: nib.Nifti1Image, screws: list[Screw],
                       thresh_list: list[float], posed_pts,
                       iso_res=1.5, initial_radius=12.,
                       artifact_mask=None, metal_thresh=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts point cloud from a post-operative CT scan.

    Returns (pts, H) where H is the scaled HU value for each point.
    """

    if metal_thresh is None:
        metal_thresh = compute_metal_threshold(postop_img.get_fdata())

    # Global bone threshold (CMA-ES re-extracts with per-vertebra adaptive scoring)
    variable_thresh = np.median(thresh_list)

    # extract cortical points as 3D point cloud
    data = postop_img.get_fdata()
    noise = np.std(ndi.laplace(data))
    sigma = ( 4.5 * noise / 10000) + .1
    pitch = np.diag(postop_img.affine)[:3]
    with timed('gaussian_filter'):
        smoothed = ndi.gaussian_filter(data, sigma / pitch)
    log.debug('Smoothed postop volume with sigma=%.2f for final noise=%.2f' % (
        sigma, np.std(ndi.laplace(smoothed))))

    scaled = smoothed / variable_thresh
    if artifact_mask is not None:
        cortical_mask = (scaled > 0.75) & (~artifact_mask)
    else:
        cortical_mask = (scaled > 0.75) & (smoothed < metal_thresh)
    pts = convert_to_points(cortical_mask, postop_img.affine)
    H = scaled[cortical_mask]
    pts, H = sparse_quantize(pts / iso_res, H)
    pts = pts.astype(float) * iso_res
    log.info('Extracted %d cortical points from post-op CT scan at %.2fmm isotropic' % (len(pts), iso_res))

    # keep points near initial positions
    preop_init = sparse_quantize(posed_pts / 5).astype(float) * 5
    kdtree = KDTree(preop_init)
    d, _ = kdtree.query(pts)
    pts, H = pts[d < initial_radius], H[d < initial_radius]
    log.debug('Filtered down to %d points near initial positions' % len(pts))

    # remove points inside screws
    if artifact_mask is None:
        # Legacy screw exclusion via winding number (only when no artifact mask)
        verts, faces = np.zeros([0, 3]), np.zeros([0, 3], dtype=int)
        for screw in screws:
            v, f = screw.build_mesh(planned=False)
            faces = np.row_stack([faces, f + len(verts)])
            verts = np.row_stack([verts, v])
        verts, faces = np.ascontiguousarray(verts), np.ascontiguousarray(faces)
        with timed('winding_number'):
            w = igl.winding_number(verts, faces, pts)
        screw_mask = (np.round(w).astype(int) % 2).astype(bool)
        pts, H = pts[~screw_mask], H[~screw_mask]
        log.debug('Filtered down to %d cortical points outside of screws' % len(pts))

    # threshold to same number of points as preop cortex; note this means there will be fewer points in the
    # vertebrae since this also includes ribs, skull, etc
    if len(pts) > len(posed_pts):
        p = 1 - len(posed_pts) / len(pts)
        thresh = np.percentile(H, 100 * p)
        keep = H > thresh
        pts, H = pts[keep], H[keep]
        log.debug('Filtered down to %d cortical points above %.2f percentile' % (len(pts), 100 * p))
    else:
        log.warning('insufficient points to filter; keeping all %d points' % len(pts))
    return pts, H


def _graphcut_cluster(data, cluster_mask, affine, metal_threshold,
                      margin_mm=15.0, sigma_hu=150.0, alpha=10.0,
                      gc_resolution_mm=1.5):
    """Graph-cut artifact segmentation for a single metal cluster.

    Parameters
    ----------
    data : np.ndarray
        Full postop CT volume (HU).
    cluster_mask : np.ndarray
        Boolean mask for this cluster (same shape as data).
    affine : np.ndarray
        (4, 4) voxel-to-world affine.
    metal_threshold : float
        Auto-detected metal HU threshold (used for sigmoid data term).
    margin_mm : float
        Margin around cluster bbox for subvolume.
    sigma_hu : float
        Edge weight sensitivity.
    alpha : float
        Unary-to-binary weight ratio.
    gc_resolution_mm : float
        Target voxel spacing for graph-cut (coarser = faster). Set to 0 to disable.

    Returns
    -------
    artifact_mask : np.ndarray
        Boolean subvolume mask. True = artifact.
    slices : tuple of slices
        Index back into full volume.
    """
    INF = 1e9
    pitch = np.abs(np.diag(affine[:3, :3]))
    margin_vox = (margin_mm / pitch).astype(int)

    # Bounding box of cluster, expanded by margin
    coords = np.argwhere(cluster_mask)
    lo = np.maximum(coords.min(0) - margin_vox, 0)
    hi = np.minimum(coords.max(0) + margin_vox, np.array(data.shape) - 1)
    slices = tuple(slice(l, h + 1) for l, h in zip(lo, hi))

    subvol_hires = data[slices].astype(np.float64)
    sub_cluster = cluster_mask[slices]
    orig_shape = subvol_hires.shape

    # Downsample to target resolution (graph-cut complexity scales with voxel count)
    ds_factor = np.maximum(1.0, gc_resolution_mm / pitch) if gc_resolution_mm > 0 else np.ones(3)
    downsampled = np.min(ds_factor) > 1.3 and min(orig_shape) > 4
    if downsampled:
        subvol = ndi.zoom(subvol_hires, 1.0 / ds_factor, order=1)
        sub_cluster_gc = ndi.zoom(sub_cluster.astype(np.float32), 1.0 / ds_factor, order=0) > 0.5
    else:
        subvol = subvol_hires
        sub_cluster_gc = sub_cluster

    # Build graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(subvol.shape)

    # Edge weights (n-links) — 6-connected grid
    for axis in range(3):
        diff = np.diff(subvol, axis=axis)
        caps = np.exp(-diff**2 / (2 * sigma_hu**2))
        # Forward direction: node → node+1 along axis
        struct_fwd = np.zeros((3, 3, 3), dtype=int)
        idx_fwd = [1, 1, 1]
        idx_fwd[axis] = 2
        struct_fwd[tuple(idx_fwd)] = 1
        w_fwd = np.pad(caps, [(0, 1) if a == axis else (0, 0) for a in range(3)])
        g.add_grid_edges(nodeids, w_fwd, struct_fwd, symmetric=0)
        # Backward direction: node → node-1 along axis
        struct_bwd = np.zeros((3, 3, 3), dtype=int)
        idx_bwd = [1, 1, 1]
        idx_bwd[axis] = 0
        struct_bwd[tuple(idx_bwd)] = 1
        w_bwd = np.pad(caps, [(1, 0) if a == axis else (0, 0) for a in range(3)])
        g.add_grid_edges(nodeids, w_bwd, struct_bwd, symmetric=0)

    # Terminal edges (t-links)
    # Soft data term: sigmoid centered between bone (800 HU) and metal
    sigmoid_center = (metal_threshold + 800) / 2
    k = np.log(99) / 250  # 1%→99% over ~500 HU (+/- 250 around center)
    sigmoid = 1.0 / (1.0 + np.exp(-k * (subvol - sigmoid_center)))
    source_caps = alpha * sigmoid
    sink_caps = alpha * (1.0 - sigmoid)

    # Hard source seeds: metal cluster voxels
    source_caps[sub_cluster_gc] = INF

    # Hard sink seeds: boundary shell + low-HU voxels
    boundary = np.zeros(subvol.shape, dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    low_hu = subvol < 200
    sink_seeds = boundary | low_hu
    sink_caps[sink_seeds] = INF
    # Don't let sink override source at metal voxels
    sink_caps[sub_cluster_gc] = 0

    g.add_grid_tedges(nodeids, source_caps, sink_caps)

    # Solve
    g.maxflow()
    segments = g.get_grid_segments(nodeids)  # True = sink (background)
    raw_artifact = ~segments

    # Keep only the connected component(s) touching the original metal cluster
    struct26 = ndi.generate_binary_structure(3, 3)
    cc_labels, n_cc = ndi.label(raw_artifact, structure=struct26)
    keep = np.zeros(n_cc + 1, dtype=bool)  # index 0 = background
    keep[cc_labels[sub_cluster_gc]] = True
    artifact_mask = keep[cc_labels]

    # Refine boundary at full resolution using high-res HU
    if downsampled:
        coarse_up = ndi.zoom(artifact_mask.astype(np.float32),
                            [o / s for o, s in zip(orig_shape, artifact_mask.shape)],
                            order=0) > 0.5
        # Core: erode to get confident artifact interior
        core = ndi.binary_erosion(coarse_up)
        # Search band: wider region around coarse boundary
        candidate = ndi.binary_dilation(coarse_up, iterations=2)
        band = candidate & ~core
        # In band, classify by high-res HU (above sigmoid center = artifact)
        artifact_mask = core | (band & (subvol_hires > sigmoid_center))
        artifact_mask |= sub_cluster  # always include metal

    # Safety dilation by 1 voxel (at full resolution)
    artifact_mask = ndi.binary_dilation(artifact_mask)

    return artifact_mask, slices


def _build_artifact_mask(postop_img, screws, screw_proximity_mm=50.0, metal_threshold=None):
    """Build artifact mask covering metal + streak artifacts via per-cluster graph-cut.

    Pipeline: threshold -> connected components -> filter near screws -> graph-cut per cluster -> union.
    Returns boolean mask (same shape as postop volume). True = artifact.
    """
    data = postop_img.get_fdata()
    affine = postop_img.affine
    threshold = metal_threshold if metal_threshold is not None else compute_metal_threshold(data)
    log.debug('Artifact mask: metal threshold=%d HU', threshold)

    metal_mask = data >= threshold
    struct = ndi.generate_binary_structure(3, 3)  # 26-connectivity
    labels, n_labels = ndi.label(metal_mask, structure=struct)
    log.debug('Artifact mask: %d connected components above threshold', n_labels)

    # Build KDTree of all screw mesh vertices (detected positions)
    screw_verts = []
    for screw in screws:
        if screw.type == 'skip':
            continue
        v, _f = screw.build_mesh(planned=False)
        screw_verts.append(v)
    if not screw_verts:
        log.warning('No non-skip screws; returning empty artifact mask')
        return np.zeros(data.shape, dtype=bool)
    screw_pts = np.row_stack(screw_verts)
    screw_tree = KDTree(screw_pts)

    # Vectorized proximity filtering: find which labels have any voxel near screws
    artifact_mask = np.zeros(data.shape, dtype=bool)
    metal_vox_idx = np.argwhere(labels > 0)
    if len(metal_vox_idx) == 0:
        log.info('Artifact mask: 0 qualifying clusters')
        return artifact_mask
    metal_labels = labels[tuple(metal_vox_idx.T)]
    metal_world = (affine[:3, :3] @ metal_vox_idx.astype(float).T + affine[:3, 3:]).T
    near = screw_tree.query(metal_world)[0] < screw_proximity_mm
    qualifying_labels = np.unique(metal_labels[near])
    label_counts = np.bincount(labels.ravel())

    # Small clusters: dilate only (no graph-cut overhead)
    SMALL_CLUSTER = 20
    small_labels = [int(lid) for lid in qualifying_labels if label_counts[lid] < SMALL_CLUSTER]
    large_labels = [int(lid) for lid in qualifying_labels if label_counts[lid] >= SMALL_CLUSTER]
    for lid in small_labels:
        artifact_mask |= ndi.binary_dilation(labels == lid, iterations=3)

    # Graph-cut for large clusters (threaded for parallelism where maxflow releases GIL)
    if large_labels:
        n_jobs = min(len(large_labels), max(1, os.cpu_count() - 2))
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_graphcut_cluster)(data, labels == lid, affine, threshold)
            for lid in large_labels)
        for gc_mask, gc_slices in results:
            artifact_mask[gc_slices] |= gc_mask

    n_qualifying = len(qualifying_labels)
    voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))
    total_vol = artifact_mask.sum() * voxel_vol
    log.info('Artifact mask: %d qualifying clusters (%d graph-cut, %d dilated), %.1f cm^3 total artifact volume',
             n_qualifying, len(large_labels), len(small_labels), total_vol / 1000)
    return artifact_mask


def _build_artifact_mask_fast(postop_img, screws, screw_proximity_mm=20.0,
                              metal_threshold=None):
    """Build artifact mask: threshold -> open -> proximity filter -> dilate -> streak threshold.

    Faster replacement for _build_artifact_mask (graph-cut based).
    """
    data = postop_img.get_fdata()
    affine = postop_img.affine
    pitch = np.abs(np.diag(affine[:3, :3]))
    metal_thresh = metal_threshold if metal_threshold is not None else compute_metal_threshold(data)
    streak_thresh = (metal_thresh + 800) / 2  # midpoint: cortical bone ~ metal

    # 1. Conservative metal mask
    metal_mask = data >= metal_thresh

    # 2. Morphological open (remove noisy isolated voxels)
    struct = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    metal_mask = ndi.binary_opening(metal_mask, structure=struct)

    # 3. Proximity filter: keep only components within screw_proximity_mm of a screw
    screw_verts = []
    for screw in screws:
        if screw.type == 'skip':
            continue
        v, _f = screw.build_mesh(planned=False)
        screw_verts.append(v)
    if not screw_verts:
        log.warning('No non-skip screws; returning empty artifact mask')
        return np.zeros(data.shape, dtype=bool)
    screw_tree = KDTree(np.row_stack(screw_verts))

    cc_struct = ndi.generate_binary_structure(3, 3)  # 26-connectivity
    labels, n_labels = ndi.label(metal_mask, structure=cc_struct)
    metal_vox = np.argwhere(labels > 0)
    if len(metal_vox) == 0:
        return np.zeros(data.shape, dtype=bool)
    metal_world = (affine[:3, :3] @ metal_vox.astype(float).T + affine[:3, 3:]).T
    near = screw_tree.query(metal_world)[0] < screw_proximity_mm
    keep_labels = set(np.unique(labels[tuple(metal_vox[near].T)]))
    metal_mask = np.isin(labels, list(keep_labels))
    log.debug('Artifact mask: %d/%d components near screws', len(keep_labels), n_labels)

    # 4. Dilate by ~5mm
    dilate_mm = 5.0
    dilate_iters = max(1, int(round(dilate_mm / pitch.min())))
    dilated = ndi.binary_dilation(metal_mask, iterations=dilate_iters)

    # 5. Within dilated zone, catch streak artifacts at lower threshold
    streak_zone = dilated & ~metal_mask
    artifact_mask = metal_mask | (streak_zone & (data >= streak_thresh))

    voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))
    log.info('Artifact mask (fast): metal_thresh=%d, streak_thresh=%d, '
             'dilate=%d iters (%.1fmm), %.1f cm^3',
             metal_thresh, int(streak_thresh), dilate_iters, dilate_mm,
             artifact_mask.sum() * voxel_vol / 1000)
    return artifact_mask


def _level_cost(p6, j, current_affs, kdtrees, assignments, postop_pts,
                postop_tree_j, landmarks_j, model_rel_twist, model_rel_trans,
                inter_vertebral_dist, nJ, iso_res=1.5, prior_weight=1.0):
    """Scalar cost for level j with all other levels fixed.

    Soft-ratio kernel exp(-d²/2σ²) with σ=iso_res directly approximates the
    hard ratio metric (count of points within iso_res). Both d2m and m2d are
    restricted to Voronoi-assigned points to prevent cross-level drift.
    """
    s2 = 2 * iso_res ** 2
    aff_j = make_aff(p6[:3][None], p6[3:][None])[0]

    # d2m: Voronoi-assigned postop points → inverse transform → soft ratio
    mask = assignments == j
    if mask.any():
        pts_rest = transform_points_inverse(aff_j, postop_pts[mask])
        d_d2m = kdtrees[j].query(pts_rest)[0]
        d2m = 1.0 - np.mean(np.exp(-d_d2m**2 / s2))
    else:
        d2m = 1.0

    # m2d: landmarks → forward transform → Voronoi-restricted postop tree
    d_m2d = postop_tree_j.query(transform_points_forward(aff_j, landmarks_j))[0]
    m2d = 1.0 - np.mean(np.exp(-d_m2d**2 / s2))

    # Prior: relative transform to parent (j > 0) and child (j < nJ-1)
    prior = 0.0
    pw = prior_weight

    if j > 0:
        parent_aff = current_affs[j - 1]
        rel_aff = inverse_rigid(parent_aff) @ aff_j
        rel_tw, rel_tr = extract_params(rel_aff[None])
        tw_diff = ((pi + model_rel_twist[j] - rel_tw[0]) % (2 * pi)) - pi
        lam = (1 / 5) * (15 / inter_vertebral_dist[j])
        tr_diff = lam * (np.linalg.norm(model_rel_trans[j]) - np.linalg.norm(rel_tr[0]))
        prior += pw * (np.sum(tw_diff**2) + tr_diff**2)

    if j < nJ - 1:
        child_aff = current_affs[j + 1]
        rel_aff = inverse_rigid(aff_j) @ child_aff
        rel_tw, rel_tr = extract_params(rel_aff[None])
        tw_diff = ((pi + model_rel_twist[j + 1] - rel_tw[0]) % (2 * pi)) - pi
        lam = (1 / 5) * (15 / inter_vertebral_dist[j + 1])
        tr_diff = lam * (np.linalg.norm(model_rel_trans[j + 1]) - np.linalg.norm(rel_tr[0]))
        prior += pw * (np.sum(tw_diff**2) + tr_diff**2)

    return d2m + m2d + prior


def _spine_cost(params: np.ndarray, initial_affs: np.ndarray, kdtrees: list[KDTree],
                masks: np.ndarray, postop_pts: np.ndarray,
                gm_breakpoint=2.0, postop_tree=None, query_landmarks=None, prior_weight=1.):
    """Residual vector for articulated ICP: Geman-McClure data loss + inter-vertebral regularization.

    Optional chamfer (model-to-data) term when postop_tree and landmarks are provided.
    """
    nJ = len(kdtrees)
    nP = len(postop_pts)
    c2 = gm_breakpoint**2
    trunk = (np.arange(nJ) - 1).tolist()

    model_rel_twist, model_rel_trans = aff_to_rel_params(trunk, initial_affs)
    inter_vertebral_dist = np.linalg.norm(model_rel_trans, axis=1)
    _lambda = (1/5) * (15 / inter_vertebral_dist[1:])  # 1/20 to balance radians vs mm, normalize to 15mm to upweight prior on close pairs of vertebrae
    # parse parameters
    params = params.reshape([nJ, 6])
    abs_twist, abs_trans = params[:, :3], params[:, 3:]
    test_affs = make_aff(abs_twist, abs_trans)
    rel_twist, rel_trans = aff_to_rel_params(trunk, test_affs)

    # data-to-model loss
    d_upper = 3 * gm_breakpoint
    D = 100 * np.ones([nJ, nP])
    for jj in range(nJ):
        mask = masks[jj]
        pts_at_rest = transform_points_inverse(test_affs[jj], postop_pts[mask])
        d = kdtrees[jj].query(pts_at_rest, distance_upper_bound=d_upper)[0]
        d[~np.isfinite(d)] = d_upper  # KDTree returns inf beyond bound
        D[jj][mask] = d

    d2 = np.min(D, axis=0)**2
    d2m_loss = (d2 / (c2 + d2)) / np.sqrt(nP)

    # model-to-data (chamfer) loss
    if postop_tree is not None and query_landmarks is not None:
        m2d_dists = []
        for jj in range(nJ):
            d = postop_tree.query(transform_points_forward(test_affs[jj], query_landmarks[jj]),
                                  distance_upper_bound=d_upper)[0]
            d[~np.isfinite(d)] = d_upper
            m2d_dists.append(d)
        m2d_parts = [d**2 / (c2 + d**2) for d in m2d_dists]
        nL = sum(len(d) for d in m2d_dists)
        m2d_loss = np.concatenate(m2d_parts) / np.sqrt(nL)
    else:
        m2d_loss = np.array([])

    # regularization loss
    rel_twist_diff = ((pi + model_rel_twist[1:] - rel_twist[1:]) % (2 * pi)) - pi
    rel_twist_diff *= prior_weight / np.sqrt(6 * nJ)
    rel_trans_diff = _lambda * (np.linalg.norm(model_rel_trans[1:], axis=1) -
                                np.linalg.norm(rel_trans[1:], axis=1))
    rel_trans_diff *= prior_weight / np.sqrt(6 * nJ)

    # total loss
    loss = np.concatenate([d2m_loss, m2d_loss, rel_twist_diff.flatten(), rel_trans_diff.flatten()])
    return loss


def articulated_registration(spine: Spine, postop_pts: np.ndarray,
                             initial_affs: np.ndarray,
                             initial_radius=12.) -> (np.ndarray, np.ndarray):
    """Multi-level articulated ICP: jointly optimize all vertebral transforms against postop points."""
    nJ = spine.nJ
    nP = len(postop_pts)

    # compute masks to speed up computations inside solver
    kdtrees, masks = [], np.zeros([nJ, nP], dtype=bool)
    for jj in range(spine.nJ):
        preop_rest = spine.landmarks[jj]
        bone_tree = KDTree(preop_rest)  # kdtree of preop cortical points at origin
        preop_inv = transform_points_inverse(initial_affs[jj], postop_pts)
        d = bone_tree.query(preop_inv)[0]

        kdtrees.append(bone_tree)
        masks[jj] = d < initial_radius

    cost_fun = lambda params : _spine_cost(params, spine.default_aff, kdtrees, masks, postop_pts)
    init_twist, init_trans = extract_params(initial_affs)
    init_params = np.column_stack((init_twist, init_trans)).flatten()
    initial_loss = float(np.sum(cost_fun(init_params)**2))

    J_sparse = np.column_stack([np.tile(m.reshape([-1, 1]), [1, 6]) for m in masks])
    # Regularization sparsity: each residual depends on only 2 adjacent joints
    J_reg = np.zeros((4 * (nJ - 1), 6 * nJ))
    for k in range(nJ - 1):
        J_reg[3 * k:3 * (k + 1), 6 * k:6 * (k + 2)] = 1       # twist diff
        J_reg[3 * (nJ - 1) + k, 6 * k:6 * (k + 2)] = 1         # trans diff
    J_sparse = np.row_stack([J_sparse, J_reg])
    with timed('least_squares (articulated ICP)'):
        result = least_squares(cost_fun, init_params, method='trf', verbose=0, ftol=0.01, jac_sparsity=J_sparse)
    opt_loss = float(np.sum(cost_fun(result.x)**2))
    opt_params = result.x.reshape([nJ, 6])
    opt_twist, opt_trans = opt_params[:, :3], opt_params[:, 3:]
    log.info('Aligned spine to postop pts: loss %.3f -> %.3f' % (initial_loss, opt_loss))

    opt_affs = make_aff(opt_twist, opt_trans)

    return opt_affs, kdtrees, initial_loss, opt_loss


def _get_paired_levels(ratios, stable, need_correction):
    """Find (anchor, floater) level pair for corrective ICP re-alignment."""
    best = np.argmax(ratios)
    if not stable[best]:
        log.warning('No stable level (best ratio=%.2f); skipping corrective registration', ratios[best])
        return None
    # find closest level that needs correction
    offset = np.where(need_correction)[0] - best
    ii = np.argmin(np.abs(offset))
    floater = offset[ii] + best
    step = -np.sign(offset[ii])
    while not stable[floater + step]:
        floater += step
    return floater + step, floater


def _wiggle_to_fit(D, kdtree, current_affs, anchor, floater, stable,
                   landmarks, model_rel_twist, model_rel_trans,
                   postop_pts, iso_res, initial_radius):
    """Refit a single floater level's transform relative to its stable anchor."""
    num_preop = len(landmarks[floater])

    if np.abs(anchor - floater) != 1:
        raise RuntimeError('anchor and floater must be adjacent levels, got %d and %d' % (anchor, floater))
    if floater > anchor:
        offset = make_aff(model_rel_twist[floater], model_rel_trans[floater])
    else:
        offset = make_aff(model_rel_twist[anchor], model_rel_trans[anchor])
        offset = np.linalg.inv(offset)

    # T1 is already applied; spine model points are at the origin
    # T2 is a random translation/rotation about the origin
    # T3 is the relative transformation from the anchor level to the floating level (put anchor at origin)
    T3 = offset
    radius = np.linalg.norm(extract_params(offset)[1])
    # T4 is a rotation about the anchor (positioned at the origin)
    # T5 transforms from the origin to the anchor position
    T5 = current_affs[anchor].copy()

    # subsample points to speed up
    W = np.exp(-D / iso_res).clip(0, 1)  # closeness score
    E = np.maximum(0.000001, np.exp(W) - 1)
    P = (D < 10) * (E / np.sum(E, axis=0, keepdims=True))  # softmax to convert to assignment probabilities

    # only consider points that are not already well aligned
    unassigned = np.all(P[stable] < 0.80, axis=0)
    close = D[floater] < (initial_radius * (2/3))
    pts = postop_pts[unassigned & close]

    ii, stalls = 0, 0
    best_tform, best_ratio, ratio_hist = np.eye(4), 0, np.zeros(10000)
    T4_twist = np.zeros(3)
    while stalls < 1000:
        t = np.exp(-stalls / 500)  # temperature for search
        # perform random translation/rotation about origin
        test_twist1 = ((6 * np.random.rand(3) - 3) * t / 5 if ii > 0 else np.zeros(3))
        test_trans1 = ((6 * np.random.rand(3) - 3) * radius * t / 5 if ii > 0 else np.zeros(3))
        T2 = make_aff(test_twist1, test_trans1)
        # apply random rotation
        test_twist2 = ((6 * np.random.rand(3) - 3) * t / 10 if ii > 0 else np.zeros(3)) + T4_twist
        T4 = make_aff(test_twist2, None)

        # Transform from model (rest) position to postop test position
        test_tform = T5 @ T4 @ T3 @ T2

        pts_at_rest = transform_points_inverse(test_tform, pts)
        d = kdtree.query(pts_at_rest)[0]
        test_ratio = np.sum(np.exp(-d**2 / (2 * iso_res**2))) / num_preop
        if test_ratio > best_ratio:
            best_ratio = test_ratio
            best_tform = test_tform.copy()
            T4_twist = test_twist2
            log.debug('New best ratio found: iteration %d  ratio %.2f  T4 twist %s' % (ii, test_ratio, T4_twist))
            stalls = 0
        ii, stalls = ii + 1, stalls + 1

    return best_tform, best_ratio


def _adaptive_postop_pts(postop_img, spine, icp_affs,
                         iso_res=1.5, initial_radius=12.,
                         artifact_mask=None, metal_thresh=None,
                         level_names=None):
    """Re-extract postop cortical points with per-vertebra adaptive scoring.

    Bone-like voxels are assigned to their nearest vertebra (Voronoi), scored
    by HU normalized by that vertebra's median density, and the top N are kept
    (N = total preop cortical points).
    """
    data = postop_img.get_fdata()
    if metal_thresh is None:
        metal_thresh = compute_metal_threshold(data)
    affine = postop_img.affine
    nJ = spine.nJ

    # Pose all preop landmarks into postop space, track ownership
    all_posed, posed_owner = [], []
    for jj in range(nJ):
        posed = transform_points_forward(icp_affs[jj], spine.landmarks[jj])
        all_posed.append(posed)
        posed_owner.append(np.full(len(posed), jj, dtype=int))
    all_posed = np.row_stack(all_posed)
    posed_owner = np.concatenate(posed_owner)

    # Global bounding box around all posed landmarks + margin
    lo_world = all_posed.min(0) - initial_radius
    hi_world = all_posed.max(0) + initial_radius
    corners_vox = transform_points_inverse(affine, np.array([lo_world, hi_world]))
    lo_vox = np.maximum(np.floor(np.minimum(corners_vox[0], corners_vox[1])).astype(int), 0)
    hi_vox = np.minimum(np.ceil(np.maximum(corners_vox[0], corners_vox[1])).astype(int),
                        np.array(data.shape) - 1)
    slices = tuple(slice(l, h + 1) for l, h in zip(lo_vox, hi_vox))
    subvol = data[slices]

    # Bone-like voxels: above cancellous floor, below metal, not artifact
    bone_mask = (subvol > 200) & (subvol < metal_thresh)
    if artifact_mask is not None:
        bone_mask &= ~artifact_mask[slices]

    sub_affine = affine.copy()
    sub_affine[:3, 3] = (affine @ np.append(lo_vox, 1))[:3]
    candidate_pts = convert_to_points(bone_mask, sub_affine)
    candidate_hu = subvol[bone_mask].astype(float)

    # Quantize to iso_res grid
    candidate_pts, candidate_hu = sparse_quantize(candidate_pts / iso_res, candidate_hu)
    candidate_pts = candidate_pts.astype(float) * iso_res

    # Assign each candidate to nearest vertebra, filter by distance
    lm_tree = KDTree(all_posed)
    d, idx = lm_tree.query(candidate_pts)
    assignment = posed_owner[idx]
    nearby = d < initial_radius
    candidate_pts, candidate_hu, assignment = (
        candidate_pts[nearby], candidate_hu[nearby], assignment[nearby])

    # Score = HU normalized by vertebra's median density
    median_hu = np.array([np.median(candidate_hu[assignment == jj])
                          if (assignment == jj).any() else 1.0
                          for jj in range(nJ)])
    scores = candidate_hu / median_hu[assignment]

    # Keep top N (N = preop total)
    n_target = sum(len(p) for p in spine.landmarks)
    if len(scores) > n_target:
        cutoff = np.partition(scores, -n_target)[-n_target]
        keep = scores >= cutoff
    else:
        keep = np.ones(len(scores), dtype=bool)
    result = candidate_pts[keep]

    # Log per-level breakdown
    kept_assignment = assignment[keep]
    for jj in range(nJ):
        name = level_names[jj] if level_names else str(jj)
        log.info('Level %s: %d kept (preop %d, median %.0f HU)',
                 name, int((kept_assignment == jj).sum()),
                 len(spine.landmarks[jj]), median_hu[jj])
    log.info('Re-extracted %d postop cortical points (preop total: %d)', len(result), n_target)
    return result


def _perlevel_refit(spine, postop_pts, icp_affs, ratios, kdtrees,
                    postop_img=None, artifact_mask=None, metal_thresh=2000,
                    level_names=None, iso_res=1.5, initial_radius=12.):
    """Per-level differential evolution refit with block coordinate descent.

    Each vertebra is optimized independently in 6D (3 rotation + 3 translation)
    using scipy.optimize.differential_evolution. Inter-vertebral coupling is
    handled by Jacobi-style sweeps: all levels are optimized in parallel against
    a frozen snapshot of neighbors, then all updated at once. Two coarse-to-fine
    stages with 3 sweeps each.
    """
    from scipy.optimize import differential_evolution

    nJ = spine.nJ
    n_jobs = max(1, os.cpu_count() - 2)

    # Re-extract postop points with per-vertebra adaptive thresholds
    if postop_img is not None:
        postop_pts = _adaptive_postop_pts(postop_img, spine, icp_affs,
                                           iso_res=iso_res, initial_radius=initial_radius,
                                           artifact_mask=artifact_mask, metal_thresh=metal_thresh,
                                           level_names=level_names)

    # Compute baseline ratios on the adaptive points (for fair before/after comparison)
    baseline_ratios = np.zeros(nJ)
    for jj in range(nJ):
        d = kdtrees[jj].query(transform_points_inverse(icp_affs[jj], postop_pts))[0]
        baseline_ratios[jj] = np.sum(d < iso_res) / len(spine.landmarks[jj])

    # Initial params from ICP result
    init_twist, init_trans = extract_params(icp_affs)
    current_params = np.column_stack((init_twist, init_trans))  # (nJ, 6)
    current_affs = icp_affs.copy()

    # Precompute model priors from ICP result
    trunk = (np.arange(nJ) - 1).tolist()
    model_rel_twist, model_rel_trans = aff_to_rel_params(trunk, icp_affs)
    inter_vertebral_dist = np.linalg.norm(model_rel_trans, axis=1)

    # Build postop KDTree
    postop_tree = KDTree(postop_pts)
    total_landmarks = sum(len(lm) for lm in spine.landmarks)
    N = min(len(postop_pts), total_landmarks)

    # Coarse-to-fine: (n_pts, iso_res_stage, n_sweeps)
    # Coarse stage uses wider kernel (3×iso_res) for smoother landscape
    stages = [(N // 4, 3 * iso_res, 3), (N, iso_res, 3)]

    with timed('per-level DE refit'):
        for stage_idx, (n_pts, stage_iso, n_sweeps) in enumerate(stages):
            # Subsample postop + landmark points
            stride = max(1, len(postop_pts) // n_pts)
            pts = postop_pts[::stride]
            nP = len(pts)

            # Voronoi assignment: each postop point → nearest vertebra
            D_assign = 100 * np.ones([nJ, nP])
            for jj in range(nJ):
                preop_inv = transform_points_inverse(current_affs[jj], pts)
                d = kdtrees[jj].query(preop_inv)[0]
                nearby = d < 2 * initial_radius
                D_assign[jj, nearby] = d[nearby]
            assignments = np.argmin(D_assign, axis=0)
            # Exclude points far from all levels
            assignments[np.min(D_assign, axis=0) >= 2 * initial_radius] = -1

            lm_stride = max(1, total_landmarks // n_pts)
            query_lm = [lm[::lm_stride] for lm in spine.landmarks]

            # Per-level postop trees (Voronoi-restricted, for m2d)
            postop_trees_j = []
            for jj in range(nJ):
                j_pts = pts[assignments == jj]
                postop_trees_j.append(KDTree(j_pts) if len(j_pts) > 0
                                      else KDTree(np.zeros((1, 3))))

            for sweep in range(n_sweeps):
                # Snapshot current affs for Jacobi-style parallel update
                snapshot_affs = current_affs.copy()

                def _opt_level(j):
                    x0_j = current_params[j]
                    bounds = list(zip(
                        x0_j[:3] - np.radians(5), x0_j[:3] + np.radians(5)
                    )) + list(zip(
                        x0_j[3:] - 8.0, x0_j[3:] + 8.0
                    ))
                    result = differential_evolution(
                        _level_cost, bounds,
                        args=(j, snapshot_affs, kdtrees, assignments, pts,
                              postop_trees_j[j], query_lm[j],
                              model_rel_twist, model_rel_trans,
                              inter_vertebral_dist, nJ, stage_iso),
                        x0=x0_j, maxiter=100, tol=0.01,
                        seed=42, polish=False)
                    return j, result.x, result.fun

                # Parallel across levels within sweep
                results = Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(_opt_level)(j) for j in range(nJ))

                # Update all levels at once (Jacobi)
                costs = np.zeros(nJ)
                for j, opt_params_j, opt_cost in results:
                    current_params[j] = opt_params_j
                    costs[j] = opt_cost
                current_affs = make_aff(current_params[:, :3], current_params[:, 3:])

                log.info('DE stage %d sweep %d: mean cost %.4f, max cost %.4f',
                         stage_idx, sweep, np.mean(costs), np.max(costs))

    # Final transforms
    opt_affs = current_affs

    # Per-level diagnostics
    new_ratios = np.zeros(nJ)
    for jj in range(nJ):
        preop_inv = transform_points_inverse(opt_affs[jj], postop_pts)
        d = kdtrees[jj].query(preop_inv)[0]
        new_ratios[jj] = np.sum(d < iso_res) / len(spine.landmarks[jj])

    opt_twist, opt_trans = current_params[:, :3], current_params[:, 3:]
    for jj in range(nJ):
        name = level_names[jj] if level_names else str(jj)
        d_trans = np.linalg.norm(opt_trans[jj] - init_trans[jj])
        d_rot = np.degrees(np.linalg.norm(opt_twist[jj] - init_twist[jj]))
        log.info('DE %s: ratio %.3f -> %.3f (%+.3f), moved %.2fmm / %.2f°',
                 name, baseline_ratios[jj], new_ratios[jj],
                 new_ratios[jj] - baseline_ratios[jj], d_trans, d_rot)

    # Sensitivity check (QC only, not used for freeze decisions)
    c2_qc = 2.0 ** 2
    eps_qc = np.array([np.radians(1.0)] * 3 + [1.0] * 3)
    for jj in range(nJ):
        def _cost_j(p6, _j=jj):
            aff_j = make_aff(p6[:3][None], p6[3:][None])[0]
            d = postop_tree.query(transform_points_forward(aff_j, spine.landmarks[_j]))[0]
            return np.mean(d**2 / (c2_qc + d**2))
        grad = approx_fprime(current_params[jj], _cost_j, eps_qc)
        name = level_names[jj] if level_names else str(jj)
        log.info('DE %s: sensitivity %.4f', name, float(np.linalg.norm(grad)))

    return opt_affs, new_ratios


def _particle_bp_refit(spine, postop_pts, icp_affs, ratios, kdtrees,
                       level_names=None, iso_res=1.5, screw_levels=None):
    """Per-level refit using particle belief propagation (D-PMP).

    Builds a chain-structured factor graph where each node is a vertebra
    parameterized by 6D (twist + translation). Local costs use m2d
    soft-ratio (preop landmarks → postop tree); pairwise costs penalize
    deviations from the model's inter-vertebral relative transform.

    Parameters
    ----------
    screw_levels : set or None
        Level names that have screws.  Levels without screws get fewer
        particles (n_min) and a fixed budget throughout.
    """
    from spinescrews.tools.particle_bp import (
        FactorGraph, ParticleBPSolver, SolverConfig)

    nJ = spine.nJ

    # Identify which nodes have screw evidence
    if screw_levels is not None and level_names is not None:
        has_screw = np.array([name in screw_levels for name in level_names])
    else:
        has_screw = np.ones(nJ, dtype=bool)

    n_data = int(has_screw.sum())
    if n_data < nJ:
        prior_names = [level_names[j] for j in range(nJ) if not has_screw[j]]
        log.info('D-PMP: %d/%d levels with screws, %d prior-only: %s',
                 n_data, nJ, nJ - n_data, ', '.join(prior_names))

    # Compute baseline ratios
    baseline_ratios = np.zeros(nJ)
    for jj in range(nJ):
        d = kdtrees[jj].query(transform_points_inverse(icp_affs[jj], postop_pts))[0]
        baseline_ratios[jj] = np.sum(d < iso_res) / len(spine.landmarks[jj])

    # Initial params from ICP result
    init_twist, init_trans = extract_params(icp_affs)
    init_params = np.column_stack((init_twist, init_trans))  # (nJ, 6)

    # Precompute model priors from ICP result
    model_rel_twist, model_rel_trans = aff_to_rel_params(
        (np.arange(nJ) - 1).tolist(), icp_affs)
    inter_vertebral_dist = np.linalg.norm(model_rel_trans, axis=1)

    # Pairwise sigmoid penalty: weighted cost = 0.1 at threshold, 1.0 at 2×threshold.
    # Thresholds set to empirical P99 of ICP→MI differential deltas
    # (neighbor-to-neighbor, 466 pairs across 25 specimens).
    rot_threshold = np.radians(11.0)  # P99 differential rotation
    axial_threshold = 2.85            # P99 differential axial (mm)
    perp_threshold = 5.4              # P99 differential perpendicular (mm)
    ivd_hat = model_rel_trans / np.maximum(inter_vertebral_dist, 1e-6)[:, None]

    def _sigmoid_cost(x, threshold):
        """Exponential ramp: 0 at 0, 0.04 at threshold, 0.4 at 2×threshold.

        With the 2.5× weight applied externally, the effective per-edge cost is:
        0.1 at threshold (P99), 1.0 at 2×threshold, 9.1 at 3×threshold.
        """
        _K = np.log(9)   # 10× ratio between threshold and 2×threshold
        _S = 0.005        # 2.5 * 0.005 * (9-1) = 0.1 at threshold
        return _S * (np.exp(_K * x / threshold) - 1)

    # Coarse-to-fine annealing: 2 stages (wide exploration, then tighten)
    sigma_schedule = np.array([4.0, 2.0])
    s2 = [2 * sigma_schedule[0] ** 2]  # mutable closure cell

    # Global postop tree for m2d queries
    postop_tree_global = KDTree(postop_pts)

    # ---- Define cost functions as closures ----

    def local_cost_fn(node_id, params_batch):
        """Vectorized local cost: m2d soft-ratio for node_id."""
        lm_j = spine.landmarks[node_id]
        affs = make_aff(params_batch[:, :3], params_batch[:, 3:])
        costs = np.empty(len(params_batch))
        for k in range(len(params_batch)):
            d_m2d = postop_tree_global.query(
                transform_points_forward(affs[k], lm_j))[0]
            costs[k] = 1.0 - np.mean(np.exp(-d_m2d**2 / s2[0]))
        return costs

    def _rel_prior_cost(rel_affs, child_idx):
        """Cost from model-relative-pose prior. rel_affs: (N, 4, 4) → (N,).

        Translation error decomposed into IVD-axial (disc compression, tight)
        and perpendicular (in-plane shear, loose) components.
        """
        rel_tw, rel_tr = extract_params(rel_affs)
        tw_diff = ((pi + model_rel_twist[child_idx] - rel_tw) % (2 * pi)) - pi
        rot_err = np.linalg.norm(tw_diff, axis=-1)
        # Vector translation difference, projected onto IVD axis
        tr_diff = rel_tr - model_rel_trans[child_idx]          # (N, 3)
        axial_proj = tr_diff @ ivd_hat[child_idx]              # (N,) signed
        axial_err = np.abs(axial_proj)
        perp_err = np.linalg.norm(
            tr_diff - axial_proj[:, None] * ivd_hat[child_idx], axis=-1)  # (N,)
        return 2.5 * (_sigmoid_cost(rot_err, rot_threshold) +
                      _sigmoid_cost(axial_err, axial_threshold) +
                      _sigmoid_cost(perp_err, perp_threshold))

    def pairwise_cost_fn(node_i, params_i, node_j, params_j):
        """Inter-vertebral prior cost between adjacent nodes (scalar version)."""
        if node_j == node_i - 1:
            child_idx, p_parent, p_child = node_i, params_j, params_i
        elif node_j == node_i + 1:
            child_idx, p_parent, p_child = node_j, params_i, params_j
        else:
            return 0.0
        aff_p = make_aff(p_parent[:3][None], p_parent[3:][None])[0]
        aff_c = make_aff(p_child[:3][None], p_child[3:][None])[0]
        return float(_rel_prior_cost((inverse_rigid(aff_p) @ aff_c)[None], child_idx)[0])

    def pairwise_cost_matrix_fn(node_i, params_i_batch, node_j, params_j_batch):
        """Vectorized pairwise cost: (n_i, n_j) matrix for all particle pairs."""
        n_i, n_j = len(params_i_batch), len(params_j_batch)
        if node_j == node_i - 1:
            child_idx = node_i
            parent_batch, child_batch = params_j_batch, params_i_batch
            n_parent, n_child = n_j, n_i
            transpose = True  # result is (n_child, n_parent) = (n_i, n_j)
        elif node_j == node_i + 1:
            child_idx = node_j
            parent_batch, child_batch = params_i_batch, params_j_batch
            n_parent, n_child = n_i, n_j
            transpose = False  # result is (n_parent, n_child) = (n_i, n_j)
        else:
            return np.zeros((n_i, n_j))

        affs_child = make_aff(child_batch[:, :3], child_batch[:, 3:])
        inv_parents = np.linalg.inv(
            make_aff(parent_batch[:, :3], parent_batch[:, 3:]))

        cost = np.empty((n_parent, n_child))
        for p in range(n_parent):
            cost[p] = _rel_prior_cost(inv_parents[p] @ affs_child, child_idx)

        if transpose:
            return cost.T
        return cost

    def neighbor_proposal_fn(node_i, node_j, params_j):
        """Propose a pose for node_i given neighbor node_j's configuration."""
        aff_j = make_aff(params_j[:3][None], params_j[3:][None])[0]
        if node_i == node_j + 1:
            # node_i is child of node_j: apply model relative transform
            rel = make_aff(model_rel_twist[node_i:node_i+1],
                           model_rel_trans[node_i:node_i+1])[0]
            proposed_aff = aff_j @ rel
        elif node_i == node_j - 1:
            # node_i is parent of node_j: apply inverse relative transform
            rel = make_aff(model_rel_twist[node_j:node_j+1],
                           model_rel_trans[node_j:node_j+1])[0]
            proposed_aff = aff_j @ inverse_rigid(rel)
        else:
            # Non-adjacent: just return the neighbor params as-is
            return params_j.copy()
        tw, tr = extract_params(proposed_aff[None])
        return np.concatenate([tw[0], tr[0]])

    # ---- Build factor graph ----
    edges = [(j, j + 1) for j in range(nJ - 1)]
    # param_scale: physical units for normalizing particle distances
    # [rad, rad, rad, mm, mm, mm]
    param_scale = np.array([np.radians(8)] * 3 + [10.0] * 3)

    graph = FactorGraph(nJ, 6, edges,
                        local_cost_fn, pairwise_cost_fn, neighbor_proposal_fn,
                        pairwise_cost_matrix_fn=pairwise_cost_matrix_fn,
                        param_scale=param_scale)

    # ---- Build stage callbacks with coarse-to-fine sigma annealing ----
    def _anneal(stage_num):
        def cb(best_params, psets):
            s2[0] = 2 * sigma_schedule[stage_num] ** 2
            log.info('D-PMP: stage %d — σ=%.1fmm', stage_num, sigma_schedule[stage_num])
        return cb

    stage_callbacks = [_anneal(k) for k in range(1, len(sigma_schedule))]

    # ---- Configure and run solver ----
    # Noise in physical units: 5° rotation, 5mm translation
    phys_noise = np.array([np.radians(5)] * 3 + [5.0] * 3)
    # Random walk: ~1.4° / 2mm initial, decays by 0.97/iter
    rw_noise = np.array([np.radians(1)] * 3 + [2.0] * 3)

    # Non-screw levels get a fixed small budget throughout
    fixed_budget = {j: 3 for j in range(nJ) if not has_screw[j]}
    init_counts = [15 if has_screw[j] else 3 for j in range(nJ)]
    config = SolverConfig(
        n_iterations=20,
        particles_per_node=15,
        total_budget=np.sum(init_counts),
        n_min=5,
        n_max=3 * 15,
        refine_iterations=4,
        random_walk_noise_std=rw_noise,
        random_walk_decay=0.97,
        neighbor_proposals=0.5,
        neighbor_noise_std=phys_noise,
        convergence_tol=0.005,
        convergence_patience=5,
        fixed_budget=fixed_budget or None,
        seed=42,
    )

    # Initialize particles: full budget for screw levels, n_min for prior-only
    rng = np.random.default_rng(42)
    initial_particles = {}
    for j in range(nJ):
        n = config.particles_per_node if has_screw[j] else config.n_min
        pts = init_params[j] + rng.normal(0, phys_noise, size=(n, 6))
        pts[0] = init_params[j]  # always include the ICP solution
        initial_particles[j] = pts

    with timed('particle BP refit'):
        result = ParticleBPSolver(graph, config).solve(
            initial_particles=initial_particles,
            stage_callbacks=stage_callbacks,
        )

    log.info('D-PMP: %d iterations, converged=%s, global_cost=%.4f',
             result.n_iterations, result.converged, result.global_cost)

    # Extract optimized transforms
    opt_params = np.array([result.best_params[j] for j in range(nJ)])
    opt_affs = make_aff(opt_params[:, :3], opt_params[:, 3:])

    # Per-level diagnostics
    new_ratios = np.zeros(nJ)
    for jj in range(nJ):
        preop_inv = transform_points_inverse(opt_affs[jj], postop_pts)
        d = kdtrees[jj].query(preop_inv)[0]
        new_ratios[jj] = np.sum(d < iso_res) / len(spine.landmarks[jj])

    opt_twist, opt_trans = opt_params[:, :3], opt_params[:, 3:]
    for jj in range(nJ):
        if not has_screw[jj]:
            continue
        name = level_names[jj] if level_names else str(jj)
        d_trans = np.linalg.norm(opt_trans[jj] - init_trans[jj])
        d_rot = np.degrees(np.linalg.norm(opt_twist[jj] - init_twist[jj]))
        if d_trans < 1.0 and d_rot < 1.0:
            continue
        log.info('BP %s: ratio %.3f -> %.3f (%+.3f), moved %.2fmm / %.2f°',
                 name, baseline_ratios[jj], new_ratios[jj],
                 new_ratios[jj] - baseline_ratios[jj], d_trans, d_rot)

    return opt_affs, new_ratios


def corrective_registrations(spine, postop_pts, icp_affs, ratios, required, kdtrees, D,
                             level_names=None, iso_res=1.5, initial_radius=12., ratio_thresh=0.75):
    """Re-align poorly-registered levels by anchoring to their best-registered neighbor."""
    thresh = max(ratio_thresh, (3 / 4) * np.percentile(ratios, 75))  # might enforce a higher threshold
    model_rel_twist, model_rel_trans = spine.aff_to_rel_params(spine.default_aff)

    need_correction = (ratios < thresh) & required
    stable = ratios > ratio_thresh
    while np.any(need_correction):
        # aligned is a boolean array indicating which levels are already well-aligned, whether or not that level has screws
        # need_correction is a boolean array indicating which levels are not well-aligned AND have screws
        result = _get_paired_levels(ratios, stable, need_correction)
        if result is None:
            break
        anchor, floater = result
        anchor_name = level_names[anchor] if level_names else str(anchor)
        floater_name = level_names[floater] if level_names else str(floater)
        old_ratio = ratios[floater]

        icp_affs[floater], ratios[floater] = _wiggle_to_fit(
            D, kdtrees[floater], icp_affs, anchor, floater, stable,
            spine.landmarks, model_rel_twist, model_rel_trans,
            postop_pts, iso_res, initial_radius)
        log.info('Corrected %s (anchor=%s): ratio %.2f -> %.2f', floater_name, anchor_name, old_ratio, ratios[floater])
        stable[floater], need_correction[floater] = True, False

        if any(need_correction):
            posed_pts = transform_points_forward(icp_affs[floater], spine.landmarks[floater])
            kdtrees[floater] = KDTree(posed_pts)
            D[floater] = kdtrees[floater].query(postop_pts)[0]

    return icp_affs, ratios
