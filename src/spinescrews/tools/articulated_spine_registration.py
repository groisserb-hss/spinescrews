"""Articulated multi-level registration of the preop spine model to the postop CT.

`align_spine_to_CT()` drives the postop fit: it builds a cortical-bone point cloud (with a
metal-streak artifact mask), runs multi-level articulated ICP with a Geman-McClure robust loss,
and refines the per-vertebra poses with a discrete particle-belief-propagation (D-PMP) refit
that respects the spine's kinematic chain.
"""

import os
import numpy as np
from numpy import pi
import logging
import igl
from bg3dtools.transforms_unified import transform_points_forward, transform_points_inverse, make_aff, extract_params, aff_to_rel_params, inverse_rigid
from spinescrews.tools.articulated_models.spine import Spine
import nibabel as nib
from scipy.spatial import KDTree
from scipy.optimize import least_squares
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


log = logging.getLogger(__name__)


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
    init_bonepts = np.vstack(spine.build_landmarks(initial_affs))
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
                              np.vstack(model_pts), empty_faces,
                              v_rgb=np.vstack(model_rgb))
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

    # Global bone threshold (median of the per-level thresholds)
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
            faces = np.vstack([faces, f + len(verts)])
            verts = np.vstack([verts, v])
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


def _build_artifact_mask_fast(postop_img, screws, screw_proximity_mm=20.0,
                              metal_threshold=None):
    """Build artifact mask: threshold -> open -> proximity filter -> dilate -> streak threshold.

    Fast morphology-based artifact mask (dilated screw region + high-HU streaks).
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
    screw_tree = KDTree(np.vstack(screw_verts))

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
    J_sparse = np.vstack([J_sparse, J_reg])
    with timed('least_squares (articulated ICP)'):
        result = least_squares(cost_fun, init_params, method='trf', verbose=0, ftol=0.01, jac_sparsity=J_sparse)
    opt_loss = float(np.sum(cost_fun(result.x)**2))
    opt_params = result.x.reshape([nJ, 6])
    opt_twist, opt_trans = opt_params[:, :3], opt_params[:, 3:]
    log.info('Aligned spine to postop pts: loss %.3f -> %.3f' % (initial_loss, opt_loss))

    opt_affs = make_aff(opt_twist, opt_trans)

    return opt_affs, kdtrees, initial_loss, opt_loss


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


