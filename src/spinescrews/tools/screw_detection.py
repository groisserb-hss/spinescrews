"""Metal pedicle-screw detection in the postop CT.

`detect_screws()` localizes each implanted screw: RANSAC/ICP initialization, articulated-spine
optimization, per-screw point-cloud fitting, and a final HU-weighted refinement of each screw's
axis and endpoints.
"""

from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

import igl
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares

from joblib import Parallel, delayed, effective_n_jobs

from spinescrews.tools import possible_levels
from spinescrews.tools.paths import timed
from spinescrews.tools.screw_models import Screw
from spinescrews.tools.nifti_utils import HU_CLIP, compute_metal_threshold, resample_to_pitch
from spinescrews.tools.articulated_models.base_unified import Articulated
from bg3dtools.transforms_unified import make_aff, transform_points_forward, extract_params, transform_points_inverse, rel_params_to_aff, aff_to_rel_params
from bg3dtools.pointclouds.registration import pc_icp
from bg3dtools.pointclouds.quantize import convert_to_points, sparse_quantize
CLOUD_GRID_MM = 2.0  # grid spacing for data sub_cloud and model screw clouds


# ---------------------------------------------------------------------------
# Geometry helpers (used by detect_screws logging)
# ---------------------------------------------------------------------------

def _endpts_to_axes(endpts: np.ndarray) -> np.ndarray:
    """Unit axis vectors from (N, 2, 3) endpoint array."""
    vecs = endpts[:, 1] - endpts[:, 0]
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row angle in degrees between unit vectors."""
    dots = np.clip(np.sum(a * b, axis=1), -1, 1)
    return np.degrees(np.arccos(dots))


def _off_axis(displacements: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Off-axis (perpendicular) component of displacement vectors."""
    along = np.sum(displacements * axes, axis=1, keepdims=True) * axes
    return np.linalg.norm(displacements - along, axis=1)


def _get_state(screws: list[Screw]) -> tuple[np.ndarray, np.ndarray]:
    """Current detected entry points and axis vectors for a list of screws."""
    entries = np.array([s.detected_entry.copy() for s in screws])
    axes = np.array([s.axis(planned=False) for s in screws])
    return entries, axes


def _log_screw_changes(stage: str,
                       prev_entries: np.ndarray, prev_axes: np.ndarray,
                       curr_entries: np.ndarray, curr_axes: np.ndarray,
                       screw_names: list[str]) -> None:
    """Log per-screw lateral displacement and rotation between two states."""
    lateral = _off_axis(curr_entries - prev_entries, curr_axes)
    angles = _angle_between(prev_axes, curr_axes)
    for name, d, a in zip(screw_names, lateral, angles):
        log.debug('%s: %s lateral %.2f mm, rot %.2f deg', name, stage, d, a)
    log.info('%s mean: lateral %.2f mm (max %.2f), rot %.2f deg (max %.2f)',
             stage, float(np.mean(lateral)), float(np.max(lateral)),
             float(np.mean(angles)), float(np.max(angles)))


# ---------------------------------------------------------------------------
# InstrumentedSpine
# ---------------------------------------------------------------------------

class InstrumentedSpine(Articulated):
    """Articulated spine model with screws branching off vertebrae.

    Planned spine transforms and screw entry/tip points define the rest pose.
    Screw transforms are computed to take a screw at the origin to its planned
    entry point, relative to the parent vertebra.
    """

    def __init__(self, screws: list[Screw], spine_tforms: dict[str, np.ndarray]) -> None:
        """Build articulated spine+screw model from planned screws and vertebral transforms."""
        num_screws = len(screws)
        # "spine" of the model is the vertebra
        spine_levels = [l for l in possible_levels if l in spine_tforms.keys()]  # sort levels anatomically
        self.spine_levels = spine_levels
        spine_trunk = np.arange(-1, len(spine_levels) - 1).tolist()
        self._spine_trunk = spine_trunk

        # screws branch off of spine
        screw_trunk = [spine_levels.index(screw.level) for screw in screws]
        self._screw_parents = screw_trunk
        trunk = spine_trunk + screw_trunk

        Articulated.__init__(self, trunk)

        #
        spine_tform_list = np.stack([spine_tforms[l] for l in spine_levels], 0)
        rel_theta, rel_trans = aff_to_rel_params(spine_trunk, spine_tform_list)
        self.plan_spine_theta = rel_theta.copy()
        self.plan_spine_theta[0] = 0
        self.plan_spine_trans = rel_trans.copy()
        self.plan_spine_trans[0] = 0
        self.plan_screw_theta = np.zeros([num_screws, 3])
        self.plan_screw_trans = np.zeros([num_screws, 3])

        # get rest position of vertebrae and screws
        self.verts = np.zeros([len(screws), 2, 3])
        self.screw_clouds = []
        for ss, screw in enumerate(screws):
            vertebral_tform = spine_tforms[screw.level]
            posed_pts = np.row_stack([screw.planned_entry, screw.planned_tip])
            unposed_pts = transform_points_inverse(vertebral_tform, posed_pts)
            self.plan_screw_trans[ss] = unposed_pts[0]
            self.verts[ss, 1] = unposed_pts[1] - unposed_pts[0]

            # build a dense point cloud from the screw mesh, quantized to the same grid as sub_cloud
            v_world, _ = screw.build_mesh(planned=True)
            v_rest = transform_points_inverse(vertebral_tform, v_world) - unposed_pts[0]
            screw_pts = sparse_quantize(v_rest / CLOUD_GRID_MM).astype(float) * CLOUD_GRID_MM
            self.screw_clouds.append(screw_pts)

    @classmethod
    def with_backend(cls, model: InstrumentedSpine, backend: str) -> InstrumentedSpine:
        """Not implemented — InstrumentedSpine only supports numpy backend."""
        raise NotImplementedError("InstrumentedSpine only supports numpy backend")

    @property
    def num_bones(self) -> int:
        """Number of vertebral levels in the model."""
        return len(self.spine_levels)

    @property
    def num_screws(self) -> int:
        """Number of placed (non-skip) screws."""
        return len(self.screw_clouds)

    def build_model(self, abs_affs: np.ndarray, shape: str | None = None) -> np.ndarray:
        """Pose screw geometry using absolute affine transforms.

        Parameters
        ----------
        abs_affs : np.ndarray
            (num_bones + num_screws, 4, 4) absolute affines.
        shape : str or None
            'endpts' or None for entry/tip pairs (N, 2, 3).
            'full' for dense screw point clouds.
        """
        if shape is None or shape == 'endpts':
            rest_pts = self.verts
        elif shape == 'full':
            rest_pts = self.screw_clouds
        else:
            raise ValueError('Invalid shape %s for build_model' % shape)

        if len(abs_affs) != (self.num_screws + self.num_bones):
            raise ValueError('expected %d affines (screws + bones), got %d' %
                             (self.num_screws + self.num_bones, len(abs_affs)))
        screw_affs = abs_affs[-self.num_screws:]  # only need screw affines

        posed_pts = []
        for pts, tform in zip(rest_pts, screw_affs):
            posed_pts.append(transform_points_forward(tform, pts))

        return np.row_stack(posed_pts)

    @property
    def nV(self) -> int:
        """Total number of rest-pose screw endpoint vertices."""
        return np.sum([len(v) for v in self.verts])

    def vectorize_params(self, twist: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """Flatten (N,3) twist and (N,3) trans arrays into a single 1D parameter vector."""
        if twist.ndim != 2 or trans.ndim != 2:
            raise ValueError('twist and trans must be 2D arrays; got ndim=%d, %d' % (twist.ndim, trans.ndim))
        return np.concatenate((twist, trans), axis=1).flatten()

    def parse_params(self, vectorized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape 1D parameter vector back into (N,3) twist and (N,3) trans arrays."""
        A = np.reshape(vectorized, (self.num_screws + self.num_bones, 6))
        return A[:, :3], A[:, 3:]

    def params_to_abs_affs(self, params: np.ndarray) -> np.ndarray:
        """Convert hybrid parameter vector to absolute affines.

        Vertebral params (indices 0..num_bones-1) are absolute (theta, trans).
        Screw params (indices num_bones..) are relative to parent vertebra.
        Returns (num_bones + num_screws, 4, 4) absolute affines.
        """
        theta, trans = self.parse_params(params)

        vert_affs = make_aff(theta[:self.num_bones], trans[:self.num_bones])

        screw_rel_affs = make_aff(theta[self.num_bones:], trans[self.num_bones:])
        parent_affs = vert_affs[self._screw_parents]
        screw_abs_affs = parent_affs @ screw_rel_affs

        return np.concatenate([vert_affs, screw_abs_affs], axis=0)

    def abs_affs_to_params(self, abs_affs: np.ndarray) -> np.ndarray:
        """Convert absolute affines to hybrid parameter vector.

        Vertebral affines -> absolute (theta, trans).
        Screw affines -> relative to parent vertebra (theta, trans).
        """
        vert_affs = abs_affs[:self.num_bones]
        screw_abs_affs = abs_affs[self.num_bones:]

        vert_theta, vert_trans = extract_params(vert_affs)

        parent_inv = np.linalg.inv(vert_affs[self._screw_parents])
        screw_rel_affs = parent_inv @ screw_abs_affs
        screw_theta, screw_trans = extract_params(screw_rel_affs)

        theta = np.row_stack([vert_theta, screw_theta])
        trans = np.row_stack([vert_trans, screw_trans])
        return self.vectorize_params(theta, trans)

    def initialize_alignment(self, pts: np.ndarray, density: np.ndarray,
                             stall_limit: int = 500,
                             n_jobs: int = 1) -> tuple[float, np.ndarray]:
        """Global search for rigid-body alignment between planned and detected screws.

        Parameters
        ----------
        pts : np.ndarray
            (N, 3) metal point cloud.
        density : np.ndarray
            Per-point density for weighted sampling.
        stall_limit : int
            Stop after this many iterations without improvement.
        n_jobs : int
            Number of parallel threads for ICP trials (1 = sequential).
        """
        rest_theta = np.row_stack([self.plan_spine_theta, self.plan_screw_theta])
        rest_trans = np.row_stack([self.plan_spine_trans, self.plan_screw_trans])
        abs_affs = rel_params_to_aff(self.trunk, rest_theta, rest_trans)
        template_pts = self.build_model(abs_affs).reshape([self.num_screws, 2, 3])
        # robust error distance determined by average shaft length
        sigma = np.mean(np.linalg.norm(np.diff(template_pts, axis=1), axis=-1))
        template_heads = template_pts[:, 0]

        # construct a skeleton of the template screws to use to compute error distances
        screw_skeleton = self.build_model(abs_affs, 'full')

        sub_cloud = sparse_quantize(pts).astype(float)
        cloud_tree = KDTree(sub_cloud)

        batch_size = effective_n_jobs(n_jobs)
        parent_rng = np.random.default_rng()

        best_err, best_tform, stall, ii, hit = 999, None, 0, 0, 0

        def _process_trial(test_err, tform):
            nonlocal best_err, best_tform, stall, ii, hit
            if test_err < best_err:
                prev_err = best_err
                best_err, best_tform = test_err, tform
                stall = 0
                hit += 1
                pct = 100 * (prev_err - best_err) / prev_err
                log.debug('  init hit %2d @ iter %4d: err %.4f (%.2f%% improvement)',
                         hit, ii, best_err, pct)
            else:
                stall += 1
            ii += 1

        trial_args = (pts, density, template_heads, screw_skeleton,
                      sub_cloud, cloud_tree, sigma, self.num_screws)
        log.info('initialize_alignment: using %d threads', batch_size)
        while stall < stall_limit:
            seeds = [int(parent_rng.integers(2**63)) for _ in range(batch_size)]
            results = Parallel(n_jobs=batch_size, prefer='threads')(
                delayed(_init_trial)(np.random.default_rng(s), *trial_args)
                for s in seeds)
            for test_err, tform in results:
                _process_trial(test_err, tform)
            if ii % 200 == 0:
                log.debug('  init progress: iter %d, best_err %.4f, stall %d/%d',
                          ii, best_err, stall, stall_limit)

        log.info("Finished initialization: %d iters, %d hits, best_err %.4f, ", ii, hit, best_err)
        # update the root transformation with best alignment
        rest_theta[0], rest_trans[0] = extract_params(best_tform)
        abs_affs = rel_params_to_aff(self.trunk, rest_theta, rest_trans)
        return best_err, abs_affs

    def optimize_alignment(self, pts: np.ndarray, init_aff: np.ndarray,
                           rel_t: float,
                           va: float = 5., vb: float = .5,
                           sa: float = 5000., sb: float = 500.
                           ) -> tuple[float, np.ndarray]:
        """Refine alignment between planned and detected screws via least-squares.

        Parameters
        ----------
        pts : np.ndarray
            (N, 3) metal point cloud.
        init_aff : np.ndarray
            (num_bones + num_screws, 4, 4) initial absolute affines.
        rel_t : float
            Relative sigma for Geman-McClure robust error (fraction of shaft length).
        va, vb : float
            Vertebral rotation/translation regularization weights.
        sa, sb : float
            Screw rotation/translation regularization weights.
        """
        # robust error distance determined by average shaft length
        template_pts = self.build_model(init_aff).reshape([self.num_screws, 2, 3])
        sigma = rel_t * np.mean(np.linalg.norm(np.diff(template_pts, axis=1), axis=-1))

        sub_cloud = sparse_quantize(pts / CLOUD_GRID_MM).astype(float) * CLOUD_GRID_MM
        cloud_tree = KDTree(sub_cloud)

        # track sizes of each cost component for breakdown
        n_sub = len(sub_cloud)
        n_tmpl = [0]  # mutable — set on first call
        _nfev = [0]
        _best_cost = [np.inf]
        _log_progress = [False]

        def cost_fun(params):
            """Residual vector: data + template distance costs + vertebral/screw regularization."""
            theta, trans = self.parse_params(params)

            # Forward: hybrid params -> absolute affines -> posed screw points
            abs_aff = self.params_to_abs_affs(params)
            posed_screws = self.build_model(abs_aff, 'full')

            # data → template: penalizes unexplained data points
            kdtree = KDTree(posed_screws)
            dist2 = kdtree.query(sub_cloud)[0] ** 2
            data_cost = dist2 / (sigma ** 2 + dist2)
            data_cost *= 1000 / len(sub_cloud)

            # template → data: penalizes unmatched template points
            d2_tmpl = cloud_tree.query(posed_screws)[0] ** 2
            tmpl_cost = d2_tmpl / (sigma ** 2 + d2_tmpl)
            tmpl_cost *= 1000 / len(posed_screws)
            n_tmpl[0] = len(posed_screws)

            # Vertebral regularization — relative params from absolute vertebral affines
            vert_affs = abs_aff[:self.num_bones]
            spine_rel_theta, spine_rel_trans = aff_to_rel_params(
                self._spine_trunk, vert_affs)
            vertebral_twist = spine_rel_theta[1:] - self.plan_spine_theta[1:]
            vertebral_twist_err = (va * vertebral_twist / self.num_bones).flatten()
            vertebral_offset = spine_rel_trans[1:] - self.plan_spine_trans[1:]
            vertebral_offset_err = (vb * vertebral_offset / self.num_bones).flatten()

            # Screw regularization — directly from optimizer's relative params
            screw_twist = theta[self.num_bones:] - self.plan_screw_theta
            screw_twist_err = (sa * screw_twist / self.num_screws).flatten()
            screw_offset = trans[self.num_bones:] - self.plan_screw_trans
            screw_offset_err = (sb * screw_offset / self.num_screws).flatten()

            err = np.concatenate((data_cost, tmpl_cost, vertebral_twist_err,
                                  vertebral_offset_err, screw_twist_err,
                                  screw_offset_err))

            if _log_progress[0]:
                _nfev[0] += 1
                total = float(np.dot(err, err))
                if _nfev[0] % 500 == 0 or total < _best_cost[0] * 0.9:
                    if total < _best_cost[0]:
                        _best_cost[0] = total
                    i = 0
                    parts = []
                    for name, size in [('data', n_sub), ('tmpl', n_tmpl[0]),
                                       ('v_rot', (self.num_bones - 1) * 3),
                                       ('v_tr', (self.num_bones - 1) * 3),
                                       ('s_rot', self.num_screws * 3),
                                       ('s_tr', self.num_screws * 3)]:
                        parts.append('%s=%.2f' % (name, float(np.sum(err[i:i + size] ** 2))))
                        i += size
                    log.debug('  nfev=%-5d cost=%.3f  %s', _nfev[0], total, '  '.join(parts))

            return err

        def _cost_breakdown(params):
            """Decompose residual vector into per-component sum-of-squares."""
            r = cost_fun(params)
            i = 0
            components = {}
            for name, size in [('data', n_sub), ('template', n_tmpl[0]),
                               ('vert_rot', (self.num_bones - 1) * 3),
                               ('vert_trans', (self.num_bones - 1) * 3),
                               ('screw_rot', self.num_screws * 3),
                               ('screw_trans', self.num_screws * 3)]:
                components[name] = float(np.sum(r[i:i + size] ** 2))
                i += size
            return components

        # perform optimization to align screws to point cloud
        init_params = self.abs_affs_to_params(init_aff)
        init_breakdown = _cost_breakdown(init_params)
        initial_cost = sum(init_breakdown.values())
        _log_progress[0] = True
        optimized = least_squares(cost_fun, init_params, method='lm', ftol=0.01, diff_step=0.001)
        _log_progress[0] = False

        final_breakdown = _cost_breakdown(optimized.x)
        opt_err = sum(final_breakdown.values())
        log.info('%s nfev=%d, cost %.2f -> %.2f, optimality %.3f',
                 optimized.message, optimized.nfev, initial_cost, opt_err,
                 optimized.optimality)
        log.debug('  cost breakdown (initial -> final):')
        for comp in init_breakdown:
            log.debug('    %-12s %7.2f -> %7.2f  (%+.2f)', comp,
                     init_breakdown[comp], final_breakdown[comp],
                     final_breakdown[comp] - init_breakdown[comp])

        # parse optimal parameters, convert to absolute transformation
        abs_aff = self.params_to_abs_affs(optimized.x)

        return opt_err, abs_aff


# ---------------------------------------------------------------------------
# Parallel worker functions
# ---------------------------------------------------------------------------

def _init_trial(rng: np.random.Generator,
                pts: np.ndarray, density: np.ndarray,
                template_heads: np.ndarray, screw_skeleton: np.ndarray,
                sub_cloud: np.ndarray, cloud_tree: KDTree,
                sigma: float, num_screws: int) -> tuple[float, np.ndarray]:
    """Run one RANSAC-style ICP trial for initialize_alignment.

    All inputs are read-only shared state.  ``rng`` is a per-call Generator
    so there is no contention between threads.
    """
    sample_density = density / np.sum(density)
    idx = rng.choice(len(pts), 2 * num_screws, p=sample_density)
    sampled_pts = pts[idx]

    init_trans = np.mean(sampled_pts, axis=0) - np.mean(template_heads, axis=0)
    init_trans += rng.standard_normal(3) * sigma
    init_tform = make_aff(rng.standard_normal(3), init_trans)
    tform = pc_icp(sampled_pts, template_heads, init_tform=init_tform, pthresh=90)
    test_screws = transform_points_forward(tform, screw_skeleton)

    # data -> template: penalizes unexplained data points
    kdtree = KDTree(test_screws)
    dist2 = kdtree.query(sub_cloud)[0] ** 2
    err_data = np.mean(dist2 / (sigma ** 2 + dist2))

    # template -> data: penalizes unmatched template points
    d2_tmpl = cloud_tree.query(test_screws)[0] ** 2
    err_tmpl = np.mean(d2_tmpl / (sigma ** 2 + d2_tmpl))

    return err_data + err_tmpl, tform


def _fit_single_screw(screw: Screw, pts: np.ndarray,
                      weights: np.ndarray | None = None,
                      ) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """Fit a single screw to a point cloud.

    Computes shaft coverage, fits the screw model, and logs per-screw changes.
    """
    coverage = _shaft_coverage(screw, pts, weights if weights is not None
                               else np.ones(len(pts)))
    prev_entry = screw.detected_entry.copy()
    prev_ax = screw.axis(planned=False).copy()

    success, mask, fit_metrics = screw.fit_cloud(pts, weights=weights)

    curr_ax = screw.axis(planned=False)
    disp = screw.detected_entry - prev_entry
    lateral = float(np.linalg.norm(disp - np.dot(disp, curr_ax) * curr_ax))
    ang = float(np.degrees(np.arccos(np.clip(np.dot(prev_ax, curr_ax), -1, 1))))
    log.debug('%s: lateral %.2f mm, rot %.2f deg (%d pts, %.0f%% inlier, '
              'coverage %.0f%%)',
              screw.name, lateral, ang, len(pts),
              100 * fit_metrics['inlier_ratio'], 100 * coverage)

    fit_metrics['shaft_coverage'] = coverage
    fit_metrics['lateral_mm'] = lateral
    fit_metrics['angle_deg'] = ang
    return screw.name, screw.detected_entry.copy(), screw.detected_tip.copy(), fit_metrics


def _parallel_fit(stage: str, screws: list[Screw], jobs,
                  fit_info: dict,
                  prev_entries: np.ndarray, prev_axes: np.ndarray,
                  screw_names: list[str],
                  n_jobs: int, prefer: str | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch per-screw fitting jobs, write back results, and log changes.

    Each job must return ``(name, entry, tip, info_dict)``.
    Returns the post-writeback ``(entries, axes)`` state.
    """
    kw = {'n_jobs': n_jobs}
    if prefer is not None:
        kw['prefer'] = prefer
    results = Parallel(**kw)(jobs)
    for screw, (name, entry, tip, info) in zip(screws, results):
        screw.detected_entry = entry
        screw.detected_tip = tip
        fit_info.setdefault(name, {}).update(info)
    curr_entries, curr_axes = _get_state(screws)
    _log_screw_changes(stage, prev_entries, prev_axes,
                       curr_entries, curr_axes, screw_names)
    return curr_entries, curr_axes


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_screws(postop_ct: nib.Nifti1Image, screws: list[Screw],
                  tforms: dict[str, np.ndarray],
                  threshold: int | None = None, n_jobs: int = -3,
                  analysis_dir: str | None = None
                  ) -> tuple[dict[str, np.ndarray], dict]:
    """Detect screws in post-op CT and return spine transforms + metrics.

    Parameters
    ----------
    postop_ct : nib.Nifti1Image
        Greyscale volume of FULL postop spine (RAS orientation).
    screws : list[Screw]
        Screw plan objects (modified in place with detected positions).
    tforms : dict
        Preop vertebral affine transforms {level_name: 4x4 ndarray}.
    threshold : int or None
        HU threshold for metal detection (auto-computed if None).
    n_jobs : int
        Parallel workers for individual screw fitting.
    analysis_dir : str or None
        When provided, intermediate MIP figures are saved to 05_detection/.

    Returns
    -------
    spine_tforms : dict
        {level_name: 4x4 ndarray} initial spine alignment transforms.
    metrics : dict
        Structured quality metrics for summary.json.
    """
    placed_screws = [S for S in screws if S.type != 'skip']
    num_screws = len(placed_screws)
    num_bones = len(tforms)
    screw_names = [s.name for s in placed_screws]
    log.info('Searching for %d screws' % num_screws)
    timings = {}

    # Normalize input to step 05's baseline pitch: slice → 0.625 mm, in-plane
    # preserved unless coarser than 0.8 mm (then → 0.4 mm). Downstream pitch-
    # dependent formulas (DBSCAN eps, grid_mm, etc.) were tuned for this range.
    pitch = np.abs(np.diag(postop_ct.affine[:3, :3]))
    target = (
        pitch[0] if pitch[0] <= 0.8 else 0.4,
        pitch[1] if pitch[1] <= 0.8 else 0.4,
        0.625,
    )
    postop_ct = resample_to_pitch(postop_ct, target)

    plan_model = InstrumentedSpine(placed_screws, tforms)
    # convert to point cloud
    with timed('extract_metal_points', timings):
        pts, density, postop_pitch = extract_metal_points(postop_ct, threshold=threshold)
    # thin to ~0.67mm grid or input pitch, whichever is coarser
    # post-resample max pitch is in [0.625, 0.8]: grid_mm = max(0.667, max_pitch)
    grid_mm = max(1.0 / 1.5, np.max(postop_pitch))
    pts, density = sparse_quantize(pts / grid_mm, density)
    pts = pts.astype(float) * grid_mm
    n_metal_pts = len(pts)
    vol_per_screw = int(n_metal_pts * grid_mm ** 3 / num_screws)
    log.info('Detected %d metal points for a volume of %d mm^3 per screw' %
             (n_metal_pts, vol_per_screw))

    if n_metal_pts == 0:
        raise RuntimeError(
            'No metal points survived clustering. The postop CT may not '
            'contain metal above the threshold, or DBSCAN parameters are too '
            'aggressive for this scan. Try setting screw_detect_threshold in '
            'config.yml to a lower value (current effective threshold logged above).')

    # Prepare MIP rendering data (reuses the threshold already resolved above)
    if analysis_dir is not None:
        from spinescrews.figures.detection_screws import render_mip_with_screws
        from spinescrews.tools.paths import detection_dir as _det_dir
        _fig_det_dir = _det_dir(analysis_dir)
        ct_data = postop_ct.get_fdata()
        ct_affine = postop_ct.affine
        _fig_threshold = threshold if threshold is not None else compute_metal_threshold(ct_data)
        metal_data = np.where(ct_data > _fig_threshold, ct_data, 0.0)

    # planned endpoints as reference
    planned_endpts = np.stack([np.stack([s.planned_entry, s.planned_tip]) for s in placed_screws])
    planned_entries = planned_endpts[:, 0]
    planned_axes = _endpts_to_axes(planned_endpts)

    # 1. match planned screws to point cloud using many randomly initialized ICP alignments
    with timed('initialize_alignment', timings):
        init_err, init_aff = plan_model.initialize_alignment(pts, density, n_jobs=n_jobs)
    log.debug('Initial alignment error: %f' % init_err)
    init_endpts = plan_model.build_model(init_aff, shape='endpts').reshape([-1, 2, 3])
    init_entries = init_endpts[:, 0]
    init_axes = _endpts_to_axes(init_endpts)
    _log_screw_changes('init_alignment', planned_entries, planned_axes,
                       init_entries, init_axes, screw_names)

    # 2a. first parametric optimization (articulated spine parameters) to find vertebral alignment
    log.debug('spine_opt: %d metal points, sigma_rel=0.5, va=0.2 vb=0.1 sa=10000 sb=1000', len(pts))
    with timed('optimize_alignment (spine)', timings):
        spine_cost, spine_aff = plan_model.optimize_alignment(pts, init_aff, 0.4, va=200., vb=10., sa=2000., sb=100.)
    spine_endpts = plan_model.build_model(spine_aff, shape='endpts').reshape([-1, 2, 3])
    spine_entries = spine_endpts[:, 0]
    spine_axes = _endpts_to_axes(spine_endpts)
    _log_screw_changes('spine_opt', init_entries, init_axes,
                       spine_entries, spine_axes, screw_names)

    if analysis_dir is not None:
        endpts = [(e[0], e[1]) for e in spine_endpts]
        render_mip_with_screws(metal_data, ct_affine, endpts,
                               os.path.join(_fig_det_dir, 'global_spine-opt.png'))

    # filter pts to only include points near the spine
    skeletons = plan_model.build_model(spine_aff, 'full')
    kdtree = KDTree(skeletons)
    search_mask = kdtree.query(pts)[0] < 20  # 2 cm search radius
    pts, density = pts[search_mask], density[search_mask]
    log.debug('Filtered to %d points within 20mm of spine model', len(pts))

    # 2b. second optimization to find screw alignment
    log.debug('screw_opt: %d metal points, sigma_rel=0.25, va=5 vb=2 sa=5 sb=2', len(pts))
    with timed('optimize_alignment (screws)', timings):
        opt_err, opt_aff = plan_model.optimize_alignment(pts, spine_aff, 0.15, va=50., vb=2.5, sa=50., sb=2.5)
    opt_endpts = plan_model.build_model(opt_aff, shape='endpts').reshape([-1, 2, 3])
    opt_entries = opt_endpts[:, 0]
    opt_axes = _endpts_to_axes(opt_endpts)
    _log_screw_changes('screw_opt', spine_entries, spine_axes,
                       opt_entries, opt_axes, screw_names)

    if analysis_dir is not None:
        endpts = [(e[0], e[1]) for e in opt_endpts]
        render_mip_with_screws(metal_data, ct_affine, endpts,
                               os.path.join(_fig_det_dir, 'global_screw-opt.png'))

    # convert transformations to screw parameters
    for screw, endpts in zip(placed_screws, opt_endpts):
        screw.detected_entry = endpts[0]
        screw.detected_tip = endpts[1]

    # multi_screw_cloud_fit disabled — articulated optimizer handles nearby screws
    prev_entries, prev_axes = _get_state(placed_screws)
    fit_info: dict = {}

    # fine-tune each screw individually
    with timed('partition_points', timings):
        point_clouds = partition_points(placed_screws, pts)
    log.debug('Partitioned points: %s', [len(pc) for pc in point_clouds])
    with timed('fit_cloud', timings):
        prev_entries, prev_axes = _parallel_fit(
            'fit_cloud', placed_screws,
            (delayed(_fit_single_screw)(screw, neighbors)
             for screw, neighbors in zip(placed_screws, point_clouds)),
            fit_info, prev_entries, prev_axes, screw_names, n_jobs=n_jobs)

    unconverged = [name for name, fm in fit_info.items() if not fm.get('converged', True)]
    if unconverged:
        log.warning('fit_cloud did not converge for %d/%d screws: %s'
                     % (len(unconverged), num_screws, ', '.join(unconverged)))

    # --- HU-weighted refinement pass ---
    with timed('fit_cloud_refine', timings):
        data = postop_ct.get_fdata()
        refine_threshold = compute_metal_threshold(data)
        refine_affine = postop_ct.affine

        # Sample HU-weighted point clouds before parallel fit
        refine_clouds = [_sample_near_screw(screw, data, refine_affine, refine_threshold)
                         for screw in placed_screws]

        # Exclude points that fall within a neighboring screw's mesh
        pre_counts = [len(pts_r) for pts_r, _ in refine_clouds]
        refine_clouds = [
            _exclude_neighbor_points(screw, pts_r, w_r, placed_screws)
            for screw, (pts_r, w_r) in zip(placed_screws, refine_clouds)
        ]
        removed = [pre - len(post[0]) for pre, post in zip(pre_counts, refine_clouds)]
        affected_total = sum(pre for pre, r in zip(pre_counts, removed) if r > 0)
        log.info('Neighbor exclusion: %d/%d points removed from %d screws',
                 sum(removed), affected_total, sum(r > 0 for r in removed))

        _parallel_fit(
            'refine', placed_screws,
            (delayed(_fit_single_screw)(screw, pts_r, weights=w_r)
             for screw, (pts_r, w_r) in zip(placed_screws, refine_clouds)),
            fit_info, prev_entries, prev_axes, screw_names,
            n_jobs=n_jobs, prefer='threads')

    if analysis_dir is not None:
        final_endpts = [(s.detected_entry, s.detected_tip) for s in placed_screws]
        render_mip_with_screws(metal_data, ct_affine, final_endpts,
                               os.path.join(_fig_det_dir, 'detection_screws.png'))

    # --- Validate: every planned screw must have metal along its shaft ---
    MIN_SHAFT_COVERAGE = 0.5
    shaft_coverage_failed = [(s.name, fit_info[s.name]['shaft_coverage']) for s in placed_screws
                             if fit_info[s.name]['shaft_coverage'] < MIN_SHAFT_COVERAGE]
    if shaft_coverage_failed:
        lines = ['Low shaft metal coverage on %d/%d screws:'
                 % (len(shaft_coverage_failed), num_screws)]
        for name, cov in shaft_coverage_failed:
            lines.append('  %s: %.0f%%' % (name, 100 * cov))
        lines.append('Consider setting screw_type=skip in preop_plan.csv for unplaced screws.')
        log.warning('\n'.join(lines))

    # convert to dict with tform keys
    spine_tforms = {k: v for k, v in zip(plan_model.spine_levels, spine_aff[:num_bones])}

    # build metrics dict
    per_screw = {}
    for screw in placed_screws:
        per_screw[screw.name] = {
            'type': screw.type,
            'shaft_len': float(screw.shaft_len),
            'shaft_rad': float(screw.shaft_rad),
            **fit_info[screw.name],
        }

    metrics = {
        'n_screws_planned': num_screws,
        'n_screws_detected': num_screws,
        'n_metal_points': n_metal_pts,
        'volume_per_screw_mm3': vol_per_screw,
        'init_alignment_error': float(init_err),
        'unconverged': unconverged,
        'per_screw': per_screw,
        'shaft_coverage_failed': shaft_coverage_failed,
        'optimization': {
            'spine_cost': float(spine_cost),
            'screw_cost': float(opt_err),
        },
        'timings': timings,
    }

    return spine_tforms, metrics


# ---------------------------------------------------------------------------
# Metal thresholding and point extraction
# ---------------------------------------------------------------------------

def extract_metal_points(postop_ct: nib.Nifti1Image,
                         threshold: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract metal point cloud from post-op CT.

    Parameters
    ----------
    postop_ct : nib.Nifti1Image
        Full post-op CT volume.
    threshold : int or None
        HU threshold for metal (auto-computed via Otsu if None).

    Returns
    -------
    pts : np.ndarray
        (N, 3) RAS metal point coordinates.
    density : np.ndarray
        (N,) per-point smoothed density.
    pitch : np.ndarray
        (3,) voxel pitch from affine diagonal.
    """
    affine = postop_ct.affine
    pitch = np.abs(np.diag(affine[:3, :3]))

    data = postop_ct.get_fdata()
    if threshold is None:
        threshold = compute_metal_threshold(data)

    # Threshold for metal
    metal_mask = data > threshold
    # ~1mm physical closure. Postop is resampled to (≤0.8, ≤0.8, 0.625) mm
    # upstream, so mean(pitch) ∈ [~0.5, ~0.74] → 1 or 2 iters.
    closing_iters = round(1.0 / np.mean(pitch))
    if closing_iters > 0:
        metal_mask = ndimage.binary_closing(metal_mask, iterations=closing_iters)

    # Convert to pointcloud, estimate density
    pts = convert_to_points(metal_mask, affine)
    n_raw = len(pts)
    sigma = 1 / np.abs(pitch)
    smooth1 = ndimage.gaussian_filter(metal_mask.astype(np.float32), sigma)
    density = smooth1[metal_mask]

    # cluster points to remove small isolated regions; eps captures face-
    # adjacent voxels along the coarsest axis. Post-resample max(pitch) is
    # in [0.625, 0.8], giving eps in [0.75, 0.96].
    eps = 1.2 * np.max(pitch)

    # Post-resample voxel_vol is bounded but varies ~10× across the in-plane
    # range; keep min_samples adaptive so sparse shafts at coarse in-plane
    # don't lose their cluster cores.
    voxel_vol = float(np.prod(pitch))
    min_samples = int(min(7, round(3.0 / voxel_vol**0.35)))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    # discard points that are not in a cluster
    keep = dbscan.labels_ != -1
    n_clustered = int(np.sum(keep))
    n_clusters = len(set(dbscan.labels_) - {-1})
    log.info('Metal extraction: %d voxels > %d HU, DBSCAN(eps=%.2f, min_samples=%d) '
             'kept %d/%d pts in %d clusters',
             n_raw, threshold, eps, min_samples, n_clustered, n_raw, n_clusters)
    pts = pts[keep]
    density = density[keep]

    return pts, density, pitch


# ---------------------------------------------------------------------------
# HU-weighted refinement helpers
# ---------------------------------------------------------------------------

def _sample_near_screw(screw: Screw, data: np.ndarray, affine: np.ndarray,
                       threshold: int, margin: float = 3.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sample voxels near the screw surface with sigmoid HU weights.

    Returns world-space points and per-point weights in [0, 1].
    High-HU voxels get weight ~1; partial-volume voxels get proportional
    weight; bone/air gets ~0.
    """
    v, f = screw.build_mesh(planned=False)
    inv_aff = np.linalg.inv(affine)
    pitch = np.abs(np.diag(affine[:3, :3]))

    # Bounding box in voxel space
    v_vox = (inv_aff @ np.c_[v, np.ones(len(v))].T).T[:, :3]
    margin_vox = margin / pitch
    lo = np.maximum(np.floor(v_vox.min(0) - margin_vox).astype(int), 0)
    hi = np.minimum(np.ceil(v_vox.max(0) + margin_vox).astype(int),
                    np.array(data.shape) - 1)

    # All voxel centers in bbox → world coords
    ri, ai, si = np.mgrid[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]
    vox = np.column_stack([ri.ravel(), ai.ravel(), si.ravel()])
    pts = (affine @ np.c_[vox.astype(float), np.ones(len(vox))].T).T[:, :3]

    # Distance to screw mesh surface
    d2 = igl.point_mesh_squared_distance(pts, v.astype(np.float64), f)[0]
    near = d2 < margin ** 2

    pts_near = pts[near]
    hu = data[vox[near, 0], vox[near, 1], vox[near, 2]]

    # Sigmoid weight on HU: smooth transition ~400 HU wide centered at threshold
    k = 0.01
    weights = 1.0 / (1.0 + np.exp(-k * (hu - threshold)))

    # Drop near-zero-weight voxels — they don't contribute to the weighted
    # cost but slow down every least_squares iteration
    keep = weights > 0.01
    return pts_near[keep], weights[keep]


def _has_close_head_neighbor(screw: Screw, others: list[Screw],
                             margin: float = 2.0) -> bool:
    """Check if any other screw's head sampling zone overlaps this screw's.

    The sampling zone for each head extends head_len/2 + head_rad from the
    entry point (sphere/cylinder extent), plus ``margin`` on each side (the
    ``_sample_near_screw`` search margin).  If the sum of two screws' zones
    exceeds the distance between their entries, their sampled voxels overlap.
    """
    joint_offset = 4 # hard-coded 4mm depth of ball joint in head
    if screw.head_len <= 0:
        return False

    r_i = 0.75 * np.sqrt((screw.head_len - joint_offset)**2 + screw.head_rad**2)
    for other in others:
        if other is screw or other.head_len <= 0:
            continue
        r_j = 0.75 * np.sqrt((other.head_len - joint_offset)**2 + other.head_rad**2)
        d = float(np.linalg.norm(screw.detected_entry - other.detected_entry))
        if d < r_i + r_j + margin:
            return True
    return False


def _exclude_neighbor_points(screw: Screw, pts: np.ndarray, weights: np.ndarray,
                             others: list[Screw], exclude_margin: float = 3.0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Remove points that fall within another screw's mesh.

    For each nearby screw (pre-filtered by entry distance), build its posed
    mesh and exclude sampled points closer than ``exclude_margin`` mm.
    Returns the filtered (pts, weights) with contaminating points removed.
    """
    if len(pts) == 0:
        return pts, weights

    joint_offset = 4  # same as _has_close_head_neighbor
    sample_margin = 3.0  # _sample_near_screw default margin
    total_margin = sample_margin + exclude_margin  # 6mm: sample zone + exclusion zone

    # Pre-compute this screw's bounding sphere radius
    r_i = 0.0
    if screw.head_len > 0:
        r_i = 0.75 * np.sqrt((screw.head_len - joint_offset)**2 + screw.head_rad**2)

    mask = np.ones(len(pts), dtype=bool)

    for other in others:
        if other is screw:
            continue

        # Sphere-overlap pre-filter: can this screw's sampled points reach the other?
        r_j = 0.0
        if other.head_len > 0:
            r_j = 0.75 * np.sqrt((other.head_len - joint_offset)**2 + other.head_rad**2)
        d = float(np.linalg.norm(screw.detected_entry - other.detected_entry))
        if d >= r_i + r_j + total_margin:
            continue

        # Expensive path: build mesh and query distance
        v, f = other.build_mesh(planned=False)
        d2 = igl.point_mesh_squared_distance(pts, v.astype(np.float64), f)[0]
        inside = d2 < exclude_margin ** 2
        n_excluded = int(np.sum(inside & mask))
        if n_excluded > 0:
            log.debug('excluded %d points from %s near %s',
                      n_excluded, screw.name, other.name)
            mask &= ~inside

    return pts[mask], weights[mask]


def _shaft_coverage(screw: Screw, pts: np.ndarray, weights: np.ndarray,
                    weight_thresh: float = 0.5) -> float:
    """Fraction of shaft length covered by high-HU metal within sampling margin.

    Divides the shaft into ~2mm bins and checks what fraction contain at least
    one high-weight voxel.  Returns 0.0 if no high-HU points exist.
    """
    if len(pts) == 0:
        return 0.0
    high_hu = weights > weight_thresh
    if not np.any(high_hu):
        return 0.0
    axis = screw.axis(planned=False)
    proj = (pts[high_hu] - screw.detected_entry) @ axis / screw.shaft_len
    n_bins = max(5, int(screw.shaft_len / 2))  # ~2mm bins
    counts, _ = np.histogram(proj, bins=np.linspace(0, 1, n_bins + 1))
    return float(np.mean(counts > 0))


# ---------------------------------------------------------------------------
# Point partitioning for individual screw fitting
# ---------------------------------------------------------------------------

def partition_points(screws: list[Screw], pts: np.ndarray) -> list[np.ndarray]:
    """Soft-assign metal points to screws based on mesh distance.

    Each point goes to screws where its normalized assignment weight > 0.6
    and raw mesh distance < 2 * d_thresh.  Points inside a screw mesh
    (determined by winding number) get distance 0.
    """
    d_thresh = 2.  # distance threshold for partitioning points
    nS, nP = len(screws), len(pts)

    # Build KDTree once for spatial pre-filtering
    pt_tree = KDTree(pts)

    # Compute per-screw search radius from shaft geometry + margin
    centroids = np.array([(s.detected_entry + s.detected_tip) / 2 for s in screws])
    half_lengths = np.array([np.linalg.norm(s.detected_entry - s.detected_tip) / 2 for s in screws])
    search_radii = half_lengths + d_thresh * 4  # generous margin

    distance = 100.0 * np.ones((nS, nP))
    for ss, screw in enumerate(screws):
        # Only test points near this screw
        nearby_idx = np.array(pt_tree.query_ball_point(centroids[ss], search_radii[ss]), dtype=int)
        if len(nearby_idx) == 0:
            continue
        nearby_pts = pts[nearby_idx]

        v, f = screw.build_mesh(planned=False)

        # get distance to screw mesh for nearby points
        d_nearby = np.sqrt(igl.point_mesh_squared_distance(nearby_pts, v, f)[0])

        # only compute expensive winding number for points close to mesh surface
        close_mask = d_nearby < d_thresh * 2
        if np.any(close_mask):
            w = igl.winding_number(v, f, nearby_pts[close_mask])
            inside = (np.round(w).astype(int) % 2).astype(bool)
            d_nearby[np.where(close_mask)[0][inside]] = 0

        distance[ss, nearby_idx] = d_nearby

    weight = np.exp(-distance / d_thresh)
    w_sum = np.sum(weight, axis=0, keepdims=True)
    w_sum[w_sum == 0] = 1  # avoid division by zero for far-away points
    weight /= w_sum

    point_clouds = []
    for ss in range(nS):
        mask = (weight[ss] > 0.6) & (distance[ss] < (d_thresh * 2))
        point_clouds.append(pts[mask])

    return point_clouds
