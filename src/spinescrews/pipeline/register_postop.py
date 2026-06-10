import os
import sys
from os.path import join, expanduser, isfile, isdir
import logging
from time import time

import igl
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from scipy import ndimage
from dipy.align.transforms import RigidTransform3D
from dipy.align.imaffine import (AffineRegistration, AffineMap,
                                  MutualInformationMetric, VerbosityLevels)
from dipy.core.optimize import Optimizer

from bg3dtools.transforms_unified import inverse

from spinescrews.tools.nifti_utils import compute_metal_threshold
from spinescrews.tools.screw_models import parse_preop_plan, sanity_check_plan
from spinescrews.tools.vertebrae import Vertebra
from spinescrews.tools.screw_detection import detect_screws
from spinescrews.tools.articulated_spine_registration import align_spine_to_CT, _build_artifact_mask_fast
from spinescrews.tools.paths import (preop_dir, preop_level_dir,
                         orient_dir, orient_level_dir,
                         detection_dir, registration_dir, registration_level_dir,
                         step_complete, write_summary, read_summary, timed)

import matplotlib
matplotlib.use('Agg')

log = logging.getLogger(__name__)


_GRADIENT_METHODS = {'L-BFGS-B', 'CG', 'BFGS', 'Newton-CG'}


class DiagnosticAffineRegistration(AffineRegistration):
    """AffineRegistration subclass that captures per-level optimizer diagnostics.

    After optimize() returns, ``self.level_diagnostics`` contains a list of
    dicts (one per resolution level, coarse-to-fine) with keys:
        level, factor, sigma, max_iter, nit, nfev, fopt, message, method
    """

    def optimize(self, static, moving, transform, params0,
                 static_grid2world=None, moving_grid2world=None,
                 starting_affine=None, ret_metric=False,
                 static_mask=None, moving_mask=None):
        self.level_diagnostics = []

        self._init_optimizer(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine,
                             static_mask, moving_mask)
        del starting_affine
        del static_mask
        del moving_mask

        original_static_shape = self.static_ss.get_image(0).shape
        original_static_grid2world = self.static_ss.get_affine(0)
        original_moving_shape = self.moving_ss.get_image(0).shape
        original_moving_grid2world = self.moving_ss.get_affine(0)
        affine_map = AffineMap(None,
                               original_static_shape,
                               original_static_grid2world,
                               original_moving_shape,
                               original_moving_grid2world)

        use_gradient = self.method in _GRADIENT_METHODS

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[-1 - level]

            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2world = self.static_ss.get_affine(level)
            current_affine_map = AffineMap(None,
                                           current_static_shape,
                                           current_static_grid2world,
                                           original_static_shape,
                                           original_static_grid2world)
            current_static = current_affine_map.transform(smooth_static)
            current_static_mask = None
            if self.static_mask is not None:
                current_static_mask = current_affine_map.transform(
                    self.static_mask, interpolation="nearest").astype(np.int32)

            current_moving = self.moving_ss.get_image(level)

            self.metric.setup(transform, current_static, current_moving,
                              current_static_grid2world,
                              original_moving_grid2world, self.starting_affine,
                              current_static_mask, self.moving_mask)

            if self.options is None:
                self.options = {'gtol': 1e-4, 'disp': False}

            if self.method == 'L-BFGS-B':
                self.options['maxfun'] = max_iter
            else:
                self.options['maxiter'] = max_iter

            if use_gradient:
                opt = Optimizer(self.metric.distance_and_gradient,
                                self.params0,
                                method=self.method, jac=True,
                                options=self.options)
            else:
                opt = Optimizer(self.metric.distance,
                                self.params0,
                                method=self.method, jac=False,
                                options=self.options)

            self.level_diagnostics.append({
                'level': level,
                'factor': int(self.factors[self.levels - 1 - level]),
                'sigma': float(self.sigmas[self.levels - 1 - level]),
                'max_iter': max_iter,
                'nit': int(opt.nit),
                'nfev': int(opt.nfev),
                'fopt': float(opt.fopt),
                'message': str(opt.message),
                'method': self.method,
            })

            params = opt.xopt
            T = self.transform.param_to_matrix(params)
            self.starting_affine = T.dot(self.starting_affine)
            self.params0 = self.transform.get_identity_parameters()

        affine_map.set_affine(self.starting_affine)
        if ret_metric:
            return affine_map, opt.xopt, opt.fopt
        return affine_map


class Registrar:

    def __init__(self, config):
        """Initialize Registrar with pipeline config; sets up directories and empty containers."""
        self.config = config
        self.template_dir = expanduser(config.template_dir)
        self.working_dir = expanduser(config.specimen_dir)
        self.analysis_dir = str(join(self.working_dir, config.output_dir))

        self.preop_verts = {}
        self.screws = []

        # postop CT data (loaded by import_from_disk)
        self.postop_img = nib.Nifti1Image(np.zeros([]), np.eye(4))
        self.postop_labels = nib.Nifti1Image(np.zeros([]), np.eye(4))
        self.metal_threshold = config.metal_mask_threshold
        self.artifact_mask = None

    def import_from_disk(self):
        """Load preop state from step 02/04 outputs + postop CT from specimen dir.

        Validates that step 04 completed (hard error if not).
        Reconstructs preop_verts from disk — same pattern as compute_accuracy.py.
        """
        # Validate preop steps completed
        o_dir = orient_dir(self.analysis_dir)
        if not step_complete(o_dir):
            raise RuntimeError(
                'Step 04 (orient) not complete — run align_preop.py first.\n'
                '  Expected: %s' % join(o_dir, 'summary.json'))

        # Load postop CT
        postop_vol = join(self.working_dir, 'postop.nii.gz')
        postop_seg = join(self.working_dir, 'postop_seg.nii.gz')
        if not isfile(postop_vol):
            raise FileNotFoundError(
                'missing postop CT: %s — run dcm2niix first' % postop_vol)

        self.postop_img = nib.as_closest_canonical(nib.load(postop_vol), True)
        # Auto-compute metal threshold from postop data, or use config override
        if self.config.metal_mask_threshold is not None:
            self.metal_threshold = self.config.metal_mask_threshold
        else:
            self.metal_threshold = compute_metal_threshold(self.postop_img.get_fdata())
            log.info('Auto-computed metal mask threshold: %d HU', self.metal_threshold)
        data = self.postop_img.get_fdata()
        is_metal = data >= self.metal_threshold
        # dilate metal mask by ~2mm so the exclusion zone covers partial-volume
        # voxels at the metal boundary
        pitch = np.abs(np.diag(self.postop_img.affine[:3, :3]))
        dilate_iters = max(1, int(round(2.0 / pitch.min())))
        is_metal = ndimage.binary_dilation(is_metal, iterations=dilate_iters)
        log.info('Metal mask: %d iters dilation (%.1fmm pitch, 2mm target)',
                 dilate_iters, pitch.min())
        metal_mask = (~is_metal).astype(np.uint8)
        self.postop_labels = nib.Nifti1Image(metal_mask, self.postop_img.affine)
        if isfile(postop_seg):
            self.postop_labels = nib.load(postop_seg)

        # Parse screw plan
        plan_file = join(self.working_dir, 'preop_plan.csv')
        if not isfile(plan_file):
            raise FileNotFoundError(
                'missing CSV screw plan: %s' % plan_file)
        _, self.screws = parse_preop_plan(plan_file)
        sanity_check_plan(self.screws)

        # Discover levels from step 02 directories on disk
        step02_dir = preop_dir(self.analysis_dir)
        level_names = []
        for entry in sorted(os.listdir(step02_dir)):
            entry_dir = join(step02_dir, entry)
            if isdir(entry_dir) and isfile(join(entry_dir, 'preop_affine.npy')):
                level_names.append(entry)

        log.info('Found %d levels from step 02: %s', len(level_names), ', '.join(level_names))

        # Reconstruct preop_verts from step 04 outputs (refined affines + volumes)
        for level in level_names:
            olvl = orient_level_dir(self.analysis_dir, level)
            preop_ldir = preop_level_dir(self.analysis_dir, level)

            vert = Vertebra(level)
            vert.affine = np.load(join(olvl, 'preop_affine-refined.npy'))
            vert.img_normalized = nib.load(join(olvl, 'preop.nii.gz'))
            vert.seg_normalized = nib.load(join(olvl, 'preop_seg.nii.gz'))

            # Load gen1 mesh from step 02, transform to refined frame
            v_gen1, f_gen1 = igl.read_triangle_mesh(join(preop_ldir, 'preop_gen1.ply'))
            orig_aff = np.load(join(preop_ldir, 'preop_affine.npy'))
            delta = np.linalg.inv(vert.affine) @ orig_aff
            v_gen1 = (delta[:3, :3] @ v_gen1.T).T + delta[:3, 3]
            vert.set_mesh(v_gen1, f_gen1)

            self.preop_verts[level] = vert

        log.info('Loaded %d preop vertebrae from disk', len(self.preop_verts))

    def detect_screws(self):
        """Detect screws in post-op CT. Outputs to 05_detection/."""
        det_dir = detection_dir(self.analysis_dir)
        os.makedirs(det_dir, exist_ok=True)
        spine_reg_file = join(det_dir, 'spine_tforms_initial.npz')

        if isfile(spine_reg_file):
            for screw in self.screws:
                file = join(det_dir, screw.name + "_screw.yml")
                screw.load_from_yaml(file)
            log.info('  loaded %d screws from pre-registered files' % len(self.screws))
            return {}  # no fresh metrics

        else:
            log.info('Performing screw detection')

            preop_aff = {name: vertebra.affine for name, vertebra in self.preop_verts.items()}
            spine_tforms, metrics = detect_screws(self.postop_img, self.screws, preop_aff,
                                                   threshold=self.config.screw_detect_threshold,
                                                   n_jobs=self.config.n_jobs,
                                                   analysis_dir=self.analysis_dir)

            log.debug('Saving screw YAMLs to %s' % det_dir)
            for screw in self.screws:
                screw_file = join(det_dir, screw.name + "_screw.yml")
                screw.save_to_yaml(screw_file)

            log.debug('Saving initial spine registrations')
            np.savez(spine_reg_file, **spine_tforms)

            return metrics

    def pointcloud_registration(self):
        """Run articulated ICP. Outputs to 06_registration/."""
        log.info('Running pointcloud registrations')
        reg_dir = registration_dir(self.analysis_dir)
        os.makedirs(reg_dir, exist_ok=True)
        output_file = join(reg_dir, 'spine_tforms_icp.npz')

        if not isfile(output_file):
            det_dir = detection_dir(self.analysis_dir)
            initial_tforms = dict(np.load(join(det_dir, 'spine_tforms_initial.npz')))
            icp_affs, icp_metrics, self.artifact_mask = align_spine_to_CT(
                self.preop_verts, self.postop_img, self.screws, initial_tforms,
                iso_res=self.config.icp_iso_res,
                initial_radius=self.config.icp_initial_radius,
                ratio_thresh=self.config.icp_ratio_thresh,
                output_dir=reg_dir)
            np.savez(output_file, **icp_affs)
            return icp_metrics
        else:
            self.artifact_mask = _build_artifact_mask_fast(self.postop_img, self.screws)
            return {}

    def volumetric_registration(self):
        """Run mutual-information refinement. Outputs to 06_registration/{LEVEL}/."""
        log.info('Running volumetric registrations')
        file_prefix = 'postop-reg'
        debug = self.config.debug
        screw_levels = list(set([screw.level for screw in self.screws if screw.level != 'skip']))
        preop_verts = {name: self.preop_verts[name] for name in screw_levels}

        # Bake artifact mask into postop_labels so resampled per-vertebra
        # volumes exclude streak artifacts from MI optimization
        if self.artifact_mask is not None:
            labels_data = self.postop_labels.get_fdata().copy()
            labels_data[self.artifact_mask] = 0
            self.postop_labels = nib.Nifti1Image(labels_data.astype(np.uint8),
                                                  self.postop_labels.affine)
            n_zeroed = int(self.artifact_mask.sum())
            log.info('Artifact mask: zeroed %d voxels in postop_labels', n_zeroed)

        reg_dir = registration_dir(self.analysis_dir)
        icp_affs = dict(np.load(join(reg_dir, 'spine_tforms_icp.npz')))

        todo_levels = {}
        per_level = {}
        for level_name in screw_levels:
            level_dir = registration_level_dir(self.analysis_dir, level_name)
            os.makedirs(level_dir, exist_ok=True)
            if not isfile(join(level_dir, file_prefix + '.nii.gz')):
                bone = Vertebra(level_name, icp_affs[level_name])
                bone.import_volumes(self.postop_img, self.postop_labels)
                todo_levels[level_name] = bone
                if debug:
                    bone.save(level_dir, 'postop-load')

        if len(todo_levels) > 0:
            todo_names = list(todo_levels.keys())
            cfg = self.config
            optimized_results = Parallel(n_jobs=cfg.mi_n_jobs, prefer='threads')(
                delayed(self.optimize_registration)(preop_verts[level_name],
                                                    todo_levels[level_name],
                                                    preop_dilate=cfg.mi_preop_dilate,
                                                    postop_dilate=cfg.mi_postop_dilate,
                                                    mi_quality_fail=cfg.mi_quality_fail,
                                                    mi_quality_warn=cfg.mi_quality_warn,
                                                    metal_threshold=self.metal_threshold,
                                                    mi_method=cfg.mi_method)
                                                    for level_name in todo_names)

            postop_volumes = {}
            for level_name, (tform, diag) in zip(todo_names, optimized_results):
                level_dir = registration_level_dir(self.analysis_dir, level_name)
                per_level[level_name] = diag
                # re-import data with optimized registration, save out
                bone = Vertebra(level_name, tform)
                bone.import_volumes(self.postop_img, self.postop_labels)
                bone.save(level_dir, file_prefix,
                         save_volume=not self.config.no_patches)
                postop_volumes[level_name] = bone.img_normalized

                # save preop binary segmentation for QC overlay in 3DSlicer
                preop_seg = preop_verts[level_name].seg_normalized
                if preop_seg is not None:
                    nib.save(preop_seg, join(level_dir, 'preop_seg.nii.gz'))

            self._postop_volumes = postop_volumes

        num_loaded = len(screw_levels) - len(todo_levels)
        if num_loaded > 0:
            log.debug('  loaded %d vertebrae from file' % num_loaded)

        # build warnings/failures lists
        warnings = []
        failures = []
        for level, diag in per_level.items():
            mi = diag['fopt']
            if mi >= self.config.mi_quality_fail:
                failures.append('%s: MI=%.2f (above %.2f threshold)' % (level, mi, self.config.mi_quality_fail))
            elif mi > self.config.mi_quality_warn:
                warnings.append('%s: MI=%.2f (above %.2f threshold)' % (level, mi, self.config.mi_quality_warn))

        return {
            'per_level': per_level,
            'warnings': warnings,
            'failures': failures,
        }

    @staticmethod
    def optimize_registration(preop_vert: Vertebra, postop_vert: Vertebra,
                              preop_dilate=4, postop_dilate=-2,
                              mi_quality_fail=-0.15, mi_quality_warn=-0.25,
                              init_affine=None,
                              metal_threshold=None,
                              mi_method='L-BFGS-B'):
        """
        Returns (final_tform, diagnostics) tuple.

        Two-level MI: 3× downsample coarse search (1.5mm effective, σ=4
        Gaussian) then full-resolution polish (0.5mm, no smoothing).
        Input volumes are always 0.5mm isotropic from step 04.

        diagnostics is a dict with keys: fopt, elapsed_s, levels (list of
        per-resolution-level dicts with nit, nfev, fopt, message, etc.).
        """
        preop_img = preop_vert.img_normalized.get_fdata().astype(np.float64)
        preop_seg = preop_vert.seg_normalized.get_fdata().astype(bool)
        postop_img = postop_vert.img_normalized.get_fdata().astype(np.float64)
        postop_seg = postop_vert.seg_normalized.get_fdata().astype(bool)

        if preop_dilate > 0:
            preop_seg = ndimage.binary_dilation(preop_seg, iterations=preop_dilate)
        elif preop_dilate < 0:
            preop_seg = ndimage.binary_erosion(preop_seg, iterations=-preop_dilate, border_value=1)
        # compute adaptive threshold for postop image (before marker exclusion)
        low, high = np.percentile(preop_img[preop_seg], [2, 98])
        # mask out any metal markers in preop image
        if metal_threshold is not None:
            marker_thresh = (metal_threshold - high) * 0.85 + high
        else:
            marker_thresh = 2000  # legacy fallback
        marker_mask = preop_img > marker_thresh
        marker_mask = ndimage.binary_dilation(marker_mask, iterations=2)
        preop_seg = preop_seg & ~marker_mask

        if postop_dilate >= 0:
            raise ValueError('postop_dilate must be negative, got %d' % postop_dilate)
        postop_screws = ~postop_seg
        postop_halo = ndimage.binary_dilation(postop_screws, iterations=6)
        # apply adaptive threshold to postop mask
        postop_screws |= postop_halo & (postop_img < low)
        postop_screws |= postop_halo & (postop_img > high)
        postop_mask = ~ndimage.binary_dilation(postop_screws, iterations=-postop_dilate)

        # Clip both images to [0, p99.9] of the preop masked range so that
        # dipy's per-image min-max normalization maps the same HU span to
        # [0, 1] on both histogram axes.  This prevents air (< 0 HU) and
        # metal-adjacent outliers from stretching the bin range and pushing
        # cortical bone into a single clamped bin.
        clip_high = float(np.percentile(preop_img[preop_seg], 99.9))
        preop_img = np.clip(preop_img, 0, clip_high)
        postop_img = np.clip(postop_img, 0, clip_high)

        # Two levels: coarse search (3× downsample) + full-resolution polish
        affreg = DiagnosticAffineRegistration(metric=MutualInformationMetric(nbins=64),
                                              factors=[3, 1],
                                              sigmas=[4, 0],
                                              level_iters=[1000, 1000],
                                              verbosity=VerbosityLevels.NONE,
                                              method=mi_method)

        t0 = time()
        adjustment_tform, _, fopt = affreg.optimize(postop_img, preop_img, RigidTransform3D(),
                                                    params0=None, starting_affine=init_affine,
                                                    static_mask=postop_mask.astype(np.float64),
                                                    moving_mask=preop_seg.astype(np.float64),
                                                    static_grid2world=postop_vert.img_normalized.affine.astype(np.float64),
                                                    moving_grid2world=preop_vert.img_normalized.affine.astype(np.float64),
                                                    ret_metric=True)
        elapsed = time() - t0

        # Summarize per-level iteration counts for log message
        level_summary = ', '.join(
            'L%d: %d/%d iter' % (d['level'], d['nit'], d['max_iter'])
            for d in affreg.level_diagnostics)

        if fopt >= mi_quality_fail:
            log.error('%s: MI=%.3f (FAIL) in %.1fs [%s]', preop_vert.name, fopt, elapsed, level_summary)
        elif fopt > mi_quality_warn:
            log.warning('%s: MI=%.3f (weak) in %.1fs [%s]', preop_vert.name, fopt, elapsed, level_summary)
        else:
            log.info('%s: MI=%.3f in %.1fs [%s]', preop_vert.name, fopt, elapsed, level_summary)

        diagnostics = {
            'fopt': float(fopt),
            'elapsed_s': round(elapsed, 1),
            'method': mi_method,
            'levels': affreg.level_diagnostics,
        }

        final_tform = postop_vert.affine @ inverse(adjustment_tform.affine)
        return final_tform, diagnostics


def _log_elapsed(label, elapsed):
    """Log elapsed time in seconds or minutes."""
    if elapsed < 60:
        log.info('*** %s took %.2f seconds' % (label, elapsed))
    else:
        log.info('*** %s took %.2f minutes' % (label, elapsed / 60))


def _run_detection(study, analysis_dir, config):
    """Step 05: screw detection + MIP figures."""
    step_dir = detection_dir(analysis_dir)
    if step_complete(step_dir):
        log.info('*** Step 05 (detection) complete, skipping')
        study.detect_screws()  # reload screw YAMLs into memory
        return

    timings = {}
    t = time()
    detection_metrics = study.detect_screws()
    elapsed = time() - t
    _log_elapsed('Screw detection', elapsed)

    # plan-vs-detected needs screw YAMLs on disk (written by study.detect_screws)
    with timed('detection_plan_vs_detected', timings):
        from spinescrews.figures.detection_plan_vs_detected import generate_detection_plan_vs_detected
        generate_detection_plan_vs_detected(analysis_dir, study.working_dir,
                                            threshold=config.screw_detect_threshold)

    # Check shaft coverage QC — warn after figures so they exist for debugging
    if detection_metrics:
        shaft_failed = detection_metrics.get('shaft_coverage_failed', [])
        if shaft_failed:
            lines = ['Low shaft metal coverage on %d screws (results may be unreliable):'
                     % len(shaft_failed)]
            for name, cov in shaft_failed:
                lines.append('  %s: %.0f%%' % (name, 100 * cov))
            lines.append('Consider setting screw_type=skip in preop_plan.csv for unplaced screws.')
            log.warning('\n'.join(lines))

        det_timings = detection_metrics.get('timings', {})
        det_timings.update(timings)
        detection_metrics['timings'] = det_timings
        detection_metrics['elapsed_s'] = round(elapsed, 1)
        write_summary(step_dir, detection_metrics)
    else:
        write_summary(step_dir, {
            'n_screws_planned': len(study.screws),
            'elapsed_s': round(elapsed, 1),
            'timings': timings,
            'note': 'loaded from existing files',
        })


def _run_registration(study, analysis_dir, config):
    """Step 06: ICP + volumetric registration + CT figures."""
    step_dir = registration_dir(analysis_dir)
    if step_complete(step_dir):
        log.info('*** Step 06 (registration) complete, skipping')
        study.pointcloud_registration()
        study.volumetric_registration()
        return

    timings = {}
    t = time()
    with timed('pointcloud_registration', timings):
        icp_metrics = study.pointcloud_registration()
    _log_elapsed('Pointcloud registration', timings.get('pointcloud_registration', 0))

    with timed('volumetric_registration', timings):
        mi_metrics = study.volumetric_registration()
    _log_elapsed('Volumetric registration', timings.get('volumetric_registration', 0))

    with timed('CT_figures', timings):
        from spinescrews.figures.CT_visualization import generate_ct_figures
        generate_ct_figures(analysis_dir, getattr(study, '_postop_volumes', None))

    elapsed = time() - t

    # merge ICP sub-timings into top-level timings
    if icp_metrics:
        icp_sub = icp_metrics.pop('timings', {})
        timings.update(icp_sub)
    reg_summary = {'elapsed_s': round(elapsed, 1), 'timings': timings}
    if icp_metrics:
        reg_summary['icp'] = icp_metrics
    if mi_metrics:
        reg_summary['volumetric'] = mi_metrics
    write_summary(step_dir, reg_summary)


def _postop_summary(analysis_dir):
    """Read step 05-06 summaries and report any quality warnings."""
    warnings = []

    # Step 05 — detection
    det_dir_ = detection_dir(analysis_dir)
    if not step_complete(det_dir_):
        warnings.append('Step 05 (detection) did not complete')
    else:
        s = read_summary(det_dir_)
        n_plan = s.get('n_screws_planned', 0)
        n_det = s.get('n_screws_detected', n_plan)
        if n_det < n_plan:
            warnings.append('Only %d/%d screws detected' % (n_det, n_plan))
        for name, m in s.get('per_screw', {}).items():
            ir = m.get('inlier_ratio', 1.0)
            sc = m.get('shaft_coverage', 1.0)
            if sc < 0.80:
                warnings.append('%s: screw fit %.0f%% shaft coverage' % (name, sc * 100))

    # Step 06 — registration
    reg_dir = registration_dir(analysis_dir)
    if not step_complete(reg_dir):
        warnings.append('Step 06 (registration) did not complete')
    else:
        s = read_summary(reg_dir)
        vol = s.get('volumetric', {})
        for w in vol.get('warnings', []):
            warnings.append('MI warning: %s' % w)
        for f in vol.get('failures', []):
            warnings.append('MI failure: %s' % f)

    # Report
    if warnings:
        log.info('=' * 40)
        log.info('=== POSTOP WARNINGS ===')
        for w in warnings:
            log.info('  - %s' % w)
        log.info('%d warning(s) found -- review before trusting results' % len(warnings))
        log.info('=' * 40)
    else:
        log.info('=== Postop steps completed cleanly (steps 05-06) ===')


def run_postop(config):
    """Run postop registration steps 05-06 from a config object.

    Callable by the orchestrator (align_vertebrae.py) without CLI arg parsing.
    Returns early if step 06 is already complete.
    """
    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)

    if step_complete(registration_dir(analysis_dir)):
        log.info('Registration steps already complete — skipping')
        return

    study = Registrar(config)
    study.import_from_disk()
    _run_detection(study, analysis_dir, config)
    _run_registration(study, analysis_dir, config)
    _postop_summary(analysis_dir)


def main():
    """CLI entry point for postop registration (steps 05-06). Called by spinescrews-postop console script."""
    t0 = time()
    import argparse
    from spinescrews.tools.config import (load_config, save_resolved_config,
                                          add_common_pipeline_args, overrides_from_args)

    parser = argparse.ArgumentParser(
        description='Postop registration (steps 05-06): metal screw detection, articulated spine '
                    'ICP + D-PMP refit, and per-level mutual-information refinement. Requires '
                    'postop.nii.gz, preop_plan.csv, and a completed preop alignment (run '
                    'spinescrews-preop first). Outputs go to <specimen_dir>/analysis/.')
    parser.add_argument('specimen_dir',
                        help='Specimen directory containing postop.nii.gz / preop_plan.csv with '
                             'preop alignment already done; results go to <specimen_dir>/analysis/.')
    add_common_pipeline_args(parser)
    parser.add_argument('--no-patches', action='store_true', default=None,
                        help='Skip writing postop-reg.nii.gz volumes (saves ~59 MB/level)')
    parser.add_argument('--mi-method', type=str, default=None,
                        choices=['L-BFGS-B', 'Powell', 'Nelder-Mead'],
                        help='Optimizer for MI registration (default: L-BFGS-B)')
    args = parser.parse_args()

    overrides = overrides_from_args(args)
    if args.no_patches is not None:
        overrides['no_patches'] = args.no_patches
    if args.mi_method is not None:
        overrides['mi_method'] = args.mi_method

    config = load_config(args.specimen_dir, overrides=overrides)
    save_resolved_config(config)

    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)

    logfile = join(analysis_dir, 'postop.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])

    log.info('*' * (35 + len(data_dir)))
    log.info('**  Postop registration for %s  **' % data_dir)
    log.info('*' * (35 + len(data_dir)))

    run_postop(config)
    log.info('*** Total postop time: %.2f minutes' % ((time() - t0) / 60))


if __name__ == '__main__':
    main()
