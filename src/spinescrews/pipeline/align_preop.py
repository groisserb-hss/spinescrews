import os
import sys
from os.path import join, basename, expanduser, isdir, isfile
import logging
from time import time

import igl
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from scipy import sparse
from scipy.sparse.linalg import splu

from bg3dtools.transforms_unified import rigid_reg, R_to_twist, twist_to_R, aff_to_rel_params
from bg3dtools.mesh.registration import surface_match
from bg3dtools.mesh.barycentric import bc2sparse
from spectral_match.pipeline import Mesh, FunctionalMapper
from spectral_match.pipeline import pmf_match

from spinescrews.tools.screw_models import parse_preop_plan, sanity_check_plan
from spinescrews.tools.vertebrae import Vertebra
from spinescrews.tools.correspondence_tools import load_vertebral_template
from spinescrews.tools import seg_val, val_seg, possible_levels
from spinescrews.tools.paths import (segmentation_dir, segmentation_file,
                         preop_dir, preop_level_dir,
                         correspondence_dir, correspondence_level_dir,
                         orient_dir, orient_level_dir,
                         step_complete, write_summary, read_summary)

import matplotlib
matplotlib.use('Agg')

log = logging.getLogger(__name__)


def _whittaker_smooth(data, w, lam, d=2):
    """Whittaker smoother: penalised least-squares with fidelity weights.

    Minimises  sum_i w_i*(y_i - z_i)^2  +  lam * sum (D^d z)^2
    """
    n = data.shape[0]
    # Build d-th order difference matrix by repeated first-differencing
    D = sparse.eye(n, format='csc')
    for _ in range(d):
        m = D.shape[0]
        D = D[1:] - D[:-1]  # first difference along rows
    W = sparse.diags(w, format='csc')
    A = (W + lam * D.T.dot(D)).tocsc()
    lu = splu(A)
    if data.ndim == 1:
        return lu.solve(w * data)
    out = np.empty_like(data)
    wy = w[:, None] * data
    for col in range(data.shape[1]):
        out[:, col] = lu.solve(wy[:, col])
    return out


def _preprocess_one(level_name, template_file, gen1_file, target_size, preprocess_size):
    """Preprocess one vertebra for template correspondence (joblib worker).

    Constructs a fresh FunctionalMapper inside the worker so all arguments
    are picklable (no logger, no shared fmapper object).
    """
    fmapper = FunctionalMapper(target_size=target_size)
    template = load_vertebral_template(template_file, fmapper)
    gen1_hr = Mesh.from_file(gen1_file, normalize=True)
    gen1_lr = fmapper.preprocess_mesh(gen1_hr.v, gen1_hr.f, target_size=preprocess_size)
    return level_name, template, (gen1_hr, gen1_lr)


class Aligner:

    def __init__(self, config):
        """Initialize Aligner with pipeline config; sets up directories and empty containers."""
        self.config = config
        self.template_dir = expanduser(config.template_dir)
        self.working_dir = expanduser(config.specimen_dir)
        self.analysis_dir = str(join(self.working_dir, config.output_dir))
        os.makedirs(self.analysis_dir, exist_ok=True)

        # study-level parameters
        self.fmapper = FunctionalMapper(target_size=config.fmapper_target_size)

        # screw-level parameters
        self.preop_verts = {}
        self.screws = []

        # CT data
        self.preop_img = nib.Nifti1Image(np.zeros([]), np.eye(4))
        self.preop_labels = nib.Nifti1Image(np.zeros([]), np.eye(4))

    def import_data(self):
        """Load preop CT, segmentation, and screw plan; validate inputs and generate seg overlay."""
        plan_file = join(self.working_dir, 'preop_plan.csv')
        preop_vol = join(self.working_dir, 'preop.nii.gz')
        preop_seg = segmentation_file(self.analysis_dir)

        required = {
            plan_file: 'CSV screw plan',
            preop_vol: 'preop CT (run dcm2niix first)',
            preop_seg: 'preop segmentation (run spinescrews-segment first)',
        }
        missing = [('%s — %s' % (path, desc)) for path, desc in required.items() if not isfile(path)]
        if missing:
            raise FileNotFoundError('missing required files:\n  ' + '\n  '.join(missing))

        self.preop_img = nib.as_closest_canonical(nib.load(preop_vol), True)
        self.preop_labels = nib.as_closest_canonical(nib.load(preop_seg), True)

        seg_values = np.unique(self.preop_labels.get_fdata()).tolist()
        seg_names = [val_seg[int(ii)] for ii in seg_values if ii != 0]
        level_names, self.screws = parse_preop_plan(plan_file)
        for level in level_names:
            if level not in seg_names:
                raise ValueError('Level %s from screw plan not found in segmentation (available: %s)' % (level, seg_names))

        seg_names = [s for s in seg_names if s not in ['C1', 'SA', 'S2']]
        self.preop_verts = {level: Vertebra(level) for level in seg_names}
        sanity_check_plan(self.screws)

    def normalize_preop(self):
        """
        Normalize preop image and segmentation to standard orientation.
        Outputs to 02_preop/{LEVEL}/.
        """
        preop_img = self.preop_img
        preop_labels = self.preop_labels
        num_loaded = 0
        label_data = preop_labels.get_fdata().astype(np.uint8)
        seg_values = np.unique(label_data).tolist()
        for seg in ['background', 'C1']:
            val = seg_val[seg]
            if val in seg_values:
                seg_values.remove(val)

        # go from superior to inferior so we can use LS/L5 as reference for SA
        # note that we extract ALL levels, regardless of whether there are screws planned at that level
        for ii in seg_values:
            level_name = val_seg[ii]
            level_dir = preop_level_dir(self.analysis_dir, level_name)
            affine_file = join(level_dir, 'preop_affine.npy')
            gen1_file = join(level_dir, 'preop_gen1.ply')

            os.makedirs(level_dir, exist_ok=True)

            if isfile(affine_file) and isfile(gen1_file):
                # use pre-computed transformation matrix and mesh; re-crop volumes
                vert = Vertebra(level_name)
                vert.affine = np.load(affine_file)
                v, f = igl.read_triangle_mesh(gen1_file)
                vert.set_mesh(v, f)
                label_world = Vertebra.binarize_label(preop_labels, ii)
                vert.import_volumes(preop_img, label_world)
                self.preop_verts[level_name] = vert
                num_loaded += 1
                continue

            log.info('  computing normalized coordinates for %s' % level_name)
            label_world = Vertebra.binarize_label(preop_labels, ii)

            if level_name == 'SA':
                neighb_above = 'LS' if 'LS' in self.preop_verts else 'L5'
                if neighb_above not in self.preop_verts:
                    raise RuntimeError('SA orientation requires %s processed first' % neighb_above)
                affine = np.load(join(preop_level_dir(self.analysis_dir, neighb_above), 'preop_affine.npy'))
            else:
                # orient to inferior end-plate of segmented vertebra (transform in real-world coordinates)
                affine, verts_g1, faces_g1, inflated_v, inflated_f, small2med, med2small = \
                    Vertebra.normalize_orientation(label_world)
                igl.write_triangle_mesh(join(level_dir, 'preop_gen1.ply'), verts_g1, faces_g1)
                igl.write_triangle_mesh(join(level_dir, 'preop_gen1-inflated.ply'), inflated_v, inflated_f)
                sparse.save_npz(join(level_dir, 'small2med.npz'), small2med)
                sparse.save_npz(join(level_dir, 'med2small.npz'), med2small)

            # verify that detected vertebra is close to corresponding screw plan
            screws = [S for S in self.screws if S.level == level_name]
            if len(screws) > 0:
                entry = [S.planned_entry for S in screws]
                tip = [S.planned_tip for S in screws]
                planned_midpt = np.nanmean(np.row_stack([entry, tip]), axis=0)
                if np.linalg.norm(affine[:3, 3] - planned_midpt) >= 50:
                    raise ValueError('Vertebra %s not close to planned screw' % level_name)

            bone = Vertebra(level_name, affine)
            bone.import_volumes(preop_img, label_world)
            np.save(join(level_dir, 'preop_affine.npy'), affine)
            self.preop_verts[level_name] = bone

        if num_loaded > 0:
            log.info('  loaded %d vertebrae from file' % num_loaded)

        return num_loaded

    def template_correspondence(self):
        """
        Match each segmented bone label to the corresponding template model.
        Outputs to 03_correspondence/{LEVEL}/.
        Returns dict of per-level metrics: {level: {'dg': float, 'coverage': float, 'mean_nnz': float}}.

        Three phases:
          1. Classify levels (done / cached / todo)
          2. Parallel preprocessing + cache save + figure generation
          3. Parallel PMF matching + save correspondence matrices
        """
        levels = self.preop_verts.keys()
        pmf_config = {"sigma": self.config.pmf_sigma, "gamma": self.config.pmf_gamma,
                      "iterations": self.config.pmf_iterations}
        dg_values = {}

        # ── Phase 1: classify levels ──────────────────────────────
        done_levels = []    # template2bone.npz exists → skip entirely
        cached_levels = []  # bone_preprocess.npz exists but template2bone.npz does not
        todo_levels = []    # neither exists → full preprocessing + PMF

        for level_name in levels:
            corr_dir = correspondence_level_dir(self.analysis_dir, level_name)
            if isfile(join(corr_dir, 'template2bone.npz')):
                done_levels.append(level_name)
            elif isfile(join(corr_dir, 'bone_preprocess.npz')):
                cached_levels.append(level_name)
            else:
                todo_levels.append(level_name)

        if done_levels:
            log.info('  %d levels already complete: %s', len(done_levels), ', '.join(done_levels))

        if not todo_levels and not cached_levels:
            log.info('  all vertebral correspondence already computed')
            return dg_values

        # ── Phase 2: parallel preprocessing + cache + figure ──────
        if todo_levels:
            log.info('Preprocessing %d levels (decimation + geodesic features): %s',
                     len(todo_levels), ', '.join(todo_levels))

            todo_args = []
            for level_name in todo_levels:
                template_file = join(self.template_dir, 'meshes', 'template_%s.ply' % level_name)
                gen1_file = join(preop_level_dir(self.analysis_dir, level_name), 'preop_gen1.ply')
                todo_args.append((level_name, template_file, gen1_file,
                                  self.config.fmapper_target_size,
                                  self.config.fmapper_preprocess_size))

            t_pre = time()
            preprocess_results = Parallel(n_jobs=self.config.n_jobs, verbose=10)(
                delayed(_preprocess_one)(*args) for args in todo_args)
            log.info('Parallel preprocessing: %.2fs for %d levels', time() - t_pre, len(todo_levels))
        else:
            preprocess_results = []

        # Save cache + generate figures for freshly preprocessed levels
        preprocessed = {}  # level_name -> (template, (gen1_hr, gen1_lr))
        for level_name, template, gen1 in preprocess_results:
            corr_dir = correspondence_level_dir(self.analysis_dir, level_name)
            os.makedirs(corr_dir, exist_ok=True)
            gen1_hr, gen1_lr = gen1
            gen1_lr.save_np(join(corr_dir, 'bone_preprocess.npz'))
            log.debug('  saved bone_preprocess.npz for %s (%d verts)',
                      level_name, gen1_lr.num_vertices())
            preprocessed[level_name] = (template, gen1)

            # generate preprocessing figure
            from spinescrews.figures.correspondence_preprocess import generate_preprocess_figure
            fig_path = join(corr_dir, 'preprocess.png')
            generate_preprocess_figure(gen1_lr, level_name, fig_path)

        # Load from cache for cached_levels
        for level_name in cached_levels:
            corr_dir = correspondence_level_dir(self.analysis_dir, level_name)
            log.debug('  loading cached preprocessing for %s', level_name)
            template_file = join(self.template_dir, 'meshes', 'template_%s.ply' % level_name)
            template = load_vertebral_template(template_file, self.fmapper)
            gen1_hr = Mesh.from_file(
                join(preop_level_dir(self.analysis_dir, level_name), 'preop_gen1.ply'),
                normalize=True)
            gen1_lr = Mesh.from_file(join(corr_dir, 'bone_preprocess.npz'))
            preprocessed[level_name] = (template, (gen1_hr, gen1_lr))

            # generate figure if missing
            fig_path = join(corr_dir, 'preprocess.png')
            if not isfile(fig_path):
                from spinescrews.figures.correspondence_preprocess import generate_preprocess_figure
                generate_preprocess_figure(gen1_lr, level_name, fig_path)

        # ── Phase 3: parallel PMF matching + save ─────────────────
        pmf_levels = list(preprocessed.keys())
        if not pmf_levels:
            return dg_values

        log.info('Spectral matching %d levels to template: %s',
                 len(pmf_levels), ', '.join(pmf_levels))

        pmf_templates = [preprocessed[ln][0] for ln in pmf_levels]
        pmf_gen1 = [preprocessed[ln][1] for ln in pmf_levels]

        t_pmf = time()
        results = Parallel(n_jobs=self.config.n_jobs, verbose=10)(
            delayed(pmf_match)(template[0], gen1[0], pmf_config, template[1], gen1[1])
            for template, gen1 in zip(pmf_templates, pmf_gen1))
        log.info('pmf_match total: %.2fs for %d vertebrae', time() - t_pmf, len(pmf_levels))

        for level_name, template, gen1, (dg, gen1_2_template, template_2_gen1) in zip(
                pmf_levels, pmf_templates, pmf_gen1, results):

            template_hr, template_lr = template
            gen1_hr, gen1_lr = gen1
            corr_dir = correspondence_level_dir(self.analysis_dir, level_name)
            os.makedirs(corr_dir, exist_ok=True)

            if dg > 0.06:
                log.warning('  %s: dg=%.3f — match sketchy, check before trusting anatomy measures', level_name, dg)
            log.debug('  matched %s to template, dg %.3f', level_name, dg)
            dg_values[level_name] = {'dg': float(dg)}

            # Save PMF correspondence matrices directly (sized for gen1_hr = preop_gen1.ply)
            sparse.save_npz(join(corr_dir, 'template2bone.npz'), template_2_gen1)
            sparse.save_npz(join(corr_dir, 'bone2template.npz'), gen1_2_template)

            # Compose full correspondence chain: template → small → inflated → raw
            vert = self.preop_verts[level_name]
            raw_v, raw_f = Vertebra.get_mesh(vert.seg_normalized)

            # Always use 3-step chain: template → small → inflated → raw
            level_dir = preop_level_dir(self.analysis_dir, level_name)
            inflated_file = join(level_dir, 'preop_gen1-inflated.ply')
            small2med_file = join(level_dir, 'small2med.npz')
            med2small_file = join(level_dir, 'med2small.npz')
            for f in (inflated_file, small2med_file, med2small_file):
                if not isfile(f):
                    raise FileNotFoundError(f'missing {basename(f)} for {level_name} — re-run step 02')

            inflated_v, inflated_f = igl.read_triangle_mesh(inflated_file)
            small2med = sparse.load_npz(small2med_file)
            med2small = sparse.load_npz(med2small_file)

            # inflated↔raw correspondence
            _, fidx, bc = surface_match(raw_v, inflated_v, inflated_f)
            inflated2raw = bc2sparse(inflated_f, fidx, bc, nV=len(inflated_v))
            _, fidx, bc = surface_match(inflated_v, raw_v, raw_f)
            raw2inflated = bc2sparse(raw_f, fidx, bc, nV=len(raw_v))

            # Compose full chain
            template2seg = inflated2raw @ small2med @ template_2_gen1
            seg2template = gen1_2_template @ med2small @ raw2inflated

            nnz_per_row = template2seg.getnnz(axis=1)
            coverage = np.count_nonzero(nnz_per_row) / len(nnz_per_row)
            mean_nnz = nnz_per_row.mean()
            log.info('  %s: template2seg coverage %.0f%%, %.1f nnz/row',
                     level_name, coverage * 100, mean_nnz)
            dg_values[level_name]['coverage'] = round(float(coverage), 3)
            dg_values[level_name]['mean_nnz'] = round(float(mean_nnz), 1)

            # Save raw mesh + composed correspondence
            igl.write_triangle_mesh(join(corr_dir, 'preop_seg.ply'), raw_v, raw_f)
            sparse.save_npz(join(corr_dir, 'template2seg.npz'), template2seg)
            sparse.save_npz(join(corr_dir, 'seg2template.npz'), seg2template)

            # generate match figure
            from spinescrews.figures.correspondence_match import generate_match_figure
            from spectral_match.tools.geometric_utilities import normalize_mesh
            label_dir = join(self.template_dir, 'labels', level_name)
            fig_path = join(corr_dir, 'match.png')
            seg_v_norm = normalize_mesh(raw_v, raw_f)[0]
            generate_match_figure(template_hr.v, template_hr.f,
                                  seg_v_norm, raw_f,
                                  template2seg, label_dir, level_name, fig_path)

        if dg_values:
            dgs = [m['dg'] for m in dg_values.values()]
            log.info('correspondence dg: median=%.3f, max=%.3f (%d levels)',
                     np.median(dgs), max(dgs), len(dgs))

        return dg_values

    def refine_orientation(self):
        """Step 04: refine preop affines using dense correspondence + spine smoothing.

        Phase A: Per-level rigid refinement from template2bone correspondence.
        Phase B: Spine-chain decomposition, outlier detection, smoothing.

        Saves preop_affine-refined.npy to 04_orient/{LEVEL}/.
        Updates self.preop_verts in memory for downstream steps.
        Returns dict of per-level metrics.
        """
        # ── Phase A: per-level rigid refinement ──────────────────
        raw_refined = {}  # level_name → 4x4 affine

        for level_name, vert in self.preop_verts.items():
            corr_dir = correspondence_level_dir(self.analysis_dir, level_name)
            t2b_file = join(corr_dir, 'template2bone.npz')
            if not isfile(t2b_file):
                log.warning('no correspondence for %s, keeping original affine', level_name)
                raw_refined[level_name] = vert.affine.copy()
                continue

            template2bone = sparse.load_npz(t2b_file)
            template_file = join(self.template_dir, 'meshes', 'template_%s.ply' % level_name)
            template_v, _ = igl.read_triangle_mesh(template_file)
            gen1_file = join(preop_level_dir(self.analysis_dir, level_name), 'preop_gen1.ply')
            gen1_v, _ = igl.read_triangle_mesh(gen1_file)

            paired_template = template2bone @ template_v   # (n_gen1, 3) template frame
            T_corr = rigid_reg(paired_template, gen1_v, scale=True)

            # Decompose similarity into scale + pure rotation.
            # Scale maps unit-normalised template → mm-scale gen1; it is saved
            # separately for visualisation and MUST NOT enter the pipeline
            # affines which stay in mm throughout.
            R_scaled = T_corr[:3, :3]
            s = np.cbrt(np.abs(np.linalg.det(R_scaled)))
            R_pure = R_scaled / s

            # Pipeline affine: rotation-only correction (no scale)
            T_pipeline = np.eye(4)
            T_pipeline[:3, :3] = R_pure
            T_pipeline[:3, 3] = gen1_v.mean(0) - R_pure @ paired_template.mean(0)
            raw_refined[level_name] = vert.affine @ T_pipeline

            # Save template→normalised scale for visualisation only
            o_dir = orient_level_dir(self.analysis_dir, level_name)
            os.makedirs(o_dir, exist_ok=True)
            np.save(join(o_dir, 'template_scale.npy'), np.float64(s))

            angle_deg = np.degrees(np.linalg.norm(R_to_twist(R_pure)))
            trans_mm = np.linalg.norm(T_pipeline[:3, 3])
            log.info('  %s raw refinement: %.2f° rotation, %.2f mm translation (scale=%.1f)',
                     level_name, angle_deg, trans_mm, s)

        # ── Phase B: sigmoid anchor weights + absolute-space smoothing ──
        ordered_levels = [l for l in possible_levels if l in raw_refined]
        n = len(ordered_levels)
        trunk = list(range(-1, n - 1))

        abs_affs = np.stack([raw_refined[l] for l in ordered_levels])  # (n, 4, 4)

        # Load scales from Phase A
        scales = np.zeros(n)
        for i, l in enumerate(ordered_levels):
            scale_file = join(orient_level_dir(self.analysis_dir, l), 'template_scale.npy')
            if isfile(scale_file):
                scales[i] = float(np.load(scale_file))
        missing = scales == 0
        if missing.any() and (~missing).any():
            scales[missing] = np.median(scales[~missing])

        # --- Stage A: anchor weights from intervertebral params ---
        rel_twist, rel_trans = aff_to_rel_params(trunk, abs_affs)

        # Per-joint z-scores (joints 1..n-1 are intervertebral)
        twist_norms = np.degrees(np.linalg.norm(rel_twist, axis=1))  # (n,)
        off_axis = np.sqrt(rel_trans[:, 0]**2 + rel_trans[:, 1]**2)  # (n,) R+A only

        jt_norms = twist_norms[1:]  # (n-1,) intervertebral only
        jt_offax = off_axis[1:]

        med_tw = np.median(jt_norms)
        mad_twist = max(1.4826 * np.median(np.abs(jt_norms - med_tw)), 3.0)
        jz_twist = np.abs(jt_norms - med_tw) / mad_twist

        med_ox = np.median(jt_offax)
        mad_offax = max(1.4826 * np.median(np.abs(jt_offax - med_ox)), 1.5)
        jz_offax = np.abs(jt_offax - med_ox) / mad_offax

        # Per-level: min of joints above and below
        z_twist = np.zeros(n)
        z_offax = np.zeros(n)
        for i in range(n):
            cand_tw, cand_ox = [], []
            if i > 0:       # joint below (between level i-1 and i)
                cand_tw.append(jz_twist[i - 1])
                cand_ox.append(jz_offax[i - 1])
            if i < n - 1:   # joint above (between level i and i+1)
                cand_tw.append(jz_twist[i])
                cand_ox.append(jz_offax[i])
            z_twist[i] = min(cand_tw)
            z_offax[i] = min(cand_ox)

        # Scale z-score (per-level)
        med_sc = np.median(scales)
        mad_scale = max(1.4826 * np.median(np.abs(scales - med_sc)), 50.0)
        z_scale = np.abs(scales - med_sc) / mad_scale

        # Sigmoid anchor weight
        SIGMOID_CENTER = 3.0
        SIGMOID_STEEPNESS = 2.0
        z_max = np.maximum(np.maximum(z_twist, z_offax), z_scale)
        anchor = 1.0 / (1.0 + np.exp(SIGMOID_STEEPNESS * (z_max - SIGMOID_CENTER)))

        # End vertebrae only have one neighboring joint and sit at anatomical
        # transitions — require a higher z-score to flag them
        END_SIGMOID_CENTER = 5.0
        for idx in (0, -1):
            anchor[idx] = 1.0 / (1.0 + np.exp(SIGMOID_STEEPNESS * (z_max[idx] - END_SIGMOID_CENTER)))

        for i in range(n):
            if anchor[i] < 0.95:
                log.info('  %s anchor=%.3f (z_twist=%.2f, z_offax=%.2f, z_scale=%.2f)',
                         ordered_levels[i], anchor[i], z_twist[i], z_offax[i], z_scale[i])

        # --- Stage B: Whittaker smoothing (rotations) + curve projection (translations) ---
        abs_twist_vecs = np.array([R_to_twist(abs_affs[i, :3, :3]) for i in range(n)])
        final_twist_vecs = _whittaker_smooth(abs_twist_vecs, anchor, lam=self.config.orient_lam_rot)

        abs_trans_vecs = abs_affs[:, :3, 3].copy()
        final_trans_vecs = _whittaker_smooth(abs_trans_vecs, anchor, lam=self.config.orient_lam_trans)

        final_scales = _whittaker_smooth(scales, anchor, lam=self.config.orient_lam_scale)

        final_affs = np.zeros((n, 4, 4))
        for i in range(n):
            final_affs[i, :3, :3] = twist_to_R(final_twist_vecs[i])
            final_affs[i, :3, 3] = final_trans_vecs[i]
            final_affs[i, 3, 3] = 1.0

        # ── Save + update in-memory state ─────────────────────────
        metrics = {}
        for i, level_name in enumerate(ordered_levels):
            o_dir = orient_level_dir(self.analysis_dir, level_name)
            os.makedirs(o_dir, exist_ok=True)

            # Update in-memory vertebra
            vert = self.preop_verts[level_name]
            old_aff = vert.affine.copy()
            vert.affine = final_affs[i]
            label_world = Vertebra.binarize_label(self.preop_labels, seg_val[level_name])
            vert.import_volumes(self.preop_img, label_world)

            # Transform cached gen1 mesh to refined frame
            if vert.verts_ is not None:
                delta_mesh = np.linalg.inv(final_affs[i]) @ old_aff
                vert.verts_ = (delta_mesh[:3, :3] @ vert.verts_.T).T + delta_mesh[:3, 3]

            # Save to 04_orient/{LEVEL}/
            np.save(join(o_dir, 'preop_affine-refined.npy'), final_affs[i])
            np.save(join(o_dir, 'template_scale.npy'), np.float64(final_scales[i]))
            ct = np.clip(np.round(vert.img_normalized.get_fdata()),
                         -32768, 32767).astype(np.int16)
            nib.save(nib.Nifti1Image(ct, vert.img_normalized.affine),
                     join(o_dir, 'preop.nii.gz'))
            nib.save(vert.seg_normalized, join(o_dir, 'preop_seg.nii.gz'))

            # Log total correction from original
            orig_aff = np.load(join(preop_level_dir(self.analysis_dir, level_name), 'preop_affine.npy'))
            delta = np.linalg.inv(orig_aff) @ final_affs[i]
            angle = np.degrees(np.linalg.norm(R_to_twist(delta[:3, :3])))
            trans = np.linalg.norm(delta[:3, 3])
            metrics[level_name] = {'angle_deg': float(angle), 'trans_mm': float(trans),
                                    'anchor_weight': float(anchor[i])}
            log.info('  %s final refinement: %.2f° / %.2f mm (anchor=%.3f)',
                     level_name, angle, trans, anchor[i])

        return metrics


def _log_elapsed(label, elapsed):
    """Log elapsed time in seconds or minutes."""
    if elapsed < 60:
        log.info('*** %s took %.2f seconds' % (label, elapsed))
    else:
        log.info('*** %s took %.2f minutes' % (label, elapsed / 60))


def _run_import(study, analysis_dir, data_dir):
    """Import data + segmentation overlay figure."""
    t = time()
    study.import_data()
    _log_elapsed('Data import', time() - t)

    # Write step 01 gate file if run_segmentation.py didn't (e.g. external segmentation)
    seg_dir = segmentation_dir(analysis_dir)
    if not step_complete(seg_dir):
        os.makedirs(seg_dir, exist_ok=True)
        write_summary(seg_dir, {
            'levels': list(study.preop_verts.keys()),
            'note': 'gate written by align_preop (segmentation ran externally)',
        })

    from spinescrews.figures.seg_overlay import generate_seg_overlay
    generate_seg_overlay(analysis_dir, data_dir)


def _run_preop(study, analysis_dir):
    """Step 02: preop normalization."""
    step_dir = preop_dir(analysis_dir)
    if step_complete(step_dir):
        log.info('*** Step 02 (preop) complete, loading from file')
        study.normalize_preop()
        return

    t = time()
    num_loaded = study.normalize_preop()
    elapsed = time() - t
    _log_elapsed('Normalization', elapsed)

    from bg3dtools.render.o3d import run_isolated
    from spinescrews.figures.spine_template import generate_spine_construct
    run_isolated(generate_spine_construct, analysis_dir, study.template_dir, step='preop')

    write_summary(step_dir, {
        'n_computed': len(study.preop_verts) - num_loaded,
        'n_loaded': num_loaded,
        'elapsed_s': round(elapsed, 1),
    })


def _run_correspondence(study, analysis_dir):
    """Step 03: template correspondence."""
    step_dir = correspondence_dir(analysis_dir)
    if step_complete(step_dir):
        log.info('*** Step 03 (correspondence) complete, loading from file')
        study.template_correspondence()
        return

    t = time()
    dg_values = study.template_correspondence()
    elapsed = time() - t
    _log_elapsed('Template correspondence', elapsed)

    per_level = {}
    for level_name in study.preop_verts:
        if level_name in dg_values:
            metrics = dg_values[level_name]
            dg = metrics['dg']
            status = 'warning' if dg > 0.06 else 'ok'
            per_level[level_name] = {
                'dg': dg,
                'coverage': metrics.get('coverage'),
                'mean_nnz': metrics.get('mean_nnz'),
                'status': status,
            }
        else:
            per_level[level_name] = {'dg': None, 'status': 'loaded'}

    write_summary(step_dir, {
        'n_computed': len(dg_values),
        'n_loaded': len(study.preop_verts) - len(dg_values),
        'per_level': per_level,
        'elapsed_s': round(elapsed, 1),
    })


def _run_orient(study, analysis_dir):
    """Step 04: orientation refinement."""
    step_dir = orient_dir(analysis_dir)
    if step_complete(step_dir):
        log.info('*** Step 04 (orient) complete, loading from file')
        for level_name, vert in study.preop_verts.items():
            o_dir = orient_level_dir(analysis_dir, level_name)
            refined_file = join(o_dir, 'preop_affine-refined.npy')
            if isfile(refined_file):
                old_aff = vert.affine.copy()
                vert.affine = np.load(refined_file)
                label_world = Vertebra.binarize_label(study.preop_labels, seg_val[level_name])
                vert.import_volumes(study.preop_img, label_world)
                if vert.verts_ is not None:
                    delta = np.linalg.inv(vert.affine) @ old_aff
                    vert.verts_ = (delta[:3, :3] @ vert.verts_.T).T + delta[:3, 3]
        return

    t = time()
    metrics = study.refine_orientation()
    elapsed = time() - t
    _log_elapsed('Orientation refinement', elapsed)

    from spinescrews.figures.orient_refinement import generate_orient_summary
    generate_orient_summary(analysis_dir)

    from spinescrews.figures.preop_orientation import generate_orientation_summary
    generate_orientation_summary(analysis_dir)

    from bg3dtools.render.o3d import run_isolated
    from spinescrews.figures.spine_template import generate_spine_construct
    run_isolated(generate_spine_construct, analysis_dir, study.template_dir, step='orient')

    write_summary(step_dir, {
        'per_level': metrics,
        'elapsed_s': round(elapsed, 1),
    })


def _preop_summary(analysis_dir):
    """Read step 01-04 summaries and report any quality warnings."""
    warnings = []

    # Step 01 — segmentation
    seg_dir = segmentation_dir(analysis_dir)
    if not step_complete(seg_dir):
        warnings.append('Step 01 (segmentation) did not complete')

    # Step 02 — preop
    pre_dir = preop_dir(analysis_dir)
    if not step_complete(pre_dir):
        warnings.append('Step 02 (preop) did not complete')

    # Step 03 — correspondence
    corr_dir_ = correspondence_dir(analysis_dir)
    if not step_complete(corr_dir_):
        warnings.append('Step 03 (correspondence) did not complete')
    else:
        s = read_summary(corr_dir_)
        for level, m in s.get('per_level', {}).items():
            if m.get('status') == 'warning':
                warnings.append('%s: correspondence match sketchy (dg=%.3f, threshold 0.06)' % (level, m['dg']))
            cov = m.get('coverage')
            if cov is not None and cov < 0.5:
                warnings.append('%s: low correspondence coverage (%.0f%%)' % (level, cov * 100))

    # Step 04 — orient
    ori_dir = orient_dir(analysis_dir)
    if not step_complete(ori_dir):
        warnings.append('Step 04 (orient) did not complete')
    else:
        s = read_summary(ori_dir)
        for level, m in s.get('per_level', {}).items():
            if m.get('angle_deg', 0) > 25:
                warnings.append('%s: large orientation correction (%.1f deg)' % (level, m['angle_deg']))
            if m.get('trans_mm', 0) > 10:
                warnings.append('%s: large translation correction (%.1f mm)' % (level, m['trans_mm']))
            if m.get('anchor_weight', 1) < 0.5:
                warnings.append('%s: low anchor weight (%.2f) -- refinement uncertain' % (level, m['anchor_weight']))

    # Report
    if warnings:
        log.info('=' * 40)
        log.info('=== PREOP WARNINGS ===')
        for w in warnings:
            log.info('  - %s' % w)
        log.info('%d warning(s) found -- review before trusting results' % len(warnings))
        log.info('=' * 40)
    else:
        log.info('=== Preop steps completed cleanly (steps 01-04) ===')


def run_preop(config):
    """Run preop alignment steps 01-04 from a config object.

    Callable by the orchestrator (align_vertebrae.py) without CLI arg parsing.
    Returns early if step 04 is already complete.
    """
    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    if step_complete(orient_dir(analysis_dir)):
        log.info('Preop steps already complete — skipping')
        return

    study = Aligner(config)
    _run_import(study, analysis_dir, data_dir)
    _run_preop(study, analysis_dir)
    _run_correspondence(study, analysis_dir)
    _run_orient(study, analysis_dir)
    _preop_summary(analysis_dir)


def main():
    """CLI entry point for preop alignment (steps 01-04). Called by spinescrews-preop console script."""
    t0 = time()
    import argparse
    from spinescrews.tools.config import (load_config, save_resolved_config,
                                          add_common_pipeline_args, overrides_from_args)

    parser = argparse.ArgumentParser(
        description='Preop alignment (steps 01-04): genus-1 mesh extraction, spectral template '
                    'correspondence, and Whittaker-smoothed orientation refinement. Requires '
                    'preop.nii.gz, preop_plan.csv, and segmentation output (run '
                    'spinescrews-segment first). Outputs go to <specimen_dir>/analysis/.')
    parser.add_argument('specimen_dir',
                        help='Specimen directory containing preop.nii.gz / preop_plan.csv and '
                             'segmentation output; results go to <specimen_dir>/analysis/.')
    add_common_pipeline_args(parser)
    args = parser.parse_args()

    overrides = overrides_from_args(args)

    config = load_config(args.specimen_dir, overrides=overrides)
    save_resolved_config(config)

    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    logfile = join(analysis_dir, 'preop.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])

    log.info('*' * (31 + len(data_dir)))
    log.info('**  Preop alignment for %s  **' % data_dir)
    log.info('*' * (31 + len(data_dir)))

    run_preop(config)

    log.info('*** Total preop time: %.2f minutes' % ((time() - t0) / 60))


if __name__ == '__main__':
    main()
