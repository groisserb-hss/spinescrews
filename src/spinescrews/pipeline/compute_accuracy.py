import os
import sys
import logging
from os.path import join, expanduser
from time import time

import nibabel
import numpy as np
import pandas as pd
import igl
import trimesh
from scipy.stats import mode

from scipy import sparse
from spinescrews.tools import ScrewMeasures, BreachMeasures, MeshLabels, dimR, dimA, dimS, possible_levels
from bg3dtools.transforms_unified import transform_points_inverse, transform_points_forward, make_aff
from bg3dtools.mesh.utils import per_face_normals, submesh, per_vertex_normals, per_vertex_smoothing
from bg3dtools.pointclouds.quantize import voxelize, convert_to_points, sparse_quantize
from bg3dtools.pointclouds.fitting import project_to_line, project_to_plane
from spectral_match.pipeline import Mesh
from bg3dtools.mesh.clean import make_manifold, remove_ears, fill_hole, largest_patch
from spinescrews.tools.screw_models import parse_preop_plan, sanity_check_plan, Screw
from spinescrews.tools.error import distance_to_pedicle, measure_screw_error, align_to_screw
from bg3dtools.mesh.mesh_io import read_colored_plyfile
from spinescrews.tools.vertebrae import Vertebra
from spinescrews.tools.paths import (preop_level_dir, correspondence_level_dir,
                         orient_level_dir,
                         detection_dir, registration_level_dir,
                         accuracy_dir, breach_mesh_dir, step_complete, write_summary, timed)

log = logging.getLogger(__name__)

_has_embree = hasattr(trimesh.ray, 'ray_pyembree')



class ErrorComputer:
    def __init__(self, config):
        """Initialize ErrorComputer with pipeline config; sets up directories and empty containers."""
        self.config = config

        self.specimen_dir = expanduser(config.specimen_dir)
        self.analysis_dir = join(self.specimen_dir, config.output_dir)
        self.anatomic_axis = config.anatomic_axis

        # screw-level parameters
        self.preop_verts = {}  # to hold preop affine transforms and raw meshes
        self.preop_gen1 = {}
        self.postop_verts = {}  # to hold postop affine transforms and aligned template meshes
        self.screws = []  # to be loaded in normalized coordinates

        self.screw_results = None
        self.breach_results = None

        self.template_dir = expanduser(config.template_dir)
        self.template_labels = {}

    def import_data(self):
        """Load screw plan, template labels, detected screws, and preop/postop geometry from disk."""
        plan_file = join(self.specimen_dir, 'preop_plan.csv')
        missing = []
        if not os.path.isfile(plan_file):
            missing.append('%s — CSV screw plan' % plan_file)
        if not os.path.isdir(self.analysis_dir):
            missing.append('%s — analysis directory (run align_vertebrae.py first)' % self.analysis_dir)
        if missing:
            raise FileNotFoundError('missing required files:\n  ' + '\n  '.join(missing))

        # load planned screw positions
        level_names, screws = parse_preop_plan(plan_file)
        level_names = [level for level in level_names if level[0]]
        screws = [screw for screw in screws if screw.level[0]]

        sanity_check_plan(screws)
        # remove sacral screws
        screws = [screw for screw in screws if 'S' not in screw.level]

        def load_color(label_file):
            """Load per-vertex template label weights from a colored PLY (green/blue channel)."""
            v, f, c = read_colored_plyfile(label_file, vert_colors=True)
            c = (255 - c[:, 1].astype(np.float32)) / 255
            return c

        # load template labels
        for level in possible_levels:
            label_dir = join(self.template_dir, 'labels', level)
            left_ped = load_color(join(label_dir, 'left_pedicle.ply'))
            right_ped = load_color(join(label_dir, 'right_pedicle.ply'))
            canal = load_color(join(label_dir, 'canal.ply'))
            endplate_top = load_color(join(label_dir, 'superior_endplate.ply'))
            endplate_bottom = load_color(join(label_dir, 'inferior_endplate.ply'))
            body_walls = load_color(join(label_dir, 'body_walls.ply'))
            self.template_labels[level] = MeshLabels(left_ped=left_ped, right_ped=right_ped,
                                                     canal=canal, body_walls=body_walls,
                                                     endplate_top=endplate_top, endplate_bottom=endplate_bottom)

        # load detected screw positions from 05_detection/
        det_dir = detection_dir(self.analysis_dir)
        for screw in screws:
            screw_file = join(det_dir, screw.name + "_screw.yml")
            screw.load_from_yaml(screw_file)

        # load transformations, mesh file
        for level in level_names:
            o_dir = orient_level_dir(self.analysis_dir, level)
            preop_ldir = preop_level_dir(self.analysis_dir, level)

            # preop data from 04_orient/{LEVEL}/ (refined affine + volumes)
            vert = Vertebra(level)
            vert.affine = np.load(join(o_dir, 'preop_affine-refined.npy'))
            vert.img_normalized = nibabel.load(join(o_dir, 'preop.nii.gz'))
            vert.seg_normalized = nibabel.load(join(o_dir, 'preop_seg.nii.gz'))

            # Load gen1 from step 02, transform to refined frame
            v_gen1, f_gen1 = igl.read_triangle_mesh(join(preop_ldir, 'preop_gen1.ply'))
            orig_aff = np.load(join(preop_ldir, 'preop_affine.npy'))
            delta = np.linalg.inv(vert.affine) @ orig_aff
            v_gen1 = (delta[:3, :3] @ v_gen1.T).T + delta[:3, 3]
            vert.set_mesh(v_gen1, f_gen1)

            self.preop_verts[level] = vert
            self.preop_gen1[level] = Mesh(v_gen1, f_gen1)

            # template correspondence from 03_correspondence/{LEVEL}/
            corr_dir = correspondence_level_dir(self.analysis_dir, level)
            vert.template2mesh = sparse.load_npz(join(corr_dir, 'template2bone.npz')).tocsr()
            vert.mesh2template = sparse.load_npz(join(corr_dir, 'bone2template.npz')).tocsr()

            # postop registered data from 06_registration/{LEVEL}/
            reg_dir = registration_level_dir(self.analysis_dir, level)
            self.postop_verts[level] = Vertebra.load(reg_dir, level, 'postop-reg')

        # apply affine transformations to normalize screw positions
        for screw in screws:
            planned_pts = np.row_stack([screw.planned_entry, screw.planned_tip])
            tform_preop = self.preop_verts[screw.level].affine
            planned_pts = transform_points_inverse(tform_preop, planned_pts)

            screw.planned_entry = planned_pts[0]
            screw.planned_tip = planned_pts[1]

            if screw.type != 'skip':
                detected_pts = np.row_stack([screw.detected_entry, screw.detected_tip])
                tform_postop = self.postop_verts[screw.level].affine
                detected_pts = transform_points_inverse(tform_postop, detected_pts)

                screw.detected_entry = detected_pts[0]
                screw.detected_tip = detected_pts[1]

        nS = len(screws)
        self.screws = screws
        self.screw_results = np.nan * np.empty([nS, len(ScrewMeasures._fields)])
        self.breach_results = np.nan * np.empty([nS, len(BreachMeasures._fields)])

    def compute_breach_and_error(self, timings=None):
        """Measure pedicle breach distance and screw placement error for all placed screws."""
        if not _has_embree:
            log.warning('embreex not installed — canal mesh construction will be slow (pip install embreex)')
        with timed('construct_canal_meshes', timings):
            canal_raw = {name: self._construct_canal_mesh(name) for name in self.preop_verts.keys()}
        canal_dict = {}
        level_times = {}
        section_totals = {'ray': 0, 'cleanup': 0, 'inflate': 0, 'boolean': 0}
        for name, (cv, cf, elapsed, sections) in canal_raw.items():
            canal_dict[name] = (cv, cf)
            level_times[name] = elapsed
            for k, v in sections.items():
                section_totals[k] += v
        top3 = sorted(level_times.items(), key=lambda x: x[1], reverse=True)[:3]
        slowest_str = ', '.join('%s %.1fs' % (k, v) for k, v in top3)
        breakdown = '  '.join('%s=%.0fs' % (k, v) for k, v in section_totals.items())
        log.info('    canal meshes: %d levels, slowest: %s', len(level_times), slowest_str)
        log.info('    breakdown (total): %s', breakdown)

        acc_dir = accuracy_dir(self.analysis_dir)
        os.makedirs(acc_dir, exist_ok=True)

        with timed('measure_screws', timings):
            breached = []
            for ii, screw in enumerate(self.screws):
                if screw.type == 'skip': continue

                log.debug('measuring error for screw %s' % screw.name)
                mesh_dir = breach_mesh_dir(self.analysis_dir, screw.name)
                self.breach_results[ii], ped_y = self.measure_pedicle_proximity(
                    screw, self.preop_verts[screw.level],
                    canal_dict[screw.level], mesh_dir)
                self.screw_results[ii] = measure_screw_error(screw, ped_y, self.anatomic_axis)
                if self.breach_results[ii, 0] > 0:
                    breached.append((screw.name, ii, self.breach_results[ii, 0]))
            placed = [s for s in self.screws if s.type != 'skip']
            if breached:
                log.info('    measured %d screws, %d breached:',
                         len(placed), len(breached))
                for name, idx, bdist in breached:
                    ml = self.screw_results[idx, 3]
                    si = self.screw_results[idx, 4]
                    log.info('      %s: M-L=%.2f  S-I=%.2f  breach=+%.1fmm',
                             name, ml, si, bdist)
            else:
                log.info('    measured %d screws, no breaches', len(placed))

    def measure_pedicle_proximity(self, screw: Screw, bone: Vertebra, canal_mesh: tuple,
                                  mesh_output_dir=None) -> (BreachMeasures, float):
        """Compute medial breach distance and closest points for one screw, returning (BreachMeasures, ped_y)."""
        planned_pts, detected_pts, bone_v, bone_f, canal_v, canal_f, ntl_tform = self.normalize_to_left(screw, bone, canal_mesh)

        # Planned screw breach (for reference / ped_y measurement)
        log.info('  %s: computing planned screw breach...', screw.name)
        planned_dist, _, close_pt = distance_to_pedicle(bone_v, bone_f, planned_pts, screw.shaft_rad, canal_v, canal_f)

        # Detected screw breach
        log.info('  %s: computing detected screw breach...', screw.name)
        hr = getattr(screw, 'head_rad', None)
        hl = getattr(screw, 'head_len', None)
        dist, ped_pt, screw_pt = distance_to_pedicle(bone_v, bone_f, detected_pts, screw.shaft_rad, canal_v, canal_f,
                                                      mesh_output_dir=mesh_output_dir,
                                                      head_rad=hr, head_len=hl)

        # Breach angle from penetration direction: sign(dist) * (screw_pt - ped_pt)
        # projected onto the axial plane (x-z, perpendicular to screw axis).
        # Angle < 180° = medial direction.  Since breached_distance already
        # filters to medial-only samples, this angle is informational.
        breach_vec = np.sign(dist) * (screw_pt - ped_pt)
        breach_angle = (np.arctan2(breach_vec[0], breach_vec[2]) * 180 / np.pi) % 360

        log.info('  %s RESULT: dist=%.3f  breach_angle=%.1f°',
                 screw.name, dist, breach_angle)
        log.info('    ped_pt=[%.2f, %.2f, %.2f]  screw_pt=[%.2f, %.2f, %.2f]  '
                 'breach_vec=[%.3f, %.3f, %.3f]',
                 ped_pt[0], ped_pt[1], ped_pt[2], screw_pt[0], screw_pt[1], screw_pt[2],
                 breach_vec[0], breach_vec[1], breach_vec[2])

        # Transform closest points back to world space:
        # 1. undo align_to_screw  2. undo R-mirror  3. normalized → world
        closest_pts = transform_points_forward(ntl_tform, np.row_stack([ped_pt, screw_pt]))
        if screw.name[-1] == 'R':
            closest_pts[:, dimR] *= -1
        closest_pts = transform_points_forward(bone.affine, closest_pts)
        ped_pt_world, screw_pt_world = closest_pts[0], closest_pts[1]

        pt_y = close_pt[dimA] - planned_pts[0][dimA]
        anatomy = BreachMeasures(breach_dist=dist, breach_angle=breach_angle, planned_breach_dist=planned_dist,
                                 screw_pt_x=screw_pt_world[0], screw_pt_y=screw_pt_world[1], screw_pt_z=screw_pt_world[2],
                                 ped_pt_x=ped_pt_world[0], ped_pt_y=ped_pt_world[1], ped_pt_z=ped_pt_world[2])
        return anatomy, pt_y

    def normalize_to_left(self, screw: Screw, vertebra: Vertebra, canal_mesh: tuple) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Mirror right screws to left side, align to planned screw axis, return transformed geometry."""
        planned = np.row_stack([screw.planned_entry, screw.planned_tip])
        detected = np.row_stack([screw.detected_entry, screw.detected_tip])
        bone_v, bone_f = vertebra.verts.copy(), vertebra.faces
        canal_v, canal_f = canal_mesh[0].copy(), canal_mesh[1]
        labels = self.template_labels[screw.level]

        vmap = vertebra.template2mesh
        if screw.name[-1] == 'R':
            planned[:, dimR] *= -1
            detected[:, dimR] *= -1
            bone_v[:, dimR] *= -1
            bone_f = bone_f[:, [0, 2, 1]]
            canal_v[:, dimR] *= -1
            canal_f = canal_f[:, [0, 2, 1]]
            ped_weight = vmap @ labels.right_ped.reshape([-1, 1])
        else:
            ped_weight = vmap @ labels.left_ped.reshape([-1, 1])

        ped_center = np.sum(ped_weight * bone_v, axis=0) / np.sum(ped_weight)
        planned_vec = planned[1] - planned[0]
        # align mesh to axis of PLANNED screw, centered on pedicle neck
        tform = align_to_screw(ped_center, ped_center + planned_vec)

        # transform points to new coordinate system
        bone_v = transform_points_inverse(tform, bone_v)
        canal_v = transform_points_inverse(tform, canal_v)
        planned = transform_points_inverse(tform, planned)
        detected = transform_points_inverse(tform, detected)

        return planned, detected, bone_v, bone_f, canal_v, canal_f, tform

    def _construct_canal_mesh(self, level_name) -> tuple[np.ndarray, np.ndarray, float, dict]:
        """Build spinal canal mesh for one level via ray-visibility + boolean difference with bone."""
        t_level = time()
        vertebra = self.preop_verts[level_name]
        verts, faces = vertebra.verts, vertebra.faces

        # use gen1 mesh already transformed to refined frame
        gen1 = self.preop_gen1[level_name]
        v_gen1, f_gen1 = gen1.v.copy(), gen1.f.copy()

        canal_center = np.zeros([1, 3])

        # find faces of genus-1 mesh that have an unobstructed view of the canal center
        bc = igl.barycenter(v_gen1, f_gen1)
        bc += per_face_normals(v_gen1, f_gen1) * 0.001
        bc_to_center = canal_center - bc
        dist_to_center = np.linalg.norm(bc_to_center, axis=1)
        directions = bc_to_center / dist_to_center[:, None]
        intersector = trimesh.Trimesh(v_gen1, f_gen1).ray
        locations, index_ray, index_tri = intersector.intersects_location(bc, directions)
        hit_dists = np.linalg.norm(locations - bc[index_ray], axis=1)

        # skip self-intersections with the originating face or its neighbors
        real_hits = hit_dists > 0.5
        index_ray = index_ray[real_hits]
        hit_dists = hit_dists[real_hits]

        # for each ray, find the closest real intersection
        clear_view = np.ones(len(f_gen1), dtype=bool)
        if len(index_ray) > 0:
            order = np.lexsort((hit_dists, index_ray))
            _, first_idx = np.unique(index_ray[order], return_index=True)
            nearest_ray = index_ray[order[first_idx]]
            nearest_dist = hit_dists[order[first_idx]]
            clear_view[nearest_ray] = nearest_dist > dist_to_center[nearest_ray]

        sub_v, sub_f = submesh(v_gen1, f_gen1, clear_view, return_indices=False)
        t_ray = time() - t_level

        # clean up mesh
        t0 = time()
        sub_v, sub_f = largest_patch(sub_v, sub_f)
        sub_v, sub_f = remove_ears(sub_v, sub_f)
        sub_v, sub_f = largest_patch(sub_v, sub_f)

        mesh = trimesh.Trimesh(sub_v, sub_f)
        trimesh.repair.fix_normals(mesh)  # igl can't handle non-watertight meshes for winding number?
        sub_v, sub_f = mesh.vertices, mesh.faces
        boundary = np.atleast_1d(igl.boundary_loop(mesh.faces))
        while len(boundary) > 2:
            sub_f = fill_hole(sub_v, sub_f, boundary)
            mesh = trimesh.Trimesh(sub_v, sub_f)
            trimesh.repair.fix_normals(mesh)
            sub_v, sub_f = mesh.vertices, mesh.faces
            boundary = np.atleast_1d(igl.boundary_loop(sub_f))
        canal_v, canal_f = make_manifold(sub_v, sub_f)
        t_cleanup = time() - t0

        # inflate canal mesh
        t0 = time()
        canal_v = per_vertex_smoothing(canal_v, canal_f)
        for _ in range(10):
            n = per_vertex_normals(canal_v, canal_f)
            canal_v += 0.5 * n
            canal_v = per_vertex_smoothing(canal_v, canal_f)
        t_inflate = time() - t0

        t0 = time()
        canal_mesh = trimesh.Trimesh(canal_v, canal_f)
        bone_mesh = trimesh.Trimesh(verts, faces)

        result = canal_mesh.difference(bone_mesh, check_volume=False)
        if isinstance(result, trimesh.Scene):
            result = result.dump(concatenate=True)
        canal_v, canal_f = largest_patch(result.vertices, result.faces)
        t_boolean = time() - t0

        log.debug('%s canal mesh: %d v, %d f  |  ray=%.1fs  cleanup=%.1fs  inflate=%.1fs  boolean=%.1fs',
                  level_name, len(canal_v), len(canal_f), t_ray, t_cleanup, t_inflate, t_boolean)
        elapsed = time() - t_level
        sections = {'ray': t_ray, 'cleanup': t_cleanup, 'inflate': t_inflate, 'boolean': t_boolean}
        return canal_v, canal_f, elapsed, sections


def main():
    """CLI entry point for accuracy computation (step 07). Called by spinescrews-accuracy console script."""
    t0 = time()
    import argparse
    from spinescrews.tools.config import load_config, save_resolved_config

    parser = argparse.ArgumentParser(description='Compute screw placement accuracy.')
    parser.add_argument('specimen_dir', type=str)
    args = parser.parse_args()

    config = load_config(args.specimen_dir)
    save_resolved_config(config)

    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)
    logfile = join(analysis_dir, 'compute_accuracy.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])

    log.info('*' * (31 + len(data_dir)))
    log.info('**  Computing accuracy for %s  **' % data_dir)
    log.info('*' * (31 + len(data_dir)))

    acc_dir = accuracy_dir(analysis_dir)

    if not step_complete(acc_dir):
        timings = {}
        study = ErrorComputer(config)
        with timed('import_data', timings):
            study.import_data()
        study.compute_breach_and_error(timings=timings)

        screw_names = [screw.name for screw in study.screws]
        col_names = list(ScrewMeasures._fields) + list(BreachMeasures._fields)
        data = np.column_stack([study.screw_results, study.breach_results])
        df = pd.DataFrame(data=data, index=screw_names, columns=col_names)
        os.makedirs(acc_dir, exist_ok=True)
        df.to_csv(join(acc_dir, 'results.csv'))

        # Build accuracy summary
        per_screw = {}
        n_breached = 0
        for ii, screw in enumerate(study.screws):
            if screw.type == 'skip':
                continue
            entry_err = np.sqrt(study.screw_results[ii, 0] ** 2 + study.screw_results[ii, 2] ** 2)
            tip_err = np.sqrt(study.screw_results[ii, 5] ** 2 + study.screw_results[ii, 7] ** 2)
            theta = study.screw_results[ii, 8]
            breach_dist = study.breach_results[ii, 0]
            breach_angle = study.breach_results[ii, 1]
            if breach_dist > 0:
                n_breached += 1
            per_screw[screw.name] = {
                'entry_error_mm': float(entry_err),
                'tip_error_mm': float(tip_err),
                'theta_deg': float(theta),
                'breach_dist_mm': float(breach_dist),
                'breach_angle_deg': float(breach_angle),
            }

        placed = [s for s in study.screws if s.type != 'skip']
        placed_idx = [ii for ii, s in enumerate(study.screws) if s.type != 'skip']
        entry_errs = [per_screw[s.name]['entry_error_mm'] for s in placed]
        tip_errs = [per_screw[s.name]['tip_error_mm'] for s in placed]
        thetas = [per_screw[s.name]['theta_deg'] for s in placed]

        ped_ml = study.screw_results[placed_idx, 3]  # ped_x = M-L
        ped_si = study.screw_results[placed_idx, 4]  # ped_z = S-I

        # Generate breach figures
        with timed('breach_figures', timings):
            from spinescrews.figures.visualize_breach import generate_breach_figure
            for screw in placed:
                level = screw.level
                side = screw.name[-1]
                generate_breach_figure(analysis_dir, level, side)
        log.info('    generated %d breach figures', len(placed))

        elapsed = round(time() - t0, 1)
        log.info('Accuracy complete: %d screws, %d breached, %.1fs total',
                 len(placed), n_breached, elapsed)
        ml_mean, ml_rmse, ml_max = np.mean(ped_ml), np.sqrt(np.mean(ped_ml**2)), np.max(np.abs(ped_ml))
        si_mean, si_rmse, si_max = np.mean(ped_si), np.sqrt(np.mean(ped_si**2)), np.max(np.abs(ped_si))
        log.info('  mid-pedicle  M-L: mean=%.2f  RMSE=%.2f  |max|=%.2f mm'
                 '    S-I: mean=%.2f  RMSE=%.2f  |max|=%.2f mm',
                 ml_mean, ml_rmse, ml_max, si_mean, si_rmse, si_max)
        write_summary(acc_dir, {
            'n_screws': len(placed),
            'n_breached': n_breached,
            'per_screw': per_screw,
            'mean_entry_error_mm': float(np.nanmean(entry_errs)),
            'mean_tip_error_mm': float(np.nanmean(tip_errs)),
            'mean_theta_deg': float(np.nanmean(thetas)),
            'elapsed_s': elapsed,
            'timings': timings,
        })

    else:
        log.info('*** Step 07 (accuracy) already complete, skipping')


if __name__ == '__main__':
    main()

