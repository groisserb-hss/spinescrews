"""Spine template construct visualization: template + seg meshes in world space."""

import os
import logging
import numpy as np
import igl
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bg3dtools.render.o3d import trisurfsm, scatt, anterior_camera
from bg3dtools.render import render_frame, CameraParams, RenderStyle, _RawGeom, RenderUnavailable
from spinescrews.tools import possible_levels
from spinescrews.tools.paths import preop_dir, preop_level_dir, orient_dir, orient_level_dir

log = logging.getLogger(__name__)

_SEG_COLOR = np.array([0.85, 0.85, 0.85])


def _render_scene(geoms, lookat, eye, up, w, h, fov=45.0):
    """Render a list of (geometry, shader) tuples to an RGB uint8 image.

    Delegates to the unified renderer (``bg3dtools.render.scan.render_frame``),
    which owns the single offscreen→legacy→RenderUnavailable fallback chain, so
    this figure shares one render implementation with the rest of the codebase.

    Parameters
    ----------
    geoms : list of (o3d.geometry, str) tuples
        Each entry is (geometry, shader_name). Meshes use ``"defaultLit"``,
        point clouds use ``"defaultUnlit"``.
    lookat, eye, up : array-like
        Camera parameters.
    w, h : int
        Image width and height.
    fov : float
        Field of view in degrees.

    Returns
    -------
    np.ndarray
        RGB uint8 image.

    Raises
    ------
    RenderUnavailable
        If no Open3D backend (offscreen or legacy) is usable on this host.
    """
    specs = [_RawGeom(geom, hint=('point' if shader == 'defaultUnlit' else 'mesh'))
             for geom, shader in geoms]
    cam = CameraParams(lookat=np.asarray(lookat, np.float32),
                       eye=np.asarray(eye, np.float32),
                       up=np.asarray(up, np.float32), fov=float(fov))
    return render_frame(specs, cam, width=w, height=h,
                        style=RenderStyle(point_size=3.0))


def _render_two_panel(geoms, all_verts, title, legend_patches, out_path):
    """Render anterior + lateral views and save a 2-panel figure.

    Parameters
    ----------
    geoms : list of (o3d.geometry, str)
        Geometry + shader tuples for ``_render_scene``.
    all_verts : np.ndarray
        Combined vertex array for camera computation.
    title : str
        Figure suptitle.
    legend_patches : list of Patch or None
        Matplotlib legend handles; omitted if None.
    out_path : str
        Output PNG path.
    """
    W, H = 600, 800

    # Anterior view
    lookat, eye, up = anterior_camera(all_verts)
    img_ant = _render_scene(geoms, lookat, eye, up, W, H)

    # Right lateral view
    center = all_verts.mean(axis=0)
    extent = np.ptp(all_verts, axis=0)
    eye_lat = center + np.array([extent[0] * 2, 0, 0])
    up_lat = np.array([0, 0, 1])
    img_lat = _render_scene(geoms, center, eye_lat, up_lat, W, H)

    # Compose figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    ax1.imshow(img_ant)
    ax1.set_title('Anterior')
    ax1.axis('off')
    ax2.imshow(img_lat)
    ax2.set_title('Right Lateral')
    ax2.axis('off')

    if legend_patches:
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=min(10, len(legend_patches)), fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_spine_construct(analysis_dir, template_dir, step='preop', dead_backends=()):
    """Build full-spine construct, save PLY and render 3 PNGs.

    Produces three 2-panel (anterior + lateral) figures:
    - ``spine_seg.png`` — seg meshes only (solid gray)
    - ``spine_template.png`` — template meshes only (colored per level)
    - ``spine_overlay.png`` — seg solid + template point cloud overlay

    Parameters
    ----------
    analysis_dir : str
        Path to analysis/ directory.
    template_dir : str
        Path to vertebra_templates/ directory.
    step : str
        'preop' uses 02_preop/{LEVEL}/preop_affine.npy,
        'orient' uses 04_orient/{LEVEL}/preop_affine-refined.npy.
    dead_backends : iterable of str
        Render-backend names the parent process already found unusable (from
        ``spinescrews.figures.probe_render_backends``). This function runs in a
        ``run_isolated`` subprocess; seeding them here lets it skip those backends
        instead of re-probing, so Open3D's native "EGL Headless" error isn't
        reprinted per subprocess.
    """
    from spinescrews.figures import seed_dead_backends
    seed_dead_backends(dead_backends)

    # Discover levels with preop_gen1.ply, ordered by possible_levels
    levels = []
    for level in possible_levels:
        gen1_file = os.path.join(preop_level_dir(analysis_dir, level), 'preop_gen1.ply')
        if os.path.isfile(gen1_file):
            levels.append(level)

    if not levels:
        log.warning('No normalized vertebrae found, skipping spine template figure')
        return

    # Per-level template colors
    n = len(levels)
    level_colors = {}
    for i, level in enumerate(levels):
        level_colors[level] = np.array(plt.cm.tab20(i / max(1, n - 1))[:3])

    # Collect per-level data
    seg_data = []     # list of (v_world, f)
    tmpl_data = []    # list of (v_world, f, level)
    seg_v_export, seg_f_export = [], []    # for PLY export
    seg_face_offset = 0
    tmpl_v_export, tmpl_f_export = [], []  # for PLY export
    tmpl_face_offset = 0

    for level in levels:
        # Seg mesh always uses original affine (true anatomical position)
        orig_aff_file = os.path.join(preop_level_dir(analysis_dir, level),
                                     'preop_affine.npy')
        if not os.path.isfile(orig_aff_file):
            log.warning('No affine for %s, skipping', level)
            continue

        orig_aff = np.load(orig_aff_file)
        R_orig, t_orig = orig_aff[:3, :3], orig_aff[:3, 3]

        # Load and transform seg mesh (gen1) — lives in step 02 normalized space
        gen1_file = os.path.join(preop_level_dir(analysis_dir, level), 'preop_gen1.ply')
        v_seg, f_seg = igl.read_triangle_mesh(gen1_file)
        v_seg_world = (R_orig @ v_seg.T).T + t_orig
        seg_data.append((v_seg_world, f_seg))
        seg_v_export.append(v_seg_world)
        seg_f_export.append(f_seg + seg_face_offset)
        seg_face_offset += len(v_seg_world)

        # Template mesh uses refined affine when available
        if step == 'orient':
            refined_file = os.path.join(orient_level_dir(analysis_dir, level),
                                        'preop_affine-refined.npy')
            tmpl_aff = np.load(refined_file) if os.path.isfile(refined_file) else orig_aff
        else:
            tmpl_aff = orig_aff
        R_tmpl, t_tmpl = tmpl_aff[:3, :3], tmpl_aff[:3, 3]

        # Load and transform template mesh
        tmpl_file = os.path.join(template_dir, 'meshes', 'template_%s.ply' % level)
        if not os.path.isfile(tmpl_file):
            continue

        v_tmpl, f_tmpl = igl.read_triangle_mesh(tmpl_file)

        # Scale template to match seg mesh size
        scale_file = os.path.join(orient_level_dir(analysis_dir, level),
                                  'template_scale.npy')
        if step == 'orient' and os.path.isfile(scale_file):
            scale = float(np.load(scale_file))
        else:
            seg_diag = np.linalg.norm(np.ptp(v_seg, axis=0))
            tmpl_diag = np.linalg.norm(np.ptp(v_tmpl, axis=0))
            scale = seg_diag / tmpl_diag if tmpl_diag > 1e-9 else 1.0

        v_tmpl_scaled = (v_tmpl - v_tmpl.mean(axis=0)) * scale + v_seg.mean(axis=0)
        v_tmpl_world = (R_tmpl @ v_tmpl_scaled.T).T + t_tmpl
        tmpl_data.append((v_tmpl_world, f_tmpl, level))

        tmpl_v_export.append(v_tmpl_world)
        tmpl_f_export.append(f_tmpl + tmpl_face_offset)
        tmpl_face_offset += len(v_tmpl_world)

    if not seg_data:
        log.warning('No meshes collected, skipping spine template figure')
        return

    # Determine output directory
    if step == 'orient':
        step_dir = orient_dir(analysis_dir)
    else:
        step_dir = preop_dir(analysis_dir)
    os.makedirs(step_dir, exist_ok=True)

    # Save seg-only PLY
    if seg_v_export:
        sv = np.vstack(seg_v_export)
        sf = np.vstack(seg_f_export)
        igl.write_triangle_mesh(os.path.join(step_dir, 'spine_seg.ply'), sv, sf)
        log.info('Saved spine_seg.ply to %s (%d verts)', step_dir, len(sv))

    # Save template-only PLY
    if tmpl_v_export:
        tv = np.vstack(tmpl_v_export)
        tf = np.vstack(tmpl_f_export)
        igl.write_triangle_mesh(os.path.join(step_dir, 'spine_template.ply'), tv, tf)
        log.info('Saved spine_template.ply to %s (%d verts)', step_dir, len(tv))

    # Build O3D geometry objects
    seg_geoms = []
    for v, f in seg_data:
        mesh = trisurfsm(v, f, colors=_SEG_COLOR, render=False)
        seg_geoms.append((mesh, "defaultLit"))

    tmpl_mesh_geoms = []
    tmpl_pc_geoms = []
    for v, f, level in tmpl_data:
        color = level_colors[level]
        mesh = trisurfsm(v, f, colors=color, render=False)
        tmpl_mesh_geoms.append((mesh, "defaultLit"))
        pc = scatt(v, colors=color, render=False)
        tmpl_pc_geoms.append((pc, "defaultUnlit"))

    # Combined vertex array for camera positioning
    all_verts = np.vstack([v for v, f in seg_data] +
                          [v for v, f, l in tmpl_data])

    step_label = 'Step 02 (preop)' if step == 'preop' else 'Step 04 (orient)'

    # Legend patches (level colors only, used for template & overlay)
    tmpl_patches = [Patch(color=level_colors[l], label=l)
                    for l in levels if l in level_colors]

    # Render diagnostic PNGs. If no Open3D backend is usable (headless host with
    # neither EGL nor a display), skip the images — the PLYs above are the
    # durable output and are already on disk.
    try:
        # 1. Seg only
        _render_two_panel(
            seg_geoms, all_verts,
            'Seg meshes — %s' % step_label,
            None,
            os.path.join(step_dir, 'spine_seg.png'))
        log.info('Saved spine_seg.png to %s', step_dir)

        # 2. Template only
        if tmpl_mesh_geoms:
            _render_two_panel(
                tmpl_mesh_geoms, all_verts,
                'Template meshes — %s' % step_label,
                tmpl_patches,
                os.path.join(step_dir, 'spine_template.png'))
            log.info('Saved spine_template.png to %s', step_dir)

        # 3. Overlay: seg solid + template point cloud
        if tmpl_pc_geoms:
            overlay_geoms = seg_geoms + tmpl_pc_geoms
            overlay_patches = [Patch(color=_SEG_COLOR, label='seg')] + tmpl_patches
            _render_two_panel(
                overlay_geoms, all_verts,
                'Overlay (seg + template) — %s' % step_label,
                overlay_patches,
                os.path.join(step_dir, 'spine_overlay.png'))
            log.info('Saved spine_overlay.png to %s', step_dir)
    except RenderUnavailable as exc:
        log.warning('spine construct: no Open3D render backend (%s); '
                    'saved PLYs but skipping PNGs', exc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate spine template construct figures.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--step', type=str, default='preop', choices=['preop', 'orient'],
                        help='Which step affines to use (default: preop)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'vertebra_templates')
    generate_spine_construct(os.path.join(specimen, 'analysis'), template_dir, step=args.step)
