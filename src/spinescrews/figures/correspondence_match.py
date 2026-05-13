"""Correspondence match figure: 2x2 label + gradient transfer visualization."""

import os
import sys
import logging
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bg3dtools.render.o3d import trisurfsm, render_mesh_to_image, overhead_camera
from bg3dtools.render.colors import xyz_to_rgb
from bg3dtools.mesh.mesh_io import read_colored_plyfile
from spinescrews.tools.paths import correspondence_level_dir

log = logging.getLogger(__name__)

REGION_COLORS = np.array([
    [0.90, 0.20, 0.20],  # left_ped (red)
    [0.20, 0.40, 0.90],  # right_ped (blue)
    [0.20, 0.85, 0.30],  # canal (green)
    [0.95, 0.75, 0.30],  # body_walls (gold)
    [0.85, 0.45, 0.85],  # endplate_top (magenta)
    [0.55, 0.25, 0.70],  # endplate_bottom (purple)
])
REGION_NAMES = ['left_ped', 'right_ped', 'canal',
                'body_walls', 'endplate_top', 'endplate_bottom']

# Label PLY filenames in the same order as REGION_NAMES
_LABEL_FILES = [
    'left_pedicle.ply', 'right_pedicle.ply', 'canal.ply',
    'body_walls.ply', 'superior_endplate.ply', 'inferior_endplate.ply',
]


def load_template_labels(label_dir):
    """Load 6 anatomical label weights from PLY files.

    Returns (n_template_verts, 6) array of per-vertex weights in [0, 1].
    """
    columns = []
    for fname in _LABEL_FILES:
        v, f, c = read_colored_plyfile(os.path.join(label_dir, fname), vert_colors=True)
        # label data stored as inverted green channel
        weight = (255 - c[:, 1].astype(np.float32)) / 255
        columns.append(weight)
    return np.column_stack(columns)


def labels_to_rgb(weights):
    """Convert (N, 6) label weights to (N, 3) RGB via argmax.

    Vertices where the max weight < 0.1 are colored gray.
    """
    idx = np.argmax(weights, axis=1)
    rgb = REGION_COLORS[idx]
    low = weights.max(axis=1) < 0.1
    rgb[low] = [0.7, 0.7, 0.7]
    return rgb


def generate_match_figure(template_v, template_f, bone_v, bone_f,
                          template2bone, label_dir, level_name, output_path):
    """Generate 2x2 correspondence match figure.

    Panels:
      Template labels (overhead)  |  Bone labels (overhead)
      Template gradient (overhead) |  Bone gradient (overhead)

    Parameters
    ----------
    template_v, template_f : ndarray
        Template mesh (normalized coordinates).
    bone_v, bone_f : ndarray
        Bone mesh (normalized coordinates).
    template2bone : sparse matrix
        (n_bone, n_template) mapping from template to bone vertices.
    label_dir : str
        Path to template label PLY directory.
    level_name : str
        Vertebra level name (e.g. 'T11').
    output_path : str
        Where to save the figure.
    """
    # Load template labels and transfer to bone
    template_weights = load_template_labels(label_dir)
    bone_weights = template2bone @ template_weights

    template_label_rgb = labels_to_rgb(template_weights)
    bone_label_rgb = labels_to_rgb(bone_weights)

    # XYZ gradient: smooth spatial coloring
    template_grad = xyz_to_rgb(template_v)
    bone_grad = template2bone @ template_grad
    # re-normalize after transfer (sparse map can slightly shift range)
    bone_grad = np.clip(bone_grad, 0, 1)

    # Camera setups
    t_lookat, t_eye, t_up = overhead_camera(template_v)
    b_lookat, b_eye, b_up = overhead_camera(bone_v)

    # Render 4 panels
    panels = []
    for v, f, rgb, lookat, eye, up in [
        (template_v, template_f, template_label_rgb, t_lookat, t_eye, t_up),
        (bone_v, bone_f, bone_label_rgb, b_lookat, b_eye, b_up),
        (template_v, template_f, template_grad, t_lookat, t_eye, t_up),
        (bone_v, bone_f, bone_grad, b_lookat, b_eye, b_up),
    ]:
        geom = trisurfsm(v, f, colors=rgb, render=False)
        img = render_mesh_to_image(geom, lookat, eye, up)
        panels.append(img)

    titles = ['Template labels', 'Bone labels',
              'Template gradient', 'Bone gradient']

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('%s — correspondence match' % level_name, fontsize=12)
    for ax, img, title in zip(axs.flat, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.debug('Saved match figure to %s', output_path)


if __name__ == '__main__':
    import argparse
    from scipy import sparse
    from spectral_match.tools.geometric_utilities import normalize_mesh

    parser = argparse.ArgumentParser(description='Generate correspondence match figure.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--level', type=str, required=True,
                        help='Vertebra level (e.g. T11)')
    parser.add_argument('--template-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             'vertebra_templates'),
                        help='Path to vertebra_templates/ directory')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    analysis_dir = os.path.join(specimen, 'analysis')
    corr_dir = correspondence_level_dir(analysis_dir, args.level)

    # Load template mesh (normalized)
    import igl
    template_file = os.path.join(args.template_dir, 'meshes',
                                 'template_%s.ply' % args.level)
    template_v, template_f = igl.read_triangle_mesh(template_file)
    template_v = normalize_mesh(template_v, template_f)[0]

    # Load raw mesh (preop_seg.ply = genus-agnostic raw mesh)
    seg_file = os.path.join(corr_dir, 'preop_seg.ply')
    bone_v, bone_f = igl.read_triangle_mesh(seg_file)
    bone_v = normalize_mesh(bone_v, bone_f)[0]

    # Load composed correspondence matrix
    template2seg = sparse.load_npz(os.path.join(corr_dir, 'template2seg.npz'))

    label_dir = os.path.join(args.template_dir, 'labels', args.level)
    output_path = os.path.join(corr_dir, 'match.png')
    generate_match_figure(template_v, template_f, bone_v, bone_f,
                          template2seg, label_dir, args.level, output_path)
