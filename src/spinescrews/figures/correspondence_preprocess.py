"""Correspondence preprocessing figure: 4-panel overhead geodesic distance visualization."""

import os
import sys
import logging
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bg3dtools.render.o3d import trisurfsm, render_mesh_to_image, overhead_camera
from spinescrews.tools.paths import correspondence_level_dir

log = logging.getLogger(__name__)


def generate_preprocess_figure(mesh, level_name, output_path):
    """4-panel overhead geodesic/eigen visualization.

    Parameters
    ----------
    mesh : spectral_match.tools.mesh_class.Mesh
        Preprocessed bone mesh (must have geodesic matrix computed).
    level_name : str
        Vertebra level name (e.g. 'T11').
    output_path : str
        Path to save the figure (e.g. .../03_correspondence/T11/preprocess.png).
    """
    from spectral_match.tools.geometric_utilities import metric_sampling

    v, f = mesh.v, mesh.f
    g = mesh.g
    lookat, eye, up = overhead_camera(v)

    # Show geodesic distance from 4 farthest-point-sampling seeds.
    # This is always available since preprocessing computes the geodesic matrix.
    seeds = metric_sampling(g, 4)

    panels = []
    titles = []
    for i, seed in enumerate(seeds):
        scalar = g[seed]
        o3d_mesh = trisurfsm(v, f, colors=scalar, render=False, colormap='parula')
        img = render_mesh_to_image(o3d_mesh, lookat, eye, up)
        panels.append(img)
        titles.append('FPS seed %d (v%d)' % (i, seed))

    # composite with matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('%s — preprocessing' % level_name, fontsize=12)
    for ax, img, title in zip(axs, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.debug('Saved preprocessing figure to %s', output_path)


if __name__ == '__main__':
    import argparse
    from spectral_match.pipeline import Mesh

    parser = argparse.ArgumentParser(description='Generate correspondence preprocessing figure.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--level', type=str, required=True,
                        help='Vertebra level (e.g. T11)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    analysis_dir = os.path.join(specimen, 'analysis')
    corr_dir = correspondence_level_dir(analysis_dir, args.level)
    npz_file = os.path.join(corr_dir, 'bone_preprocess.npz')

    if not os.path.isfile(npz_file):
        log.error('bone_preprocess.npz not found at %s' % npz_file)
        sys.exit(1)

    mesh = Mesh.from_file(npz_file)
    output_path = os.path.join(corr_dir, 'preprocess.png')
    generate_preprocess_figure(mesh, args.level, output_path)
