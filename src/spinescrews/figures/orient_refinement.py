"""Step 04 QC figure: bar charts of orientation refinement corrections."""

import os
import logging
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spinescrews.tools import possible_levels
from spinescrews.tools.paths import orient_dir, orient_level_dir, preop_level_dir, read_summary

log = logging.getLogger(__name__)


def _load_from_summary(analysis_dir):
    """Load per-level metrics from 04_orient/summary.json."""
    summary = read_summary(orient_dir(analysis_dir))
    per_level = summary.get('per_level', {})
    # order by possible_levels
    levels, angles, translations, anchors = [], [], [], []
    for level in possible_levels:
        if level in per_level:
            entry = per_level[level]
            levels.append(level)
            angles.append(entry['angle_deg'])
            translations.append(entry['trans_mm'])
            anchors.append(entry.get('anchor_weight', 1.0))
    return levels, angles, translations, anchors


def _load_from_affines(analysis_dir):
    """Fallback: compute corrections from affine files."""
    from bg3dtools.transforms_unified import R_to_twist

    levels, angles, translations, anchors = [], [], [], []
    for level in possible_levels:
        original = os.path.join(preop_level_dir(analysis_dir, level), 'preop_affine.npy')
        refined = os.path.join(orient_level_dir(analysis_dir, level), 'preop_affine-refined.npy')
        if os.path.isfile(original) and os.path.isfile(refined):
            old_aff = np.load(original)
            new_aff = np.load(refined)
            delta = np.linalg.inv(old_aff) @ new_aff
            angle = np.degrees(np.linalg.norm(R_to_twist(delta[:3, :3])))
            trans = np.linalg.norm(delta[:3, 3])
            levels.append(level)
            angles.append(angle)
            translations.append(trans)
            anchors.append(1.0)
    return levels, angles, translations, anchors


def generate_orient_summary(analysis_dir):
    """Bar charts of orientation refinement. Saves to 04_orient/orient_refinement.png."""

    # try summary.json first, fall back to affine files
    try:
        levels, angles, translations, anchors = _load_from_summary(analysis_dir)
    except (FileNotFoundError, KeyError):
        levels, angles, translations, anchors = _load_from_affines(analysis_dir)

    if not levels:
        log.warning('No orientation refinement data found')
        return

    n = len(levels)
    angles = np.array(angles)
    translations = np.array(translations)
    anchors = np.array(anchors, dtype=float)

    cmap = plt.cm.RdYlBu  # red=0 (low anchor), blue=1 (high anchor)
    colors = [cmap(a) for a in anchors]

    fig, (ax_rot, ax_trans) = plt.subplots(2, 1, figsize=(max(8, n), 6), sharex=True)

    x = np.arange(n)

    # Top: rotation
    ax_rot.bar(x, angles, color=colors)
    median_angle = np.median(angles)
    ax_rot.axhline(median_angle, color='gray', linestyle='--', linewidth=0.8,
                   label='median = %.1f\u00b0' % median_angle)
    ax_rot.set_ylabel('Rotation (\u00b0)')
    ax_rot.legend(fontsize=8)
    if n < 12:
        for i, v in enumerate(angles):
            ax_rot.text(i, v + 0.1, '%.1f' % v, ha='center', va='bottom', fontsize=7)

    # Bottom: translation
    ax_trans.bar(x, translations, color=colors)
    median_trans = np.median(translations)
    ax_trans.axhline(median_trans, color='gray', linestyle='--', linewidth=0.8,
                     label='median = %.1f mm' % median_trans)
    ax_trans.set_ylabel('Translation (mm)')
    ax_trans.set_xticks(x)
    ax_trans.set_xticklabels(levels, rotation=45, ha='right')
    ax_trans.legend(fontsize=8)
    if n < 12:
        for i, v in enumerate(translations):
            ax_trans.text(i, v + 0.05, '%.1f' % v, ha='center', va='bottom', fontsize=7)

    # colorbar for anchor weight
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_rot, ax_trans], pad=0.02, aspect=30)
    cbar.set_label('Anchor weight', fontsize=9)

    fig.suptitle('Step 04: Orientation Refinement', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    step_dir = orient_dir(analysis_dir)
    os.makedirs(step_dir, exist_ok=True)
    out_path = os.path.join(step_dir, 'orient_refinement.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info('Saved orient refinement figure to %s', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate orientation refinement QC figure.')
    parser.add_argument('specimen_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    specimen = os.path.expanduser(args.specimen_dir)
    generate_orient_summary(os.path.join(specimen, 'analysis'))
