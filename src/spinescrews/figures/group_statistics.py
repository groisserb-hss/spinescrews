import logging
import os
import re
import argparse
import numpy as np
import pandas as pd
import yaml

from spinescrews.figures._plot_helpers import plot_error_clouds, plot_error_bundles, plot_3d_error_clouds
from spinescrews.tools import possible_levels
from spinescrews.tools.screw_models import parse_preop_plan, SkipScrew

import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

log = logging.getLogger(__name__)

COLORS = [
    [0, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840],
]


def _expand_level_range(start, end):
    """Return set of levels from *start* (inferior) to *end* (superior), inclusive."""
    try:
        i = possible_levels.index(start)
    except ValueError:
        raise ValueError(f"unknown level '{start}'")
    try:
        j = possible_levels.index(end)
    except ValueError:
        raise ValueError(f"unknown level '{end}'")
    if i > j:
        raise ValueError(f"'{start}' is superior to '{end}'; expected [inferior-superior]")
    return set(possible_levels[i:j + 1])


def _extract_level(screw_name):
    """'L5L' -> 'L5', 'T10R' -> 'T10', 'LSL' -> 'LS'."""
    return screw_name[:-1]


def _read_exclude_screws(specimen_dir):
    """Read exclude-screws list from specimen config.yml, if present."""
    config_path = os.path.join(specimen_dir, 'config.yml')
    if not os.path.isfile(config_path):
        return set()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return set(raw.get('exclude-screws', []))


def _load_results(group_dict, working_dir):
    """Load results.csv for every specimen, return {group_name: DataFrame}.

    Each value in *group_dict* is a list of ``(specimen_id, levels)`` tuples
    where *levels* is either a set of level strings or ``None`` (all levels).
    """
    out = {}
    for name, specimens in group_dict.items():
        frames = []
        for ID, levels in specimens:
            subj = f'specimen_{ID}'
            specimen_dir = os.path.join(working_dir, subj)
            results_file = os.path.join(specimen_dir, 'analysis', '07_accuracy', 'results.csv')
            df = pd.read_csv(results_file, index_col=0)
            if levels is not None:
                mask = [_extract_level(n) in levels for n in df.index]
                df = df.loc[mask]
            # drop skip screws (all-NaN measurement rows)
            df = df.dropna(subset=['entry_x'])
            # drop per-specimen excluded screws
            excluded = _read_exclude_screws(specimen_dir)
            if excluded:
                df = df.loc[~df.index.isin(excluded)]
            # verify no NaNs remain in measurement columns
            measure_cols = ['entry_x', 'entry_y', 'entry_z',
                            'tip_x', 'tip_y', 'tip_z',
                            'ped_x', 'ped_z']
            nans = df[measure_cols].isna()
            if nans.any().any():
                bad = nans.any(axis=1)
                raise ValueError(
                    f'{subj}: unexpected NaN in measurement columns for '
                    f'{list(df.index[bad])}')
            frames.append(df)
        out[name] = pd.concat(frames, ignore_index=True)
    return out


def _specimen_manifest(specimens):
    """Build human-readable specimen list: '07, 10, 02[L2-T5]'."""
    parts = []
    for sid, levels in specimens:
        if levels is None:
            parts.append(sid)
        else:
            ordered = [l for l in possible_levels if l in levels]
            parts.append(f"{sid}[{ordered[0]}-{ordered[-1]}]")
    return ', '.join(parts)


def compute_group_statistics(group_dict, out_dir, working_dir):
    """Compute per-axis error statistics at entry/mid-pedicle/tip and save CSV report."""
    os.makedirs(out_dir, exist_ok=True)
    results_by_group = _load_results(group_dict, working_dir)
    group_names = list(group_dict.keys())

    locations = [('entry',       [('x', 'entry_x'), ('y', 'entry_y'), ('z', 'entry_z')]),
                 ('mid-pedicle', [('x', 'ped_x'),   ('z', 'ped_z')]),
                 ('tip',         [('x', 'tip_x'),   ('y', 'tip_y'),   ('z', 'tip_z')])]

    rows = []
    for name in group_names:
        df = results_by_group[name]
        manifest = _specimen_manifest(group_dict[name])
        for loc_name, axes in locations:
            for axis_label, col in axes:
                vals = df[col].values
                n = len(vals)
                rows.append({
                    'group': name,
                    'specimens': manifest,
                    'location': loc_name,
                    'axis': axis_label,
                    'n': n,
                    'signed_mean': np.mean(vals),
                    'std': np.std(vals, ddof=1),
                    'mae': np.mean(np.abs(vals)),
                    'rmse': np.sqrt(np.mean(vals**2)),
                    'min': np.min(vals),
                    'max': np.max(vals),
                })

    report = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'statistics.csv')
    report.to_csv(csv_path, index=False, float_format='%.3f')
    log.info('Saved statistics to %s', csv_path)


def plot_target_groups(group_dict: dict, out_dir, working_dir, scale=None):
    """
    Generates scatter plots for target groups.

    Parameters
    ----------
    group_dict : dict
        ``{group_name: [(specimen_id, levels | None), ...]}``.
    out_dir : str
        Output directory to save the plots.
    working_dir : str
        Base directory containing specimen folders.
    scale : int, optional
        Scale to be used for plotting (default is None, auto-computed).
    """
    os.makedirs(out_dir, exist_ok=True)

    results_by_group = _load_results(group_dict, working_dir)
    group_names = list(group_dict.keys())
    num_groups = len(group_names)

    grouped_tails = []
    grouped_tips = []
    grouped_peds = []

    for name in group_names:
        df = results_by_group[name]
        grouped_tails.append(df[['entry_x', 'entry_z']].values)
        grouped_tips.append(df[['tip_x', 'tip_z']].values)
        grouped_peds.append(df[['ped_x', 'ped_z']].values)

    for name in group_names:
        df = results_by_group[name]
        for label, cols in [('entry', ['entry_x', 'entry_z']),
                            ('tip', ['tip_x', 'tip_z']),
                            ('pedicle', ['ped_x', 'ped_z'])]:
            pts = df[cols].values
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            log.info('%s / %s (n=%d): mean=%.2f  std=%.2f  max=%.2f mm',
                     name, label, len(r), np.mean(r), np.std(r), np.max(r))

    plt.close('all')
    colors = COLORS[:num_groups]
    symbols = ['o'] * num_groups

    if scale is None:
        tail_scale = max(np.nanmax(np.abs(tails)) for tails in grouped_tails)
        ped_scale = max(np.nanmax(np.abs(peds)) for peds in grouped_peds)
        tip_scale = max(np.nanmax(np.abs(tips)) for tips in grouped_tips)
        scale = int(np.ceil(max([tail_scale, ped_scale, tip_scale])))
        scale = max(scale, 5)

    plot_error_clouds(os.path.join(out_dir, 'tail'), group_names, grouped_tails, colors, symbols, scale)
    plot_error_clouds(os.path.join(out_dir, 'ped'), group_names, grouped_peds, colors, symbols, scale)
    plot_error_clouds(os.path.join(out_dir, 'tip'), group_names, grouped_tips, colors, symbols, scale)
    log.info('Saved scatter plots to %s', out_dir)


def plot_bundled_groups(group_dict, out_dir, working_dir):
    """
    Generates bundle plots (entry-to-tip lines) for target groups.

    Parameters
    ----------
    group_dict : dict
        ``{group_name: [(specimen_id, levels | None), ...]}``.
    out_dir : str
        Output directory to save the plots.
    working_dir : str
        Base directory containing specimen folders.
    """
    os.makedirs(out_dir, exist_ok=True)

    group_names = list(group_dict.keys())
    num_groups = len(group_names)

    grouped_tails = []
    grouped_tips = []

    for name, specimens in group_dict.items():
        log.info(f'Group {name}: {specimens}')
        tail_chunks, tip_chunks = [], []

        for ID, levels in specimens:
            subj = f'specimen_{ID}'
            log.info(f'  Subject: {subj}')

            results_file = os.path.join(working_dir, subj, 'analysis', '07_accuracy', 'results.csv')
            results = pd.read_csv(results_file, index_col=0)

            # get screw lengths from plan
            screws = parse_preop_plan(os.path.join(working_dir, subj, 'preop_plan.csv'))[1]

            if levels is not None:
                row_mask = [_extract_level(n) in levels for n in results.index]
                results = results.loc[row_mask]
                screws = [s for s in screws if s.level in levels]

            # filter skip screws and per-specimen excluded screws
            specimen_dir = os.path.join(working_dir, subj)
            excluded = _read_exclude_screws(specimen_dir)
            screws = [s for s in screws
                       if not isinstance(s, SkipScrew) and s.name not in excluded]
            keep = {s.name for s in screws}
            results = results.loc[results.index.isin(keep)]

            y_target = np.array([s.shaft_len for s in screws])

            new_tails = np.full([len(y_target), 3], np.nan)
            xyz = results[['entry_x', 'entry_y', 'entry_z']].values
            new_tails[:len(xyz)] = xyz
            new_tails[:, 1] -= y_target / 2

            new_tips = np.full([len(y_target), 3], np.nan)
            xyz = results[['tip_x', 'tip_y', 'tip_z']].values
            new_tips[:len(xyz)] = xyz
            new_tips[:, 1] += y_target / 2

            tail_chunks.append(new_tails)
            tip_chunks.append(new_tips)

        grouped_tails.append(np.concatenate(tail_chunks))
        grouped_tips.append(np.concatenate(tip_chunks))

    plt.close('all')
    colors = COLORS[:num_groups]
    symbols = ['o'] * num_groups

    axial_entry = [g[:, [1, 0]] for g in grouped_tails]
    axial_tip = [g[:, [1, 0]] for g in grouped_tips]
    plot_error_bundles(os.path.join(out_dir, 'axial'), group_names, axial_entry, axial_tip, colors, symbols)

    sagittal_entry = [g[:, [1, 2]] for g in grouped_tails]
    sagittal_tip = [g[:, [1, 2]] for g in grouped_tips]
    plot_error_bundles(os.path.join(out_dir, 'sagittal'), group_names, sagittal_entry, sagittal_tip, colors, symbols)

    plot_3d_error_clouds(os.path.join(out_dir, '3d'), group_names, grouped_tails, grouped_tips, colors)
    log.info('Saved bundle plots to %s', out_dir)


# -- CLI helpers ---------------------------------------------------------------

_SPEC_RE = re.compile(r'^([^[]+?)(?:\[([A-Z][A-Za-z0-9]+)-([A-Z][A-Za-z0-9]+)\])?$')


def _parse_specimen(token):
    """Parse '02[L2-T5]' -> ('02', {levels}) or '07' -> ('07', None)."""
    m = _SPEC_RE.match(token.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad specimen spec '{token}'")
    sid, lo, hi = m.group(1), m.group(2), m.group(3)
    if lo is not None:
        return sid, _expand_level_range(lo, hi)
    return sid, None


def _parse_group(s):
    """Parse 'name:id1,id2[L2-T5],...' into (name, [(id, levels|None), ...])."""
    name, _, body = s.partition(':')
    if not body:
        raise argparse.ArgumentTypeError(f"expected 'name:id1,id2,...', got '{s}'")
    return name, [_parse_specimen(tok) for tok in body.split(',')]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')

    parser = argparse.ArgumentParser(
        description="Group statistics and scatter plots",
        epilog="example: python -m spinescrews.figures.group_statistics "
               "--base_dir ~/Documents/ScrewAccuracyCT "
               "--out_dir results --group 'mazor:07,10,02[L2-T5]' --group 'freehand:04,25'",
    )
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing specimen folders")
    parser.add_argument("--group", type=_parse_group, action="append", required=True,
                        help="Group as 'name:id1,id2[Lo-Hi],...' (repeatable)")
    parser.add_argument('--scale', type=float, default=None, help='Scale for plotting')
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save the plots")
    args = parser.parse_args()

    working_dir = os.path.expanduser(args.base_dir)
    group_dict = {name: specs for name, specs in args.group}
    out_dir = os.path.join(working_dir, 'results3', args.out_dir)

    compute_group_statistics(group_dict, out_dir, working_dir)
    plot_target_groups(group_dict, out_dir, working_dir, scale=args.scale)
    plot_bundled_groups(group_dict, out_dir, working_dir)
