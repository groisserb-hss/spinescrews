"""
Multi-backend vertebral segmentation wrapper.

Supports:
  - totalseg: TotalSegmentator (Apache-2.0, pip-installable)
  - inria: Inria vertebrae_segmentation (CC-BY-NC-SA-4.0, separate env)

Usage:
    python -m spinescrews.pipeline.run_segmentation --backend totalseg --input preop.nii.gz --output_dir specimen_01/
"""

import sys
import os
from os.path import join, expanduser, isdir, isfile, basename, dirname, abspath
import glob
import subprocess
import logging

import numpy as np
import nibabel as nib

from spinescrews.tools import val_seg
from spinescrews.tools.totalseg_segmentor import run_totalseg

log = logging.getLogger(__name__)

def _run_inria(input_path, output_dir, inria_repo=None, inria_env='verse20'):
    """Run Inria/SPINE vertebrae_segmentation via subprocess."""


    # default repo location: spinescrews/tools/inria_segmentor/vertebrae_segmentation/
    if inria_repo is None:
        tools_dir = join(dirname(dirname(abspath(__file__))), 'tools')
        inria_repo = join(tools_dir, 'inria_segmentor', 'vertebrae_segmentation')
    inria_repo = expanduser(inria_repo)

    if not isdir(inria_repo):
        raise FileNotFoundError(
            'Inria segmentor repo not found at %s\n'
            'Run: bash tools/inria_segmentor/setup.sh' % inria_repo
        )

    test_script = join(inria_repo, 'test.py')
    if not isfile(test_script):
        raise FileNotFoundError(
            'test.py not found in inria repo: %s' % inria_repo
        )

    # verify conda is available
    conda_path = subprocess.run(
        ['which', 'conda'], capture_output=True, text=True
    )
    if conda_path.returncode != 0:
        raise RuntimeError('conda not found on PATH')

    cmd = [
        'conda', 'run', '-n', inria_env,
        'python', test_script, '-D', str(input_path), '-S', str(output_dir),
    ]
    log.info('running: %s', ' '.join(cmd))

    # cwd must be the repo dir — test.py uses relative paths to models/
    # PYTHONPATH must include the repo dir — conda run can strip the script
    # directory from sys.path, breaking bare imports (consistency_loop, utils, etc.)
    env = os.environ.copy()
    env['PYTHONPATH'] = inria_repo + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=inria_repo,
                            env=env)
    if result.returncode != 0:
        log.error('inria segmentation failed (rc=%d):\n%s',
                  result.returncode, result.stderr)
        raise RuntimeError('Inria segmentation failed — see log for details')
    if result.stderr:
        log.debug('inria stderr:\n%s', result.stderr)

    # Inria outputs {scanname}_seg.nii.gz, not preop_seg.nii.gz
    # Find the output and rename it
    seg_path = join(output_dir, 'preop_seg.nii.gz')
    if not isfile(seg_path):
        seg_candidates = glob.glob(join(output_dir, '*_seg.nii.gz'))
        if len(seg_candidates) == 1:
            log.info('renaming %s -> preop_seg.nii.gz', basename(seg_candidates[0]))
            os.rename(seg_candidates[0], seg_path)
        elif len(seg_candidates) > 1:
            # pick the one matching input basename
            input_stem = basename(input_path).replace('.nii.gz', '').replace('.nii', '')
            expected = join(output_dir, input_stem + '_seg.nii.gz')
            if isfile(expected):
                log.info('renaming %s -> preop_seg.nii.gz', basename(expected))
                os.rename(expected, seg_path)
            else:
                raise FileNotFoundError(
                    'Multiple *_seg.nii.gz found but none match input: %s'
                    % seg_candidates
                )
        else:
            raise FileNotFoundError(
                'Inria did not produce expected output in %s' % output_dir
            )

    seg_img = nib.load(seg_path)
    log.info('loaded inria segmentation from %s', seg_path)
    return seg_img


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
BACKENDS = {
    'totalseg': run_totalseg,
    'inria': _run_inria,
}


def run_segmentation(input_path, output_dir, backend='totalseg', **kwargs):
    """Run vertebral segmentation and save to output_dir/preop_seg.nii.gz.

    Parameters
    ----------
    input_path : str
        Path to input NIfTI volume.
    output_dir : str
        Directory for output files.
    backend : str
        Backend name (key in BACKENDS).
    **kwargs
        Passed to the backend function.

    Returns
    -------
    str
        Path to the saved segmentation file.
    """
    if backend not in BACKENDS:
        raise ValueError('unknown backend %r — choose from: %s'
                         % (backend, ', '.join(BACKENDS)))

    if not isfile(input_path):
        raise FileNotFoundError('input not found: %s' % input_path)
    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    log.info('backend=%s, input=%s, output_dir=%s', backend, input_path, output_dir)
    seg_img = BACKENDS[backend](input_path, output_dir, **kwargs)

    # validate: all nonzero labels must be in val_seg
    seg_data = np.asarray(seg_img.dataobj)
    labels_present = set(np.unique(seg_data)) - {0}
    unknown = labels_present - set(val_seg.keys())
    if unknown:
        log.warning('unknown labels in segmentation output (removing): %s', unknown)
        mask = np.isin(seg_data, list(unknown))
        seg_data = seg_data.copy()
        seg_data[mask] = 0
        seg_img = nib.Nifti1Image(seg_data.astype(np.uint8), seg_img.affine, seg_img.header)

    output_path = join(output_dir, 'preop_seg.nii.gz')
    nib.save(seg_img, output_path)
    log.info('saved segmentation to %s', output_path)

    # Build segmentation summary info
    seg_data = np.asarray(seg_img.dataobj)
    voxel_vol = float(np.abs(np.linalg.det(seg_img.affine[:3, :3])))
    per_level = {}
    levels_found = []
    for label_int in sorted(labels_present):
        if label_int not in val_seg:
            continue
        name = val_seg[label_int]
        voxels = int(np.sum(seg_data == label_int))
        per_level[name] = {
            'voxels': voxels,
            'volume_mm3': round(voxels * voxel_vol, 1),
        }
        levels_found.append(name)

    summary = {
        'levels_found': levels_found,
        'n_levels': len(levels_found),
        'per_level': per_level,
        'backend': backend,
    }

    return output_path, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    """CLI entry point for vertebral segmentation. Called by spinescrews-segment console script."""
    import argparse
    import yaml as _yaml

    parser = argparse.ArgumentParser(
        description='Vertebral segmentation (step 01): label the vertebrae in a CT volume with a '
                    'selectable backend (TotalSegmentator by default, or Inria). Writes '
                    'preop_seg.nii.gz to the output directory.')
    parser.add_argument('--input', required=True, type=str,
                        help='Input NIfTI volume (e.g. preop.nii.gz)')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Output directory for preop_seg.nii.gz')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for segmentation settings')
    parser.add_argument('--backend', type=str, default=None,
                        choices=list(BACKENDS.keys()),
                        help='Segmentation backend (default: totalseg)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'gpu', 'mps'],
                        help='Compute device for totalseg (default: cpu)')
    parser.add_argument('--fast', action='store_true', default=None,
                        help='Use fast mode for totalseg (lower resolution)')
    parser.add_argument('--inria_repo', type=str, default=None,
                        help='Path to Inria/SPINE vertebrae_segmentation repo')
    parser.add_argument('--inria_env', type=str, default=None,
                        help='Conda env for Inria segmentor (default: verse20)')
    parser.add_argument('--debug', action='store_true',
                        help='Verbose debug logging.')
    args = parser.parse_args()

    # load config defaults from YAML if provided
    cfg_backend = 'totalseg'
    cfg_device = 'cpu'
    cfg_fast = False
    cfg_inria_repo = None
    cfg_inria_env = 'verse20'

    if args.config and isfile(expanduser(args.config)):
        with open(expanduser(args.config), 'r') as _f:
            _raw = _yaml.safe_load(_f) or {}
        _seg = _raw.get('segmentation', _raw)
        cfg_backend = _seg.get('backend', cfg_backend)
        cfg_device = _seg.get('device', cfg_device)
        cfg_fast = _seg.get('fast', cfg_fast)
        cfg_inria_repo = _seg.get('inria_repo', cfg_inria_repo)
        cfg_inria_env = _seg.get('inria_env', cfg_inria_env)

    # CLI args override config values when provided
    backend = args.backend if args.backend is not None else cfg_backend
    device = args.device if args.device is not None else cfg_device
    fast = args.fast if args.fast is not None else cfg_fast
    inria_repo = args.inria_repo if args.inria_repo is not None else cfg_inria_repo
    inria_env = args.inria_env if args.inria_env is not None else cfg_inria_env

    output_dir = expanduser(args.output_dir)
    logfile = join(output_dir, 'segmentation.log')
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])

    log.info('=== Vertebral segmentation ===')
    log.info('backend: %s', backend)

    # build kwargs for the selected backend
    kwargs = {}
    if backend == 'totalseg':
        kwargs = dict(device=device, fast=fast)
    elif backend == 'inria':
        kwargs = dict(inria_repo=inria_repo, inria_env=inria_env)

    output_path, summary = run_segmentation(
        input_path=expanduser(args.input),
        output_dir=output_dir,
        backend=backend,
        **kwargs,
    )

    # Move segmentation into 01_segmentation/ and write summary gate file
    from spinescrews.tools.paths import segmentation_dir, segmentation_file, write_summary
    if isfile(join(output_dir, 'preop_seg.nii.gz')):
        analysis = join(output_dir, 'analysis')
        os.makedirs(analysis, exist_ok=True)
        seg_step_dir = segmentation_dir(analysis)
        os.makedirs(seg_step_dir, exist_ok=True)

        # move file from specimen root → 01_segmentation/
        final_path = segmentation_file(analysis)
        if not isfile(final_path):
            os.rename(output_path, final_path)
            output_path = final_path

        summary['params'] = {'device': device, 'fast': fast} if backend == 'totalseg' else {}
        write_summary(seg_step_dir, summary)

    log.info('done — output: %s', output_path)


if __name__ == '__main__':
    main()
