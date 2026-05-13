"""TotalSegmentator backend for vertebral segmentation.

Handles label remapping from TotalSegmentator's own integer scheme
to project integers (seg_val / val_seg).

Model weights are stored locally in this directory (via TOTALSEG_HOME_DIR).
"""

import os
import logging
from os.path import dirname, abspath

import numpy as np
import nibabel as nib
from os.path import basename

from spinescrews.tools import seg_val, val_seg

log = logging.getLogger(__name__)

# Keep model weights inside tools/totalseg_segmentor/ rather than ~/.totalsegmentator/
_PACKAGE_DIR = dirname(abspath(__file__))
os.environ.setdefault('TOTALSEG_HOME_DIR', _PACKAGE_DIR)

# ---------------------------------------------------------------------------
# TotalSegmentator: structure-name → project vertebra name
# ---------------------------------------------------------------------------
TOTALSEG_NAME_TO_PROJECT = {
    'vertebrae_C1': 'C1', 'vertebrae_C2': 'C2', 'vertebrae_C3': 'C3',
    'vertebrae_C4': 'C4', 'vertebrae_C5': 'C5', 'vertebrae_C6': 'C6',
    'vertebrae_C7': 'C7',
    'vertebrae_T1': 'T1', 'vertebrae_T2': 'T2', 'vertebrae_T3': 'T3',
    'vertebrae_T4': 'T4', 'vertebrae_T5': 'T5', 'vertebrae_T6': 'T6',
    'vertebrae_T7': 'T7', 'vertebrae_T8': 'T8', 'vertebrae_T9': 'T9',
    'vertebrae_T10': 'T10', 'vertebrae_T11': 'T11', 'vertebrae_T12': 'T12',
    'vertebrae_L1': 'L1', 'vertebrae_L2': 'L2', 'vertebrae_L3': 'L3',
    'vertebrae_L4': 'L4', 'vertebrae_L5': 'L5',
    'vertebrae_S1': 'LS',   # sacral base → project LS (lumbosacral junction)
    'sacrum': 'SA',
}

# TotalSegmentator v2 hardcoded label integers (fallback when class_map
# cannot be loaded programmatically)
TOTALSEG_V2_CLASS_MAP = {
    1: 'vertebrae_L5', 2: 'vertebrae_L4', 3: 'vertebrae_L3',
    4: 'vertebrae_L2', 5: 'vertebrae_L1',
    6: 'vertebrae_T12', 7: 'vertebrae_T11', 8: 'vertebrae_T10',
    9: 'vertebrae_T9', 10: 'vertebrae_T8', 11: 'vertebrae_T7',
    12: 'vertebrae_T6', 13: 'vertebrae_T5', 14: 'vertebrae_T4',
    15: 'vertebrae_T3', 16: 'vertebrae_T2', 17: 'vertebrae_T1',
    18: 'vertebrae_C7', 19: 'vertebrae_C6', 20: 'vertebrae_C5',
    21: 'vertebrae_C4', 22: 'vertebrae_C3', 23: 'vertebrae_C2',
    24: 'vertebrae_C1',
    25: 'sacrum', 26: 'vertebrae_S1',
}

# Vertebrae ROI subset for TotalSegmentator (avoids segmenting organs, etc.)
ROI_SUBSET = list(TOTALSEG_NAME_TO_PROJECT.keys())


def _get_class_map():
    """Try to load TotalSegmentator's class_map programmatically.

    Returns dict {int_label: structure_name} or None if unavailable.
    """
    try:
        from totalsegmentator.map_to_binary import class_map
        # class_map is {task_name: {int_str: name, ...}}
        if 'total' in class_map:
            return {int(k): v for k, v in class_map['total'].items()}
    except (ImportError, KeyError, AttributeError):
        pass
    return None


def _remap(seg_data, class_map_inv):
    """Remap TotalSegmentator integer labels to project integers.

    Parameters
    ----------
    seg_data : ndarray
        Label volume with TotalSeg integers.
    class_map_inv : dict
        {totalseg_int: structure_name} mapping.

    Returns
    -------
    ndarray
        Label volume with project integers (from seg_val).
    """
    max_label = int(seg_data.max())
    lut = np.zeros(max_label + 1, dtype=np.uint8)

    for ts_int, ts_name in class_map_inv.items():
        if ts_int > max_label:
            continue
        project_name = TOTALSEG_NAME_TO_PROJECT.get(ts_name)
        if project_name is None:
            continue
        project_int = seg_val.get(project_name)
        if project_int is None:
            log.warning('no project label for %s — skipping', ts_name)
            continue
        lut[ts_int] = project_int

    remapped = lut[seg_data.astype(np.intp)]

    # log found vertebrae
    found = np.unique(remapped)
    found_names = [val_seg[int(v)] for v in found if v != 0]
    log.info('found %d vertebrae: %s', len(found_names), ', '.join(found_names))
    for name in found_names:
        count = np.sum(remapped == seg_val[name])
        log.debug('  %s: %d voxels', name, count)

    return remapped


def run_totalseg(input_path, output_dir, device='cpu', fast=False):
    """Run TotalSegmentator and return remapped Nifti1Image."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        raise RuntimeError(
            'TotalSegmentator not installed.\n'
            'Run: bash tools/totalseg_segmentor/setup.sh'
        )

    input_img = nib.load(input_path)
    log.info('running TotalSegmentator on %s (device=%s, fast=%s)',
             basename(input_path), device, fast)

    seg_img = totalsegmentator(
        input=input_img,
        ml=True,
        fast=fast,
        device=device,
        roi_subset=ROI_SUBSET,
        skip_saving=True,
    )

    seg_data = np.asarray(seg_img.dataobj, dtype=np.int16)

    # build class_map: totalseg_int → structure_name
    class_map_inv = _get_class_map()
    if class_map_inv is None:
        log.warning('could not load TotalSegmentator class_map — '
                     'falling back to hardcoded v2 mapping')
        class_map_inv = TOTALSEG_V2_CLASS_MAP

    remapped = _remap(seg_data, class_map_inv)
    return nib.Nifti1Image(remapped.astype(np.uint8), seg_img.affine, seg_img.header)
