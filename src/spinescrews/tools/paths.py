"""
Centralized path helpers for the pipeline output directory structure.

Each pipeline step writes to a numbered subdirectory under analysis/:
    01_segmentation/  02_preop/  03_correspondence/  04_orient/
    05_detection/  06_registration/  07_accuracy/

Each step directory contains a summary.json gate file written atomically as
the last action.  Presence of this file signals that the step completed
successfully and can be skipped on re-run.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from os.path import join, isfile

# Step directory names
STEP_SEGMENTATION = '01_segmentation'
STEP_PREOP = '02_preop'
STEP_CORRESPONDENCE = '03_correspondence'
STEP_ORIENT = '04_orient'
STEP_DETECTION = '05_detection'
STEP_REGISTRATION = '06_registration'
STEP_ACCURACY = '07_accuracy'


def setup_logging(logfile, debug=False):
    """Configure root logging: a UTF-8 file handler plus a stderr stream handler.

    Forcing UTF-8 keeps non-ASCII characters in log messages (->, deg, mm^3,
    sigma, ...) from raising UnicodeEncodeError on Windows consoles and log files
    that default to cp1252. ``debug`` controls only the console verbosity; the
    log file always records DEBUG.
    """
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except (AttributeError, ValueError):
        pass  # stream doesn't support reconfigure (already wrapped / redirected)
    fh = logging.FileHandler(logfile, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])


# ---------------------------------------------------------------------------
# Directory builders
# ---------------------------------------------------------------------------

def segmentation_dir(analysis_dir):
    """Path to 01_segmentation/ under the analysis directory."""
    return join(analysis_dir, STEP_SEGMENTATION)


def segmentation_file(analysis_dir):
    """Path to the whole-volume preop_seg.nii.gz in 01_segmentation/."""
    return join(segmentation_dir(analysis_dir), 'preop_seg.nii.gz')


def preop_dir(analysis_dir):
    """Path to 02_preop/ under the analysis directory."""
    return join(analysis_dir, STEP_PREOP)


def preop_level_dir(analysis_dir, level):
    """Path to 02_preop/{LEVEL}/ for a specific vertebral level."""
    return join(analysis_dir, STEP_PREOP, level)


def correspondence_dir(analysis_dir):
    """Path to 03_correspondence/ under the analysis directory."""
    return join(analysis_dir, STEP_CORRESPONDENCE)


def correspondence_level_dir(analysis_dir, level):
    """Path to 03_correspondence/{LEVEL}/ for a specific vertebral level."""
    return join(analysis_dir, STEP_CORRESPONDENCE, level)


def orient_dir(analysis_dir):
    """Path to 04_orient/ under the analysis directory."""
    return join(analysis_dir, STEP_ORIENT)


def orient_level_dir(analysis_dir, level):
    """Path to 04_orient/{LEVEL}/ for a specific vertebral level."""
    return join(analysis_dir, STEP_ORIENT, level)


def detection_dir(analysis_dir):
    """Path to 05_detection/ under the analysis directory."""
    return join(analysis_dir, STEP_DETECTION)


def registration_dir(analysis_dir):
    """Path to 06_registration/ under the analysis directory."""
    return join(analysis_dir, STEP_REGISTRATION)


def registration_level_dir(analysis_dir, level):
    """Path to 06_registration/{LEVEL}/ for a specific vertebral level."""
    return join(analysis_dir, STEP_REGISTRATION, level)


def accuracy_dir(analysis_dir):
    """Path to 07_accuracy/ under the analysis directory."""
    return join(analysis_dir, STEP_ACCURACY)


def breach_mesh_dir(analysis_dir, screw_name):
    """Path to 07_accuracy/breach_{screw_name}/ for mesh exports."""
    return join(analysis_dir, STEP_ACCURACY, 'breach_%s' % screw_name)


# ---------------------------------------------------------------------------
# Gate file helpers
# ---------------------------------------------------------------------------

def summary_path(step_dir):
    """Path to summary.json gate file inside a step directory."""
    return join(step_dir, 'summary.json')


def step_complete(step_dir):
    """True if summary.json exists in step_dir, indicating the step completed."""
    return isfile(summary_path(step_dir))


def write_summary(step_dir, data):
    """Write summary.json atomically (write to tmp then rename)."""
    os.makedirs(step_dir, exist_ok=True)
    out = summary_path(step_dir)
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, out)


def read_summary(step_dir):
    """Load and return the parsed summary.json from a step directory."""
    with open(summary_path(step_dir), 'r') as f:
        return json.load(f)


def _json_default(obj):
    """JSON serializer for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Object of type %s is not JSON serializable' % type(obj).__name__)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

_timing_log = logging.getLogger('pipeline.timing')


@contextmanager
def timed(label, timings=None):
    """Log elapsed time for a block; optionally store in a dict."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    _timing_log.info('  %-40s  %7.1fs' % (label, elapsed))
    if timings is not None:
        timings[label] = round(elapsed, 2)
