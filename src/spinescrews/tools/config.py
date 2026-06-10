"""
Layered YAML config system for the pedicle screw pipeline.

Config resolution order:
    1. defaults.yml  (repo root, version-controlled)
    2. specimen_dir/config.yml  (per-specimen overrides)
    3. programmatic overrides  (e.g. CLI flags)

Usage:
    from spinescrews.tools.config import load_config
    config = load_config('/path/to/specimen_XX')
"""

import os
from os.path import join, expanduser, isdir, isfile, dirname, abspath
from dataclasses import dataclass, fields, asdict
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Repo root (src/ layout: repo/src/spinescrews/tools/config.py)
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = dirname(dirname(abspath(__file__)))   # .../src/spinescrews/
_REPO_ROOT = dirname(dirname(_PACKAGE_ROOT))           # .../spinescrews/ (repo root)

# ---------------------------------------------------------------------------
# Mapping from nested YAML paths to flat dataclass field names.
# Flat keys that already match a field name also work, so simple
# config.yml files can skip nesting entirely.
# ---------------------------------------------------------------------------
_YAML_KEY_MAP = {
    # paths
    'paths.template_dir':              'template_dir',
    'paths.output_dir':                'output_dir',
    # runtime
    'runtime.debug':                   'debug',
    'runtime.n_jobs':                  'n_jobs',
    'runtime.anatomic_axis':           'anatomic_axis',
    'runtime.no-patches':              'no_patches',
    # thresholds
    'thresholds.metal_mask_threshold': 'metal_mask_threshold',
    'thresholds.screw_detect_threshold': 'screw_detect_threshold',
    # geometry
    'geometry.preop_voxel-size':       'preop_voxel_size',
    # orientation refinement
    'orient.lam-rot':                  'orient_lam_rot',
    'orient.lam-trans':                'orient_lam_trans',
    'orient.lam-scale':                'orient_lam_scale',
    # pmf
    'pmf.sigma':                       'pmf_sigma',
    'pmf.gamma':                       'pmf_gamma',
    'pmf.iterations':                  'pmf_iterations',
    'pmf.fmapper_target_size':         'fmapper_target_size',
    'pmf.fmapper_preprocess_size':     'fmapper_preprocess_size',
    # volumetric registration
    'registration.volumetric.method':          'mi_method',
    'registration.volumetric.preop_dilate':   'mi_preop_dilate',
    'registration.volumetric.postop_dilate':  'mi_postop_dilate',
    'registration.volumetric.quality_fail':   'mi_quality_fail',
    'registration.volumetric.quality_warn':   'mi_quality_warn',
    'registration.volumetric.n_jobs':         'mi_n_jobs',
    # icp
    'registration.icp.iso_res':        'icp_iso_res',
    'registration.icp.initial_radius': 'icp_initial_radius',
    'registration.icp.ratio_thresh':   'icp_ratio_thresh',
    # segmentation
    'segmentation.backend':            'seg_backend',
    'segmentation.device':             'seg_device',
    'segmentation.fast':               'seg_fast',
    'segmentation.inria_repo':         'seg_inria_repo',
    'segmentation.inria_env':          'seg_inria_env',
}


@dataclass(frozen=True)
class PipelineConfig:
    # --- paths ---
    specimen_dir: str = ''
    template_dir: str = join(_REPO_ROOT, 'vertebra_templates')
    output_dir: str = 'analysis'

    # --- runtime ---
    debug: bool = False
    n_jobs: int = -3
    anatomic_axis: bool = False
    no_patches: bool = False
    exclude_screws: tuple = ()

    # --- thresholds ---
    metal_mask_threshold: Optional[int] = None
    screw_detect_threshold: Optional[int] = None

    # --- geometry ---
    preop_voxel_size: float = 0.5

    # --- orientation refinement (step 04) ---
    orient_lam_rot: float = 3.0
    orient_lam_trans: float = 0.001
    orient_lam_scale: float = 0.01

    # --- pmf (product manifold filter) ---
    pmf_sigma: float = 3.0
    pmf_gamma: float = 0.7
    pmf_iterations: int = 8
    fmapper_target_size: int = 3200
    fmapper_preprocess_size: int = 3100

    # --- volumetric registration (mutual information) ---
    mi_method: str = 'L-BFGS-B'
    mi_preop_dilate: int = 4
    mi_postop_dilate: int = -2
    mi_quality_fail: float = -0.15
    mi_quality_warn: float = -0.25
    mi_n_jobs: int = -3

    # --- icp registration ---
    icp_iso_res: float = 1.5
    icp_initial_radius: float = 12.0
    icp_ratio_thresh: float = 0.75

    # --- segmentation ---
    seg_backend: str = 'totalseg'
    seg_device: str = 'cpu'
    seg_fast: bool = False
    seg_inria_repo: Optional[str] = None
    seg_inria_env: str = 'verse20'


def _flatten_yaml(data, prefix=''):
    """Flatten a nested dict into dotted-path keys."""
    flat = {}
    for key, value in data.items():
        full_key = '%s.%s' % (prefix, key) if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_yaml(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _resolve_yaml(raw):
    """Map YAML keys (nested or flat) to PipelineConfig field names."""
    field_names = {f.name for f in fields(PipelineConfig)}
    flat = _flatten_yaml(raw)
    resolved = {}
    for dotted_key, value in flat.items():
        if dotted_key in _YAML_KEY_MAP:
            resolved[_YAML_KEY_MAP[dotted_key]] = value
        elif dotted_key in field_names:
            # flat key that directly matches a field
            resolved[dotted_key] = value
        else:
            # try dash → underscore normalization (e.g. preop_voxel-size → preop_voxel_size)
            norm_key = dotted_key.replace('-', '_')
            if norm_key in field_names:
                resolved[norm_key] = value
            # else: ignore unknown keys (forward-compat)
    return resolved


def _coerce_tuples(kwargs):
    """Convert lists from YAML into tuples for tuple-typed fields."""
    _tuple_fields = {f.name for f in fields(PipelineConfig) if f.type is tuple}
    for key, value in kwargs.items():
        if key in _tuple_fields and isinstance(value, list):
            kwargs[key] = tuple(value)
    return kwargs


def _validate(config):
    """Validate a PipelineConfig. Collects all errors, raises one ValueError."""
    errors = []

    if config.specimen_dir and not isdir(expanduser(config.specimen_dir)):
        errors.append('specimen_dir does not exist: %s' % config.specimen_dir)

    if not isdir(expanduser(config.template_dir)):
        errors.append('template_dir does not exist: %s' % config.template_dir)

    if config.metal_mask_threshold is not None and config.metal_mask_threshold <= 0:
        errors.append('metal_mask_threshold must be positive, got: %d' % config.metal_mask_threshold)

    if config.screw_detect_threshold is not None and config.screw_detect_threshold <= 0:
        errors.append('screw_detect_threshold must be positive, got: %d' % config.screw_detect_threshold)

    if config.preop_voxel_size <= 0:
        errors.append('preop_voxel_size must be positive, got: %f' % config.preop_voxel_size)

    if errors:
        raise ValueError('Config validation failed:\n  ' + '\n  '.join(errors))


def load_config(specimen_dir, overrides=None):
    """Load and merge config from defaults.yml + specimen config.yml + overrides.

    Parameters
    ----------
    specimen_dir : str
        Path to specimen directory.
    overrides : dict, optional
        Programmatic overrides (e.g. from CLI flags). Keys are field names.

    Returns
    -------
    PipelineConfig
    """
    kwargs = {'specimen_dir': expanduser(specimen_dir)}

    # 1. repo-level defaults
    defaults_file = join(_REPO_ROOT, 'defaults.yml')
    if isfile(defaults_file):
        with open(defaults_file, 'r') as f:
            raw = yaml.safe_load(f) or {}
        kwargs.update(_resolve_yaml(raw))

    # 2. per-specimen overrides
    specimen_config = join(expanduser(specimen_dir), 'config.yml')
    if isfile(specimen_config):
        with open(specimen_config, 'r') as f:
            raw = yaml.safe_load(f) or {}
        kwargs.update(_resolve_yaml(raw))

    # 3. programmatic overrides (CLI flags etc.)
    if overrides:
        field_names = {f.name for f in fields(PipelineConfig)}
        for key, value in overrides.items():
            if key in field_names and value is not None:
                kwargs[key] = value

    # specimen_dir is always authoritative
    kwargs['specimen_dir'] = expanduser(specimen_dir)

    # resolve template_dir relative to repo root (defaults.yml documents this)
    td = kwargs.get('template_dir', '')
    if td and not os.path.isabs(expanduser(td)):
        kwargs['template_dir'] = join(_REPO_ROOT, td)

    kwargs = _coerce_tuples(kwargs)
    config = PipelineConfig(**kwargs)
    _validate(config)
    return config


def save_resolved_config(config):
    """Write the fully-resolved config to analysis/config_resolved.yml."""
    specimen_dir = expanduser(config.specimen_dir)
    output_dir = join(specimen_dir, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_path = join(output_dir, 'config_resolved.yml')
    data = asdict(config)
    # convert tuples to lists for clean YAML output
    for key, value in data.items():
        if isinstance(value, tuple):
            data[key] = list(value)

    with open(out_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared CLI helpers for the pipeline console scripts
# ---------------------------------------------------------------------------
def add_common_pipeline_args(parser):
    """Add the --debug / --n-jobs flags shared by the alignment console scripts."""
    parser.add_argument('--debug', action='store_true', default=None,
                        help='Run single-threaded with verbose logging (n_jobs=1); for troubleshooting.')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='CPU cores for parallel steps (-1 = all, -3 = all but 2, 1 = serial).')


def overrides_from_args(args):
    """Build a config-override dict from the shared CLI args (--debug, --n-jobs).

    Script-specific flags (e.g. --no-patches, --mi-method) are merged in by the caller.
    """
    overrides = {}
    if getattr(args, 'debug', None):
        overrides['debug'] = True
        overrides['n_jobs'] = 1
        overrides['mi_n_jobs'] = 1
    if getattr(args, 'n_jobs', None) is not None:
        overrides['n_jobs'] = args.n_jobs
        overrides['mi_n_jobs'] = args.n_jobs
    return overrides
