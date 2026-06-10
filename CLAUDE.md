# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical imaging pipeline for assessing pedicle screw placement accuracy in CT scans. Compares planned vs. detected screw positions by registering pre-operative and post-operative CT volumes, detecting metal screws, and computing positional/angular errors relative to vertebral anatomy.

Tested on macOS and Linux. Python >= 3.10.

## Environment Setup

```bash
conda env create --file environment.yml
conda activate screws310
pip install -e '/path/to/bg3dtools[mesh,viz,graph]'   # external dep (provides bg3dtools + spectral_match)
pip install -e .                                       # install spinescrews in editable mode
```

If `triangle` fails via pip: `git clone --recurse-submodules https://github.com/drufat/triangle.git && cd triangle && python setup.py install`

Segmentation backend setup (run once):
```bash
bash src/spinescrews/tools/totalseg_segmentor/setup.sh   # TotalSegmentator (default)
bash src/spinescrews/tools/inria_segmentor/setup.sh      # Inria (alternative, separate conda env)
```

External CLI tools: `dcm2niix`, `dcmdump`, `jq`.

## Build & Quality

The main pipeline has **no tests, linting, CI/CD, or formatting configs**. Quality assurance is manual via pipeline runs, gate files (`summary.json`), and visual QC through auto-generated figures.

Exception: `slicer_tools/` ships an opt-in test suite (`slicer_tools/tests/test_burn_screw_endpoints.py`, runnable as plain `python …` or under `pytest`) for the screw-endpoint burn tool. Pure-math geometry tests run with no data; DICOM read/write and SimpleITK-vs-pydicom equivalence tests are gated on `SCREWS_TEST_DICOM_DIR` (a CT series folder) and `SCREWS_TEST_SITK=1`. No DICOMs are committed. The burn tool defaults to the `pydicom` backend (SimpleITK optional via `--backend simpleitk`); its `--burn-value` is in **HU** and output is written as int16 + `RescaleIntercept −1024` (consistent with `nifti_utils.py`).

## Running the Pipeline

Five console scripts (installed by `pip install -e .`):

```bash
spinescrews-segment --input /path/to/preop.nii.gz --output_dir /path/to/specimen_XX
spinescrews-preop /path/to/specimen_XX        # steps 02-04
spinescrews-postop /path/to/specimen_XX       # steps 05-06
spinescrews-align /path/to/specimen_XX        # steps 02-06 (preop + postop combined)
spinescrews-accuracy /path/to/specimen_XX     # step 07
```

Flags: `--backend totalseg|inria`, `--device cpu|gpu|mps`, `--fast`, `--debug`.

Re-generate any figure standalone: `python -m spinescrews.figures.<module_name> /path/to/specimen_XX [--level T11] [--side L] [--step preop|orient]`

### Configuration

Resolved in layers: `defaults.yml` (repo root) -> `specimen_XX/config.yml` (per-specimen) -> CLI flags. Receipt written to `analysis/config_resolved.yml`. See `src/spinescrews/tools/config.py` (`PipelineConfig` dataclass) for all fields.

### Output Convention

Each pipeline step writes to a numbered subdirectory under `analysis/` (01_segmentation through 07_accuracy). A `summary.json` gate file is written atomically as the last action of each step -- its presence signals successful completion and causes the step to be skipped on re-run. Figures are generated automatically at the end of each step.

## Architecture

### Pipeline Flow

```
DICOM -> NIfTI -> Step 01: Segmentation -> Steps 02-04: Preop Alignment -> Steps 05-06: Postop Registration -> Step 07: Accuracy
```

### Package Structure

```
src/spinescrews/
├── pipeline/            # entry points (one module per console script)
├── tools/               # core algorithms and data structures
│   ├── articulated_models/  # Articulated base class + Spine kinematic chain
│   ├── inria_segmentor/     # vendored Inria segmentation (CC-BY-NC-SA)
│   └── totalseg_segmentor/  # TotalSegmentator wrapper (Apache-2.0)
├── figures/             # 14 visualization modules (importable + standalone CLI)
└── dicom_utils/         # shell scripts for DICOM survey + conversion
```

### Pipeline Entry Points (`pipeline/`)

| Module | Class | Steps | What it does |
|--------|-------|-------|-------------|
| `run_segmentation.py` | -- | 01 | Multi-backend vertebral segmentation, converts labels to `seg_val` encoding |
| `align_preop.py` | `Aligner` | 02-04 | Genus-1 mesh extraction, spectral template correspondence, Whittaker-smoothed orientation refinement |
| `register_postop.py` | `Registrar` | 05-06 | Screw detection via `InstrumentedSpine`, articulated ICP + D-PMP refit, per-level MI refinement |
| `align_vertebrae.py` | -- | 02-06 | Thin orchestrator: calls `run_preop()` then `run_postop()` |
| `compute_accuracy.py` | `ErrorComputer` | 07 | Entry/tip positional errors, pedicle breach distances, angular deviations |

### Core Modules (`tools/`)

- **`__init__.py`** -- Central data structures: `seg_val`/`val_seg` (vertebra label <-> integer), `ScrewMeasures`, `BreachMeasures`, `MeshLabels` namedtuples. Coordinate axes: `dimR=0, dimA=1, dimS=2`.
- **`paths.py`** -- Step directory constants, path builders, gate file helpers (`step_complete()`, `write_summary()`), `timed()` context manager.
- **`config.py`** -- `PipelineConfig` frozen dataclass, layered YAML loader (`load_config()`).
- **`vertebrae.py`** -- `Vertebra` class: skeleton-based canal loop detection, genus-1 mesh extraction, erosion unwinding, orientation normalization, affine transforms, volume cropping.
- **`correspondence_tools.py`** -- `SpectralDescriptor` class for spectral + anatomical feature matching. `load_vertebral_template()` loads and caches preprocessed templates.
- **`screw_models.py`** -- `Screw` class (types: `skip`, `fixed`, `headless`, `poly`). `parse_preop_plan()` reads 3DSlicer CSV plans.
- **`screw_detection.py`** -- `detect_screws()`: RANSAC ICP init -> articulated spine optimization -> screw optimization -> per-screw cloud fitting -> HU-weighted refinement.
- **`articulated_spine_registration.py`** -- `align_spine_to_CT()`: multi-level articulated ICP with Geman-McClure loss + D-PMP belief propagation refit. Streak artifact mask construction.
- **`error.py`** -- `measure_screw_error()`, `distance_to_pedicle()`, `breached_distance()`. Medial/lateral classification via bone contour normals.

### Template System (`vertebra_templates/`)

Pre-computed PLY meshes for C2-LS in `meshes/`, anatomical region labels in `labels/`, ResNet weights for geodesic features in `resnet_weights/`. Build script: `scripts/construct_mesh_templates.py`.

## Key Conventions

- **Coordinate system**: RAS (Right-Anterior-Superior) throughout
- **Vertebra naming**: LS (sacral base/S1), then L5...L1, T13...T1, C7...C2 (caudal-to-cranial). Integer mappings in `seg_val` dict.
- **Preop CT**: standardized to 0.5mm isotropic voxels
- **Metal thresholds**: auto-computed via Otsu (`compute_metal_threshold()` in `nifti_utils.py`); override with integer values `metal_mask_threshold`/`screw_detect_threshold` in config
- **Parallelism**: `joblib.Parallel` (`n_jobs` in config, default -3 = all cores minus 2)
- **Logging**: Python `logging` with file (`pipeline.log`) + stderr handlers
- **Gate files**: `summary.json` per step -- presence means step completed successfully
