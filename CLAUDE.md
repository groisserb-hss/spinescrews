# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical imaging pipeline for assessing pedicle screw placement accuracy in CT scans. Compares planned vs. detected screw positions by registering pre-operative and post-operative CT volumes, detecting metal screws, and computing positional/angular errors relative to vertebral anatomy.

## Package Structure

The project is organized as the `spinescrews` pip-installable package:

```
spinescrews/
├── pyproject.toml               # pip install -e .
├── src/
│   └── spinescrews/             # the installable package
│       ├── pipeline/            # main entry points
│       ├── tools/               # core modules
│       │   ├── articulated_models/  # Articulated base class + Spine model
│       │   ├── inria_segmentor/     # Inria vertebrae_segmentation (CC-BY-NC-SA)
│       │   └── totalseg_segmentor/  # TotalSegmentator wrapper (Apache-2.0)
│       ├── figures/             # figure generation
│       └── dicom_utils/         # DICOM survey + conversion scripts
├── vertebra_templates/          # template meshes + labels (data, not code)
├── defaults.yml                 # repo-level config defaults
└── environment.yml
```

## Environment Setup

```bash
conda env create --file environment.yml
conda activate screws310
pip install -e .   # install spinescrews in editable mode
```

Requires `bg3dtools` (repo: `~/dev/bg3dtools`, install with `pip install -e`). Provides `bg3dtools` and `spectral_match` packages.

External tool: `dcm2niix` for DICOM-to-NIfTI conversion.

## Running the Pipeline

Console scripts (installed by `pip install -e .`):

```bash
# Step 1: Vertebral segmentation (two backends: totalseg [default] or inria)
spinescrews-segment --input /path/to/preop.nii.gz --output_dir /path/to/specimen_XX
# Optional flags: --backend totalseg|inria  --device cpu|gpu|mps  --fast

# Steps 02-06: Alignment & registration
spinescrews-align /path/to/specimen_XX        # full pipeline (preop + postop)
spinescrews-preop /path/to/specimen_XX        # preop only (steps 01-04)
spinescrews-postop /path/to/specimen_XX       # postop only (steps 05-06)

# Step 07: Error measurement
spinescrews-accuracy /path/to/specimen_XX

# Re-generate figures standalone:
python -m spinescrews.figures.seg_overlay /path/to/specimen_XX
python -m spinescrews.figures.preop_orientation /path/to/specimen_XX
python -m spinescrews.figures.correspondence_preprocess /path/to/specimen_XX --level T11
python -m spinescrews.figures.correspondence_match /path/to/specimen_XX --level T11
python -m spinescrews.figures.orient_refinement /path/to/specimen_XX
python -m spinescrews.figures.spine_template /path/to/specimen_XX --step preop
python -m spinescrews.figures.spine_template /path/to/specimen_XX --step orient
python -m spinescrews.figures.detection_screws /path/to/specimen_XX
python -m spinescrews.figures.detection_plan_vs_detected /path/to/specimen_XX
python -m spinescrews.figures.CT_visualization /path/to/specimen_XX
python -m spinescrews.figures.visualize_breach /path/to/specimen_XX --level T11 --side L
```

Figures are generated automatically at the end of each pipeline step; the standalone scripts above are for manual re-generation.

### Configuration

Config is resolved in layers: `defaults.yml` (repo root) → `specimen_XX/config.yml` (per-specimen overrides) → CLI flags. A `config_resolved.yml` receipt is written to `analysis/` after each run.

CLI flag `--debug` overrides config values. See `src/spinescrews/tools/config.py` for all fields.

### Output Directory Structure

Each pipeline step writes to a numbered subdirectory under `analysis/`. A `summary.json` gate file is written atomically as the last action of each step — its presence signals the step completed successfully and can be skipped on re-run.

```
analysis/
├── 01_segmentation/
│   ├── preop_seg.nii.gz             # whole-volume segmentation
│   ├── summary.json                 # gate file: voxel counts, levels found
│   └── seg_overlay.png              # axial/coronal/sagittal with colored labels
│
├── 02_preop/
│   ├── {LEVEL}/
│   │   ├── preop_affine.npy
│   │   ├── preop_gen1.ply           # genus-1 decimated mesh
│   │   ├── preop_gen1-inflated.ply  # inflated mesh (erosion unwound)
│   │   ├── small2med.npz            # sparse: decimated ↔ inflated correspondence
│   │   └── med2small.npz
│   ├── summary.json                 # per-vertebra normalization
│   ├── spine_seg.png                # seg meshes: anterior + lateral
│   ├── spine_template.png           # template meshes: anterior + lateral
│   ├── spine_overlay.png            # seg solid + template point cloud
│   └── spine_template.ply           # combined template mesh
│
├── 03_correspondence/
│   ├── {LEVEL}/
│   │   ├── bone_preprocess.npz      # preprocessing cache (mesh + geodesics)
│   │   ├── preprocess.png           # 4-panel overhead geodesic visualization
│   │   ├── match.png                # 2×2 label + gradient correspondence QC
│   │   ├── template2bone.npz
│   │   ├── bone2template.npz
│   │   └── preop_seg.ply
│   └── summary.json                 # per-vertebra dg, template match quality
│
├── 04_orient/
│   ├── {LEVEL}/
│   │   ├── preop_affine-refined.npy
│   │   ├── template_scale.npy
│   │   ├── preop.nii.gz             # normalized CT (refined frame)
│   │   └── preop_seg.nii.gz         # normalized segmentation (refined frame)
│   ├── summary.json                 # per-level: angle_deg, trans_mm, outlier flag
│   ├── orient_refinement.png        # bar charts: rotation + translation corrections
│   ├── preop_orientation.png        # grid: CT slices + mesh per level
│   ├── spine_seg.png                # seg meshes: anterior + lateral (refined)
│   ├── spine_template.png           # template meshes: anterior + lateral (refined)
│   ├── spine_overlay.png            # seg solid + template point cloud (refined)
│   └── spine_template.ply           # combined template mesh (refined)
│
├── 05_detection/
│   ├── {LEVEL}{SIDE}_screw.yml
│   ├── spine_tforms_initial.npz
│   ├── summary.json                 # init_err, opt_err, n_metal_pts, etc.
│   ├── global_spine-opt.png         # intermediate MIP after spine optimization
│   ├── global_screw-opt.png         # intermediate MIP after screw optimization
│   ├── detection_screws.png         # 2x2 multi-angle MIP with detected screws
│   └── detection_plan-vs-detected.png  # 2x2 multi-angle MIP: planned vs detected
│
├── 06_registration/
│   ├── {LEVEL}/
│   │   ├── postop-reg.nii.gz
│   │   ├── postop-reg_affine.npy
│   │   ├── postop-reg_seg.nii.gz
│   │   └── preop_seg.nii.gz          # preop binary seg for QC overlay
│   ├── spine_tforms_icp.npz
│   ├── icp_postop.ply               # postop cortical point cloud (QC)
│   ├── icp_model.ply                # color-coded preop model point cloud (QC)
│   ├── summary.json                 # per-vertebra: ICP ratios, MI fopt
│   └── {LEVEL}.png                  # 4-panel CT visualization
│
├── 07_accuracy/
│   ├── results.csv
│   ├── summary.json                 # per-screw errors, breach summary
│   ├── breach_{LEVEL}{SIDE}.png     # 3-panel screw-aligned CT
│   └── breach_{LEVEL}{SIDE}/        # mesh exports for Blender/MeshLab review
│       ├── bone.ply
│       ├── canal.ply
│       ├── screw.ply
│       └── line.ply
│
├── config_resolved.yml
└── pipeline.log
```

## Architecture

### Pipeline Flow

```
DICOM → NIfTI → Vertebral Segmentation → spinescrews-align → spinescrews-accuracy
                                          (steps 02-06)       (step 07)
```

### Main Entry Points (`src/spinescrews/pipeline/`)

- **`run_segmentation.py`** — Step 01. Multi-backend segmentation wrapper (TotalSegmentator or Inria). Converts output labels to the project's `seg_val` encoding.
- **`align_preop.py`** — `Aligner` class. Steps 02-04: extract vertebra meshes with skeleton-guided genus-1 enforcement, compute spectral template correspondence, refine orientation via Whittaker-smoothed corrections.
- **`register_postop.py`** — `Registrar` class. Steps 05-06: detect screws via articulated `InstrumentedSpine` model, articulated ICP with particle belief propagation (D-PMP) refit, then per-level volumetric mutual-information (MI) refinement.
- **`align_vertebrae.py`** — Thin orchestrator calling `run_preop()` then `run_postop()`.
- **`compute_accuracy.py`** — `ErrorComputer` class. Step 07: measures entry/tip positional errors, pedicle breach distances (medial direction via bone contour normals), angular deviations. Writes results to 07_accuracy/ with breach figures and mesh exports.

### Core Modules (`src/spinescrews/tools/`)

- **`__init__.py`** — Central data structures: `seg_val`/`val_seg` (vertebra label ↔ integer mappings), `ScrewMeasures`, `BreachMeasures`, `MeshLabels` namedtuples. Coordinate axes: `dimR=0, dimA=1, dimS=2` (Right-Anterior-Superior).
- **`paths.py`** — Centralized step directory constants and path builders. Gate file helpers (`step_complete()`, `write_summary()`). Timing context manager (`timed()`).
- **`config.py`** — `PipelineConfig` frozen dataclass, layered YAML config loader (`load_config()`).
- **`nifti_utils.py`** — NIfTI processing helpers: `compute_metal_threshold()` (Otsu-based auto metal detection), `nonzero_box()` (tight bounding-box crop).
- **`vertebrae.py`** — `Vertebra` class. Skeleton-based canal loop detection (`initialize_canal_loop`), genus-1 mesh extraction with skeleton support field (`get_mesh_genus1`), erosion unwinding to recover lost anatomy, orientation normalization, affine transforms, volume cropping, save/load. Module-level `fit_canal()` for iterative canal fitting.
- **`correspondence_tools.py`** — `SpectralDescriptor` class for spectral + anatomical feature matching. `load_vertebral_template()` loads PLY templates and caches preprocessed spectral descriptors.
- **`screw_models.py`** — `Screw` class with type attribute (`skip`, `fixed`, `headless`, `poly`). `parse_preop_plan()` parses 3DSlicer CSV screw plans, tracks planned vs detected positions, serializes to YAML.
- **`screw_detection.py`** — `detect_screws()`: multi-stage detection pipeline using `InstrumentedSpine` articulated model. RANSAC ICP initialization → articulated spine optimization → screw optimization → per-screw cloud fitting → HU-weighted refinement with neighbor point exclusion. Metal threshold auto-computed via Otsu. Returns `(spine_tforms, metrics)`.
- **`articulated_spine_registration.py`** — `align_spine_to_CT()`: multi-level articulated ICP with Geman-McClure robust loss, followed by particle belief propagation (D-PMP) refit for poorly-aligned levels. Includes fast artifact mask construction (`_build_artifact_mask_fast`) for streak artifact exclusion. Returns `(icp_affs, metrics, artifact_mask)`.
- **`articulated_models/`** — `Articulated` base class (`base_unified.py`) and `Spine` model (`spine.py`) for kinematic chain deformation of per-vertebra meshes.
- **`error.py`** — `measure_screw_error()`, `distance_to_pedicle()`, `breached_distance()`. Error quantification relative to template anatomy. Medial/lateral classification uses bone mesh cross-section normals rather than simple angle thresholds. Saves breach meshes to permanent directory.

### Figure Generation (`src/spinescrews/figures/`)

- **`seg_overlay.py`** — 3-panel segmentation overlay (axial/coronal/sagittal).
- **`preop_orientation.py`** — Grid of oriented vertebrae: CT slices + genus-1 mesh per level (step 04).
- **`correspondence_preprocess.py`** — 4-panel overhead geodesic distance visualization per level.
- **`correspondence_match.py`** — 2×2 label + gradient correspondence QC per level.
- **`orient_refinement.py`** — Bar charts of rotation/translation corrections per level.
- **`spine_template.py`** — 3 figures per step: seg-only, template-only, and overlay (seg solid + template point cloud), each with anterior + lateral views.
- **`detection_screws.py`** — 2x2 multi-angle MIP with detected screw lines (red). Views: coronal, oblique +/-45 deg, sagittal.
- **`detection_plan_vs_detected.py`** — 2x2 multi-angle MIP comparing planned (green) vs detected (red) screws.
- **`CT_visualization.py`** — 4-panel CT visualization per level (axial/coronal/sagittal L/R).
- **`visualize_breach.py`** — 3-panel screw-aligned breach visualization.
- **`group_statistics.py`** — Cross-specimen error analysis (separate workflow).

All figure scripts have both importable functions (called by the pipeline) and standalone CLI entry points for re-generation.

### Data Formats

| Format | Usage |
|--------|-------|
| `.nii.gz` | CT volumes, segmentation masks |
| `.ply` | 3D meshes (template and per-vertebra) |
| `.npy` | Affine transform matrices |
| `.npz` | Template correspondence (sparse matrices) |
| `.yml` | Detected screw parameters |
| `.csv` | Surgical screw plans (input, 3DSlicer format) |
| `.json` | Step quality summaries (gate files) |

### Vertebra Naming Convention

LS = lumbosacral junction (sacral base / S1). Levels ordered caudal-to-cranial: LS, L5...L1, T13, T12...T1, C7...C2. Label integers in `seg_val` dict.

### Template System (`vertebra_templates/`)

Pre-computed PLY meshes for C2–LS in `meshes/`. Anatomical region labels (pedicles, canal, endplates, body walls) in `labels/`. ResNet weights for geodesic feature computation in `resnet_weights/`.

## Key Conventions

- Coordinate system is RAS (Right-Anterior-Superior) throughout
- Preop CT standardized to 0.5mm isotropic voxels
- Metal detection thresholds auto-computed via Otsu's method (`compute_metal_threshold()`); can be overridden with integer values in `config.yml` (`metal_mask_threshold`, `screw_detect_threshold`)
- Parallelism via `joblib.Parallel` for template matching, registration, and screw fitting
- Logging via Python `logging` module with file + stderr handlers
- Per-step output directories under `analysis/` with `summary.json` gate files
- Figures always generated automatically (no `--render` flag needed)
