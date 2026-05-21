# spinescrews

Automated pipeline for assessing pedicle screw placement accuracy in CT scans.
Given pre-operative and post-operative CT volumes, the pipeline segments vertebrae,
registers the two scans, detects metal screws, and computes positional and angular
errors relative to the planned trajectories and pedicle anatomy.

## Installation

Tested on macOS and Linux. Windows is not tested.

### 1. Conda environment

```bash
conda env create --file environment.yml
conda activate screws310
```

### 2. bg3dtools

Install [bg3dtools](https://github.com/bgroisser/bg3dtools) (provides the
`bg3dtools` and `spectral_match` packages):

```bash
pip install -e /path/to/bg3dtools
```

### 3. Install spinescrews

```bash
pip install -e .
```

This makes the `spinescrews` package importable and installs console scripts
(`spinescrews-align`, `spinescrews-preop`, `spinescrews-postop`,
`spinescrews-accuracy`, `spinescrews-segment`).

### 4. External tools

| Tool | Purpose | Install |
|------|---------|---------|
| `dcm2niix` | DICOM-to-NIfTI conversion | `brew install dcm2niix` or `apt install dcm2niix` |
| `dcmdump` | DICOM tag inspection | Part of [DCMTK](https://dicom.offis.de/dcmtk/) (`brew install dcmtk` / `apt install dcmtk`) |
| `jq` | JSON processing for DICOM survey | `brew install jq` / `apt install jq` |

### 5. Segmentation backend

**TotalSegmentator** (default, Apache-2.0) — installs directly into `screws310`:

```bash
bash src/spinescrews/tools/totalseg_segmentor/setup.sh
```

**Inria** (alternative, CC-BY-NC-SA-4.0) — requires a separate conda environment:

```bash
bash src/spinescrews/tools/inria_segmentor/setup.sh
```

See [src/spinescrews/tools/totalseg_segmentor/README.md](src/spinescrews/tools/totalseg_segmentor/README.md) and
[src/spinescrews/tools/inria_segmentor/README.md](src/spinescrews/tools/inria_segmentor/README.md) for details.

## Configuration

Settings are resolved in layers, where each layer overrides the previous:

1. `defaults.yml` (repo root) — sensible defaults for all specimens
2. `specimen_XX/config.yml` — per-specimen overrides
3. CLI flags (e.g. `--debug`)

A `config_resolved.yml` receipt is written to `analysis/` after each run so you
can see exactly which settings were used.

See `defaults.yml` for the full list of settings and their defaults.

## Usage

### Step 0: DICOM preparation

Use `survey_dicoms.sh` to scan your DICOM directories and build a metadata index:

```bash
src/spinescrews/dicom_utils/survey_dicoms.sh -o metadata.json /path/to/dicom_dir1 /path/to/dicom_dir2
```

Then use `convert_to_nii.sh` to extract matching series as NIfTI. You can filter
by any DICOM field (case-insensitive substring match), or run it without filters
for an interactive selection menu:

```bash
# Extract pre-op CT (filter by series description)
src/spinescrews/dicom_utils/convert_to_nii.sh metadata.json /path/to/specimen_XX preop series_description:"BONE STD"

# Extract post-op CT
src/spinescrews/dicom_utils/convert_to_nii.sh metadata.json /path/to/specimen_XX postop series_description:"MAZOR BONE"

# Interactive mode (no filters — presents a selection menu)
src/spinescrews/dicom_utils/convert_to_nii.sh metadata.json /path/to/specimen_XX preop
```

This produces `preop.nii.gz` and `postop.nii.gz` in the specimen directory.

### Step 1: Vertebral segmentation

```bash
spinescrews-segment --input /path/to/specimen_XX/preop.nii.gz \
                     --output_dir /path/to/specimen_XX
```

To use the Inria backend instead of TotalSegmentator:

```bash
spinescrews-segment --input /path/to/specimen_XX/preop.nii.gz \
                     --output_dir /path/to/specimen_XX \
                     --backend inria
```

### Steps 2-6: Alignment and registration

Run the full pipeline (pre-operative normalization, template correspondence,
orientation refinement, screw detection, articulated registration):

```bash
spinescrews-align /path/to/specimen_XX
```

Or run pre-op and post-op stages separately:

```bash
spinescrews-preop /path/to/specimen_XX    # steps 01-04
spinescrews-postop /path/to/specimen_XX   # steps 05-06
```

Each step writes a `summary.json` gate file when it finishes. On re-run, completed
steps are skipped automatically.

### Step 7: Accuracy measurement

```bash
spinescrews-accuracy /path/to/specimen_XX
```

Results are written to `analysis/07_accuracy/results.csv`. Breach figures and
mesh exports (PLY files for review in MeshLab or Blender) are generated
automatically.

## Re-generating figures

Figures are produced automatically at each pipeline step. To regenerate them
independently:

```bash
# Segmentation overlay (axial / coronal / sagittal)
python -m spinescrews.figures.seg_overlay /path/to/specimen_XX

# Oriented vertebrae grid (CT slices + mesh per level)
python -m spinescrews.figures.preop_orientation /path/to/specimen_XX

# Correspondence QC
python -m spinescrews.figures.correspondence_preprocess /path/to/specimen_XX --level T11
python -m spinescrews.figures.correspondence_match /path/to/specimen_XX --level T11

# Orientation refinement bar charts
python -m spinescrews.figures.orient_refinement /path/to/specimen_XX

# Spine construct (seg + template overlay)
python -m spinescrews.figures.spine_template /path/to/specimen_XX --step preop
python -m spinescrews.figures.spine_template /path/to/specimen_XX --step orient

# Screw detection MIP
python -m spinescrews.figures.detection_screws /path/to/specimen_XX

# Planned vs detected comparison
python -m spinescrews.figures.detection_plan_vs_detected /path/to/specimen_XX

# 4-panel CT visualization per level
python -m spinescrews.figures.CT_visualization /path/to/specimen_XX

# Breach visualization for a specific screw
python -m spinescrews.figures.visualize_breach /path/to/specimen_XX --level T11 --side L
```

## License

Copyright 2026 Hospital for Special Surgery.

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free for research, education, and other noncommercial use; commercial use requires a separate license. See [LICENSE](LICENSE) for the full terms.

Note: the optional Inria `vertebrae_segmentation` backend (`tools/inria_segmentor/`) is separately licensed under CC-BY-NC-SA and is fetched at setup time via `setup.sh`; the optional TotalSegmentator backend is Apache-2.0. Neither is bundled in this repository.
