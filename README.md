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
`bg3dtools` and `spectral_match` packages). spinescrews uses the `mesh`,
`viz`, and `graph` extras:

```bash
pip install -e '/path/to/bg3dtools[mesh,viz,graph]'
```

(spinescrews's own `pip install -e .` will request the same extras
transitively, but installing bg3dtools editable first keeps the source of
truth obvious.)

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

### Screw planning (3D Slicer)

`slicer_tools/` holds companion utilities for authoring pedicle-screw plans in
[3D Slicer](https://www.slicer.org/) and exporting them for surgical navigation.
They run inside Slicer / as a standalone script (not via the `spinescrews-*`
console scripts) and produce the RAS entry/tip CSV the pipeline reads as its
planned reference.

**Plan screws.** The *Hybrid Screw Planner* turns a Markups Line (place the
entry point first, the tip second) into a screw cylinder of a chosen length,
radius, and **screw type** (Fixed / Headless / Polyaxial / Skip) — the entry
stays put, the tip is re-snapped to the length, and the cylinder auto-updates as
you drag the line. Selecting a line reloads its saved settings, so revisiting a
screw doesn't overwrite it. "Export all line coordinates..." writes one
pipeline-ready row per line (`line_name, screw_type, line_id, entry_ras_*,
tip_ras_*, length_mm, cylinder_radius_mm, cylinder_model_name`) and logs a
warning for anything `parse_preop_plan` would reject — names that aren't
`<level><side>`, or levels missing an L/R partner. Install it by adding
`slicer_tools/HybridScrewPlanner` under *Edit > Application Settings > Modules >
Additional module paths* and restarting; it then appears as "Hybrid Screw
Planner" in the *Planning* category. Self-tests: *Testing > Self-Tests*, or
`slicer.selfTests["HybridScrewPlanner"]()` (see
`slicer_tools/HybridScrewPlanner/tests.txt`).

**Burn endpoints into DICOM (for Mazor).** `burn_screw_endpoints.py` paints a
small high-HU sphere at every entry and tip from the planner CSV into a copy of
the source CT DICOM series (RAS→LPS conversion and DERIVED/SECONDARY tagging
handled), so the planned points appear as fiducials when the series is loaded
into navigation software such as Mazor. It runs on `pydicom` (already in the
`screws310` environment — no extra install):

```bash
python slicer_tools/burn_screw_endpoints.py \
    --dicom-dir /path/to/ct_dicom_series \
    --csv /path/to/screw_line_coordinates.csv \
    --out-dir /path/to/burned_dicom_export
```

Optional flags: `--backend` (`pydicom` default, or `simpleitk` if it is
installed), `--series-id` (when the folder holds multiple series), `--radius-mm`
(default 1.0), `--burn-value` (default 3000 HU). Compressed source series need a
pydicom codec (`pip install pylibjpeg pylibjpeg-libjpeg`, or `python-gdcm`).

**Using a plan with the pipeline.** The exported CSV is already the format
`parse_preop_plan()` reads as the per-specimen `preop_plan.csv` (see
[Step 7](#step-7-accuracy-measurement)) — the `screw_type` column is written for
you. Just name each Markups line `<level><side>` (e.g. `T11L`, `L1R`); the
planner warns at export about any name that doesn't match or any level missing
its L/R partner (use a `Skip`-type line to stand in for an un-instrumented side).

A backend-equivalence test suite lives in `slicer_tools/tests/` — it runs
pure-math geometry checks by default; point `SCREWS_TEST_DICOM_DIR` at a CT
series (and set `SCREWS_TEST_SITK=1`) to validate the pydicom backend against
SimpleITK on real data:

```bash
SCREWS_TEST_DICOM_DIR=/path/to/ct_dicom_series SCREWS_TEST_SITK=1 \
    python slicer_tools/tests/test_burn_screw_endpoints.py
```

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
