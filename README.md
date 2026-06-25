# spinescrews

Automated pipeline for assessing pedicle screw placement accuracy in CT scans.
Given pre-operative and post-operative CT volumes, the pipeline segments vertebrae,
registers the two scans, detects metal screws, and computes positional and angular
errors relative to the planned trajectories and pedicle anatomy.

## Installation

Tested on macOS and Linux. Windows is not officially supported (untested), though the install
and tools are cross-platform and expected to work — Windows-specific notes are flagged inline.
Run the commands below in a terminal, in order. Steps 3-7 are run **from the repo root** — the
`spinescrews` folder created in step 2.

### 1. Install Anaconda

The pipeline runs inside a conda environment, which also provides Python 3.10 (no separate Python
install needed). If you don't already have conda, install **Anaconda** from
<https://www.anaconda.com/download> (or the lighter **Miniconda** from
<https://docs.conda.io/projects/miniconda/en/latest/>), accept the installer defaults, then open
a new terminal so the `conda` command is available.

### 2. Get the code

```bash
git clone https://github.com/groisserb-hss/spinescrews.git
cd spinescrews
```

Run the remaining steps from inside this `spinescrews` folder.

### 3. Create the conda environment

```bash
conda env create --file environment.yml
conda activate screws310
```

This builds a minimal environment named `screws310` (Python 3.10 and pip — the scientific
libraries are installed by steps 4-5) and activates it. Run `conda activate screws310` in every
new terminal before using the pipeline.

### 4. Install bg3dtools

spinescrews depends on [bg3dtools](https://github.com/groisserb-hss/bg3dtools) (which provides the
`bg3dtools` and `spectral_match` packages). Clone it **next to** the spinescrews folder — the
`../` below keeps it out of this repository — and install it with the `mesh`, `viz`, and `graph`
extras **before** installing spinescrews:

```bash
git clone https://github.com/groisserb-hss/bg3dtools.git ../bg3dtools
pip install -e "../bg3dtools[mesh,viz,graph]"
```

`-e` is an "editable" install, so a later `git pull` inside `../bg3dtools/` updates the package
in place.

### 5. Install spinescrews

```bash
pip install -e ".[fast]"
```

The `[fast]` extra adds `embreex` (Embree-accelerated ray–mesh intersection), which markedly
speeds up canal-mesh construction in the accuracy step; it installs as a prebuilt wheel, so no
compiler is needed. Plain `pip install -e .` also works — that step is just slower.

This installs the pipeline's Python dependencies (NumPy, SciPy, and the rest of the scientific
stack, as declared in `pyproject.toml`), makes the `spinescrews` package importable, and installs
the console scripts
(`spinescrews-segment`, `spinescrews-preop`, `spinescrews-postop`, `spinescrews-align`,
`spinescrews-accuracy`). Every command supports `--help`.

### 6. External command-line tool

The Step 0 survey/conversion scripts are pure Python (they use `pydicom`, already installed), so
the only external tool needed is **dcm2niix**, for the actual DICOM-to-NIfTI conversion:

| Tool | Purpose | Install |
|------|---------|---------|
| `dcm2niix` | DICOM-to-NIfTI conversion | `conda install -c conda-forge dcm2niix` (any OS) — or `brew install dcm2niix` / `apt install dcm2niix` |

(`dcmdump` and `jq` are no longer required: the survey step reads DICOM headers directly.)

### 7. Segmentation backend

Choose **one** backend (TotalSegmentator is the default and recommended). Run from the repo root:

**TotalSegmentator** (default, Apache-2.0) — installs into the `screws310` environment; model
weights (~1.5 GB) download automatically on first use:

```bash
bash src/spinescrews/tools/totalseg_segmentor/setup.sh
```

On Windows (no bash), run `pip install TotalSegmentator` directly — that is all `setup.sh` does,
plus a smoke test — or run the script under Git Bash.

**Inria** (alternative, CC-BY-NC-SA-4.0) — creates a **separate** conda environment named
`verse20`. You do not activate it yourself; `spinescrews-segment --backend inria` calls it for you.

```bash
bash src/spinescrews/tools/inria_segmentor/setup.sh
```

On Windows, run this under Git Bash or WSL (it provisions a separate conda environment).

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

### Specimen directory layout

The pipeline works inside a per-specimen directory (`specimen_XX` below — name it whatever you
like). You supply the inputs; the pipeline creates everything under `analysis/`:

```
specimen_XX/
├── preop.nii.gz      # input: pre-operative CT  (from Step 0)
├── postop.nii.gz     # input: post-operative CT (from Step 0)
├── preop_plan.csv    # input: screw plan        (from the Hybrid Screw Planner)
├── config.yml        # optional: per-specimen setting overrides
└── analysis/         # created by the pipeline
    ├── 01_segmentation/ … 07_accuracy/   # one folder per step, each with a summary.json gate
    └── config_resolved.yml               # the exact settings used
```

A ready-to-run instance of this layout ships in [`sample/`](sample/); see
[Try it on the included sample data](#try-it-on-the-included-sample-data).

## Usage

Before your first run, make sure you have:

- [ ] Anaconda/Miniconda installed (install step 1)
- [ ] the repository cloned (step 2)
- [ ] the `screws310` environment created and activated (step 3)
- [ ] bg3dtools installed (step 4)
- [ ] spinescrews installed, so the `spinescrews-*` commands work (step 5)
- [ ] `dcm2niix` installed (step 6)
- [ ] one segmentation backend set up (step 7)
- [ ] CT scans converted to `preop.nii.gz` / `postop.nii.gz` ([Step 0](#step-0-dicom-preparation))
- [ ] a screw plan exported to `preop_plan.csv` ([Screw planning](#screw-planning-3d-slicer))

The pipeline then runs as numbered steps; every `spinescrews-*` command supports `--help`.

**Each time you open a new terminal**, move into the repo and activate the environment before
running anything:

```bash
cd /path/to/spinescrews      # the folder cloned in step 2
conda activate screws310
```

Your prompt should then begin with `(screws310)`. The `spinescrews-*` commands work from any
directory once the environment is active — a `command not found` error almost always means it
isn't — while the relative `sample/…` and `dicom_tools/…` paths below assume you are in the repo
root.

### Try it on the included sample data

The quickest way to confirm your install works end-to-end is to run the pipeline on the example
specimen bundled in [`sample/`](sample/):

| File | What it is |
|------|------------|
| `sample/preop.nii.gz` | pre-operative CT volume |
| `sample/postop.nii.gz` | post-operative CT volume (screws implanted) |
| `sample/preop_plan.csv` | 30 planned polyaxial screws (T2-L4, bilateral), in the Hybrid Screw Planner export format |

Because the converted volumes and the plan are already provided, you can skip
[Step 0](#step-0-dicom-preparation) (DICOM-to-NIfTI) and [screw planning](#screw-planning-3d-slicer),
and you don't need the `dcm2niix` tool from install step 6 — only install
steps 1-5 and a segmentation backend (step 7). `sample/` already matches the
[specimen layout](#specimen-directory-layout), so it doubles as the specimen directory. Run these
from the repo root, with `screws310` active:

```bash
# Step 1 — segment the vertebrae in the pre-op CT
spinescrews-segment --input sample/preop.nii.gz --output_dir sample

# Steps 2-6 — pre-op alignment, then post-op registration + screw detection
spinescrews-align sample

# Step 7 — planned-vs-detected accuracy
spinescrews-accuracy sample
```

Segmentation runs on CPU by default; add `--device gpu` (or `--device mps` on Apple Silicon) and/or `--fast` 
to the first command to speed it up. On the very first run TotalSegmentator also downloads its model weights (~1.5 GB).

Everything the pipeline produces lands in `sample/analysis/` (git-ignored, so it won't show up as
repository changes) — the per-screw measurements in `sample/analysis/07_accuracy/results.csv`, with
QC figures generated alongside each step. Delete `sample/analysis/` to start over.

### Run on your own data

Same three pipeline commands as the sample, but you supply the inputs first:

1. **Convert DICOM → NIfTI** ([Step 0](#step-0-dicom-preparation)) — writes `preop.nii.gz` and
   `postop.nii.gz` into your own `specimen_XX/` directory.
2. **Plan the screws** ([Screw planning](#screw-planning-3d-slicer)) — export `preop_plan.csv`
   into that same directory.
3. **Run the pipeline** (with `screws310` active):

   ```bash
   spinescrews-segment --input /path/to/specimen_XX/preop.nii.gz --output_dir /path/to/specimen_XX
   spinescrews-align    /path/to/specimen_XX     # steps 2-6
   spinescrews-accuracy /path/to/specimen_XX     # step 7
   ```

Per-screw results land in `specimen_XX/analysis/07_accuracy/results.csv`. The per-step sections
below cover each command's options (segmentation backend, running pre-op/post-op separately, etc.).

### Screw planning (3D Slicer)

`dicom_tools/` holds companion utilities for authoring pedicle-screw plans in
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
`dicom_tools/HybridScrewPlanner` under *Edit > Application Settings > Modules >
Additional module paths* and restarting; it then appears as "Hybrid Screw
Planner" in the *Planning* category. Self-tests: *Testing > Self-Tests*, or
`slicer.selfTests["HybridScrewPlanner"]()` (see
`dicom_tools/HybridScrewPlanner/tests.txt`).

**Burn endpoints into DICOM (for Mazor).** `burn_screw_endpoints.py` paints a
small high-HU sphere at every entry and tip from the planner CSV into a copy of
the source CT DICOM series (RAS→LPS conversion and DERIVED/SECONDARY tagging
handled), so the planned points appear as fiducials when the series is loaded
into navigation software such as Mazor. It runs on `pydicom` (already in the
`screws310` environment — no extra install):

```bash
python dicom_tools/burn_screw_endpoints.py \
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

### Step 0: DICOM preparation

These two scripts run as plain Python (`pydicom` + `dcm2niix`) — no bash — so they work on
Windows, macOS, and Linux. Use `survey_dicoms.py` to scan your DICOM directories and build a
metadata index:

```bash
python dicom_tools/survey_dicoms.py -o metadata.json /path/to/dicom_dir1 /path/to/dicom_dir2
```

Then use `convert_to_nii.py` to extract matching series as NIfTI. You can filter
by any series field (case-insensitive substring match), or run it without filters
for an interactive selection menu:

```bash
# Extract pre-op CT (filter by series description)
python dicom_tools/convert_to_nii.py metadata.json /path/to/specimen_XX preop series_description:"BONE STD"

# Extract post-op CT
python dicom_tools/convert_to_nii.py metadata.json /path/to/specimen_XX postop series_description:"MAZOR BONE"

# Interactive mode (no filters — presents a selection menu)
python dicom_tools/convert_to_nii.py metadata.json /path/to/specimen_XX preop
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

Requires the screw plan `specimen_XX/preop_plan.csv` exported by the
[Hybrid Screw Planner](#screw-planning-3d-slicer); this step compares planned vs. detected screws.

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
