# TotalSegmentator Backend

CT organ/structure segmentation using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).
Licensed under **Apache-2.0**.

Segments vertebrae (C1–L5) plus sacral base (`vertebrae_S1`) and `sacrum`.
Uses its own label numbering (different from VerSe); the wrapper in `run_segmentation.py` remaps to project integers via `seg_val`.

## Label mapping

| TotalSeg structure | TotalSeg int | Project name | Project int |
|--------------------|-------------|--------------|-------------|
| `vertebrae_C1`–`C7` | 24–18 | C1–C7 | 1–7 |
| `vertebrae_T1`–`T12` | 17–6 | T1–T12 | 8–19 |
| `vertebrae_L1`–`L5` | 5–1 | L1–L5 | 20–24 |
| `vertebrae_S1` | 26 | LS | 25 |
| `sacrum` | 25 | SA | — |

## Prerequisites

- `screws310` conda env active
- ~1.5 GB disk for model weights (downloaded automatically on first run to `tools/totalseg_segmentor/nnunet/results/`)

## Installation

```bash
bash tools/totalseg_segmentor/setup.sh
```

This will:
1. Verify the `screws310` conda env is active
2. `pip install TotalSegmentator` into the current env
3. Run a smoke test (import check)

## Usage

```bash
conda activate screws310
python tools/run_segmentation.py --backend totalseg --input <nifti> --output_dir <dir>
```

### Options

- `--device cpu|gpu|mps` — compute device (default: `cpu`)
- `--fast` — lower resolution, faster inference
