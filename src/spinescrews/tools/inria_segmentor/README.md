# Inria Vertebrae Segmentation

Deep-learning vertebral segmentation from [Inria/SPINE](https://gitlab.inria.fr/spine/vertebrae_segmentation).
Licensed under **CC-BY-NC-SA-4.0**.

Uses the VerSe convention to produce integer labels 1-25 for free vertebrae (C1-L5 plus sacral base).
In our pipeline, label 25 = LS (the lumbosacral junction / sacral base).

## Prerequisites

- Anaconda or Miniconda (`conda` on PATH)
- ~2 GB disk for model weights + conda environment

## Installation

One command:

```bash
bash tools/inria_segmentor/setup.sh
```

This will:
1. Clone the repo into `tools/inria_segmentor/vertebrae_segmentation/` (pinned to a known-good commit)
2. Create the `verse20` conda environment with compatible dependencies
3. Patch source files for CPU/ARM64 compatibility
4. Run a smoke test

## Usage

### Via wrapper (recommended)

```bash
conda activate screws310
spinescrews-segment --backend inria --input <nifti> --output_dir <dir>
```

### Standalone

```bash
conda activate verse20
cd tools/inria_segmentor/vertebrae_segmentation
python test.py -D <input.nii.gz> -S <output_dir>
```

## What the patches fix

- **`torch.load` map_location**: adds `map_location='cpu'` and `weights_only=False` so models load on CPU-only and ARM64 machines without error.
- **CUDA guard**: wraps `model.to(torch.device("cuda"))` in a `torch.cuda.is_available()` check so non-GPU machines don't crash at inference time.

## Pinned commit

The repo is pinned to commit `08745b1` for patch compatibility.
If you update the commit, verify the patches in `setup.sh` still apply correctly.
