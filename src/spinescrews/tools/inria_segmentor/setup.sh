#!/usr/bin/env bash
# setup.sh — One-click setup for the Inria vertebral segmentation backend.
#
# Clones the repo, creates a modern conda env (verse20), patches source files
# for CPU/ARM64 compatibility, and runs a smoke test.
#
# Usage:
#   bash tools/inria_segmentor/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/vertebrae_segmentation"
ENV_NAME="verse20"
# Pin to known-good commit — patches assume this exact source.
PINNED_COMMIT="08745b1f4f6942d8d39a525050e30b3469b4e6b9"

# ── 1. Clone repo ────────────────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo ">>> Repo already exists at $REPO_DIR — skipping clone"
else
    echo ">>> Cloning Inria vertebrae_segmentation repo..."
    git clone https://gitlab.inria.fr/spine/vertebrae_segmentation.git "$REPO_DIR"
    echo ">>> Checking out pinned commit $PINNED_COMMIT..."
    git -C "$REPO_DIR" checkout "$PINNED_COMMIT" --quiet
fi

# ── 2. Write environment_modern.yml ──────────────────────────────────────────
echo ">>> Writing environment_modern.yml..."
cat > "$REPO_DIR/environment_modern.yml" <<'ENVEOF'
name: verse20
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch>=2.0,<2.6
  - numpy>=1.21,<1.24
  - scipy>=1.7,<1.14
  - matplotlib
  - scikit-image
  - scikit-learn
  - nilearn
  - networkx
  - nibabel
  - simpleitk
  - jupyter
  - pandas
ENVEOF

# ── 3. Create conda env ─────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ">>> Conda env '$ENV_NAME' already exists — skipping creation"
else
    echo ">>> Creating conda env '$ENV_NAME' from environment_modern.yml..."
    conda env create -f "$REPO_DIR/environment_modern.yml"
fi

# ── 4. Write patch_for_cpu.py ────────────────────────────────────────────────
echo ">>> Writing patch_for_cpu.py..."
cat > "$REPO_DIR/patch_for_cpu.py" <<'PATCHEOF'
"""Patch Inria vertebrae_segmentation for CPU / ARM64 compatibility.

Applies two kinds of fixes:
  1. torch.load(...) calls get map_location='cpu' and weights_only=False
  2. A CUDA guard wraps model.to(torch.device("cuda")) in identify.py
"""
import re
from pathlib import Path

REPO = Path(__file__).parent

# ── torch.load patches ──────────────────────────────────────────────────────
TORCH_LOAD_FILES = [
    'segment_spine.py',
    'segment_vertebra.py',
    'identify.py',
    'locate.py',
]

old_load = 'state_dict = torch.load(model_file)'
new_load = "state_dict = torch.load(model_file, map_location='cpu', weights_only=False)"

for fname in TORCH_LOAD_FILES:
    fpath = REPO / fname
    if not fpath.exists():
        print(f'  SKIP (not found): {fname}')
        continue
    text = fpath.read_text()
    if new_load in text:
        print(f'  already patched: {fname}')
        continue
    if old_load not in text:
        print(f'  WARNING: expected pattern not found in {fname}')
        continue
    text = text.replace(old_load, new_load)
    fpath.write_text(text)
    print(f'  patched torch.load: {fname}')

# ── CUDA guard in identify.py ───────────────────────────────────────────────
identify = REPO / 'identify.py'
if identify.exists():
    text = identify.read_text()
    old_cuda = '    model.to(torch.device("cuda"))\n    model.eval()'
    new_cuda = ('    if torch.cuda.is_available():\n'
                '        model.to(torch.device("cuda"))\n'
                '    model.eval()')
    if new_cuda in text:
        print('  already patched: identify.py (CUDA guard)')
    elif old_cuda in text:
        text = text.replace(old_cuda, new_cuda)
        identify.write_text(text)
        print('  patched CUDA guard: identify.py')
    else:
        print('  WARNING: CUDA guard pattern not found in identify.py')

print('Patching complete.')
PATCHEOF

# ── 5. Run the patch ────────────────────────────────────────────────────────
echo ">>> Applying patches..."
conda run -n "$ENV_NAME" python "$REPO_DIR/patch_for_cpu.py"

# ── 6. Smoke test ───────────────────────────────────────────────────────────
echo ">>> Running smoke test (import torch + nibabel)..."
conda run -n "$ENV_NAME" python -c "import torch; import nibabel; print('Smoke test OK — torch', torch.__version__)"

echo ""
echo "=== Setup complete ==="
echo "Repo:  $REPO_DIR"
echo "Env:   $ENV_NAME"
echo ""
echo "Run segmentation with:"
echo "  python tools/run_segmentation.py --backend inria --input <nifti> --output_dir <dir>"
