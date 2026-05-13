#!/usr/bin/env bash
# setup.sh — Install TotalSegmentator into the active screws310 env.
#
# Usage:
#   conda activate screws310
#   bash tools/totalseg_segmentor/setup.sh

set -euo pipefail

# ── 1. Verify screws310 env is active ─────────────────────────────────────
if [ "${CONDA_DEFAULT_ENV:-}" != "screws310" ]; then
    echo "ERROR: screws310 conda env is not active."
    echo "Run:  conda activate screws310"
    exit 1
fi

# ── 2. Install TotalSegmentator ───────────────────────────────────────────
echo ">>> Installing TotalSegmentator..."
pip install TotalSegmentator

# ── 3. Smoke test ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TOTALSEG_HOME_DIR="$SCRIPT_DIR"

echo ">>> Running smoke test..."
python -c "from totalsegmentator.python_api import totalsegmentator; print('Smoke test OK')"

echo ""
echo "=== Setup complete ==="
echo "Model weights will be downloaded on first run (~1.5 GB) to $SCRIPT_DIR/nnunet/results/"
echo ""
echo "Run segmentation with:"
echo "  python tools/run_segmentation.py --backend totalseg --input <nifti> --output_dir <dir>"
