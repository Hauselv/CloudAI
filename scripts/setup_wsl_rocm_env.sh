#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/CloudAI/cloud-aesthetics}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/cloud-aesthetics-rocm}"
ROCM_WHL_VERSION="${ROCM_WHL_VERSION:-7.2.1}"

echo "Preparing Cloud Aesthetics WSL/ROCm Python environment"
echo "project: $PROJECT_DIR"
echo "venv:    $VENV_DIR"
echo "rocm:    $ROCM_WHL_VERSION"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory not found. Set PROJECT_DIR=/path/to/cloud-aesthetics." >&2
  exit 2
fi

if ! grep -q 'VERSION_ID="24.04"\|VERSION_ID="22.04"' /etc/os-release; then
  echo "Ubuntu 24.04 or 22.04 is expected for AMD ROCm on WSL." >&2
fi

if [ ! -e /dev/dxg ]; then
  echo "Missing /dev/dxg. WSL GPU virtualization is not visible yet." >&2
  echo "Update WSL and install the AMD WSL-compatible Windows driver before continuing." >&2
  exit 3
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is missing. Install python3 and python3-venv in Ubuntu first." >&2
  exit 4
fi

if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
  echo "Could not create venv. Install python3-venv in Ubuntu first:" >&2
  echo "  sudo apt update && sudo apt install -y python3-venv python3-pip" >&2
  exit 5
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e "$PROJECT_DIR[dev,explain]"

cat <<'NOTE'

Base Python environment is ready.

ROCm runtime status is intentionally not forced by this script:
- Current AMD WSL/ROCDXG support depends on the Windows driver, /dev/dxg,
  librocdxg, and the official GPU compatibility matrix.
- RX 6700 XT is not listed in the current ROCDXG supported-products list.
- If you still want to try an unsupported RDNA2 path, do it explicitly and
  keep it separate from normal project setup.

After installing a compatible ROCm/ROCDXG runtime, install matching ROCm
PyTorch and Torchvision wheels into this venv, then validate:

  source "$VENV_DIR/bin/activate"
  HSA_ENABLE_DXG_DETECTION=1 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version, 'hip', None)); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

Run project training from WSL:

  cd "$PROJECT_DIR"
  HSA_ENABLE_DXG_DETECTION=1 python -m cloud_aesthetics.cli train --config configs/models/deep_cnn_rocm.yaml

NOTE
