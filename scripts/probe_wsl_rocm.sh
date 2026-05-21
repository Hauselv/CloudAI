#!/usr/bin/env bash
set -euo pipefail

echo "== WSL / OS =="
uname -a
cat /etc/os-release | sed -n 's/^\(PRETTY_NAME\|VERSION_ID\)=/\1=/p'

echo
echo "== GPU device nodes =="
for path in /dev/dxg /dev/kfd /dev/dri; do
  if [ -e "$path" ]; then
    ls -ld "$path"
  else
    echo "missing: $path"
  fi
done

echo
echo "== ROCm tools =="
if command -v rocminfo >/dev/null 2>&1; then
  echo "rocminfo: $(command -v rocminfo)"
  HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}" rocminfo | sed -n '1,120p'
else
  echo "rocminfo: missing"
fi

echo
echo "== Python / PyTorch =="
python3 --version || true
python3 - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc!r}")
else:
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", getattr(torch.version, "cuda", None))
    print("hip_version:", getattr(torch.version, "hip", None))
    if torch.cuda.is_available():
        print("device_name:", torch.cuda.get_device_name(0))
PY

echo
echo "== Project =="
if [ -d /mnt/d/CloudAI/cloud-aesthetics ]; then
  echo "repo: /mnt/d/CloudAI/cloud-aesthetics"
else
  echo "repo not mounted at /mnt/d/CloudAI/cloud-aesthetics"
fi
