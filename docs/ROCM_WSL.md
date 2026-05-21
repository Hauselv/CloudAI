# ROCm / WSL GPU Training

This project can train on ROCm when PyTorch exposes an AMD GPU through the normal `torch.cuda` API. On WSL this requires a working AMD WSL GPU stack before Cloud Aesthetics can use the GPU.

## Current Machine Snapshot

Checked on 2026-05-22:

- Windows GPU: AMD Radeon RX 6700 XT
- WSL: installed and active as Ubuntu WSL2
- Ubuntu: 24.04.4 LTS
- WSL kernel: 6.6.87.2
- WSL GPU bridge: `/dev/dxg` exists
- ROCm tools in WSL: not installed
- PyTorch in WSL: not installed
- Windows Python environment: `torch 2.11.0+cpu`, no CUDA/ROCm backend

The RX 6700 XT is not listed in AMD's current ROCDXG WSL supported-products table. Treat ROCm-on-WSL for this exact GPU as experimental unless AMD expands support or the machine is switched to a supported GPU.

## Official Path

AMD's current WSL path is ROCDXG:

1. Install a WSL-compatible AMD Adrenalin driver.
2. Use Ubuntu 24.04 or 22.04 under WSL2.
3. Install/build `librocdxg`.
4. Install ROCm and a matching ROCm PyTorch wheel.
5. Validate with `torch.cuda.is_available()` and `torch.version.hip`.

Useful references:

- AMD WSL guide: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html>
- ROCDXG repository: <https://github.com/ROCm/librocdxg>
- AMD PyTorch for ROCm guide: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html>

## Project Commands

Run a non-destructive probe from PowerShell:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/CloudAI/cloud-aesthetics && bash scripts/probe_wsl_rocm.sh"
```

Prepare a WSL Python environment:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/CloudAI/cloud-aesthetics && bash scripts/setup_wsl_rocm_env.sh"
```

Run training through WSL after ROCm PyTorch validates:

```powershell
powershell -ExecutionPolicy Bypass -File D:\CloudAI\cloud-aesthetics\scripts\train_wsl_rocm.ps1
```

The ROCm-specific config is:

```text
configs/models/deep_cnn_rocm.yaml
```

It disables AMP by default because ROCm mixed precision support varies by wheel, GPU, and driver. The model still writes device, HIP, split sizes, best epoch, checkpoint paths, and software versions into `summary.json`.

## Validation

A usable ROCm environment should pass:

```bash
HSA_ENABLE_DXG_DETECTION=1 python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("gpu", torch.cuda.is_available())
print("name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Expected:

- `hip` is not `None`
- `gpu` is `True`
- `name` prints the Radeon device or a ROCm architecture name

If this fails, stay on CPU or use a supported GPU/driver/runtime combination. Do not trust timing or accuracy comparisons between CPU and a partially working unsupported ROCm stack.
