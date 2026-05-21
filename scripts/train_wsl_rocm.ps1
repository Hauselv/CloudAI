param(
    [string]$Distro = "Ubuntu",
    [string]$ProjectDir = "/mnt/d/CloudAI/cloud-aesthetics",
    [string]$VenvDir = "~/.venvs/cloud-aesthetics-rocm",
    [string]$Config = "configs/models/deep_cnn_rocm.yaml",
    [string]$HsaOverrideGfxVersion = ""
)

$ErrorActionPreference = "Stop"

$override = ""
if ($HsaOverrideGfxVersion.Trim()) {
    $override = "export HSA_OVERRIDE_GFX_VERSION='$HsaOverrideGfxVersion';"
}

$command = @"
set -euo pipefail
cd '$ProjectDir'
if [ ! -d $VenvDir ]; then
  echo 'Missing WSL venv. Run scripts/setup_wsl_rocm_env.sh inside WSL first.' >&2
  exit 2
fi
source $VenvDir/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
$override
python -m cloud_aesthetics.cli train --config '$Config'
"@

wsl.exe -d $Distro -- bash -lc $command
