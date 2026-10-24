[CmdletBinding()]
param(
    [string]$ManifestPath = "data/processed/image_manifest.parquet",
    [string]$OutputApk = "data/artifacts/cloud_friend_labeler.apk",
    [string]$PackageName = "cloud_mobile_labeler",
    [string]$RecipientEmail = "",
    [int]$MaxImageEdge = 1600,
    [int]$JpegQuality = 88,
    [int]$ImageLimit = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppRoot = Join-Path $ProjectRoot "android\friend-labeler"
$BuildRoot = Join-Path $AppRoot "build"
$GeneratedRoot = Join-Path $BuildRoot "generated"
$AssetsRoot = Join-Path $GeneratedRoot "assets"
$GeneratedJava = Join-Path $GeneratedRoot "java"
$ClassesDir = Join-Path $BuildRoot "classes"
$DexDir = Join-Path $BuildRoot "dex"
$ResCompiled = Join-Path $BuildRoot "compiled_res.zip"
$UnsignedApk = Join-Path $BuildRoot "unsigned.apk"
$ApkWithDex = Join-Path $BuildRoot "with_dex.apk"
$AlignedApk = Join-Path $BuildRoot "aligned.apk"
$Keystore = Join-Path $AppRoot "debug.keystore"
$ClassesJar = Join-Path $BuildRoot "classes.jar"

function Check-Exit([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return Join-Path $ProjectRoot $PathValue
}

function Reset-Directory([string]$PathValue) {
    if (Test-Path $PathValue) {
        $resolved = (Resolve-Path $PathValue).Path
        if (-not $resolved.StartsWith($BuildRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to delete outside Android build directory: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Find-Python {
    $candidates = @(
        $env:CLOUD_AESTHETICS_PYTHON,
        "D:\CloudAI\.venv\Scripts\python.exe",
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        "python"
    ) | Where-Object { $_ -and $_.Trim().Length -gt 0 }
    foreach ($candidate in $candidates) {
        try {
            & $candidate -c "import pandas, cv2, pyarrow" *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }
    throw "No Python with pandas, pyarrow, and cv2 found."
}

if (-not $env:ANDROID_HOME) {
    throw "ANDROID_HOME is not set."
}

$BuildToolsRoot = Join-Path $env:ANDROID_HOME "build-tools"
$BuildTools = Get-ChildItem $BuildToolsRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
if (-not $BuildTools) {
    throw "No Android build-tools found under $BuildToolsRoot."
}

$Aapt2 = Join-Path $BuildTools.FullName "aapt2.exe"
$D8 = Join-Path $BuildTools.FullName "d8.bat"
$Zipalign = Join-Path $BuildTools.FullName "zipalign.exe"
$Apksigner = Join-Path $BuildTools.FullName "apksigner.bat"
$AndroidJar = Join-Path $env:ANDROID_HOME "platforms\android-36\android.jar"
if (-not (Test-Path $AndroidJar)) {
    throw "android-36 platform not found: $AndroidJar"
}

$JavaBin = Split-Path (Get-Command java).Source
$Javac = Join-Path $JavaBin "javac.exe"
$Jar = Join-Path $JavaBin "jar.exe"
$Keytool = Join-Path $JavaBin "keytool.exe"

Reset-Directory $BuildRoot
New-Item -ItemType Directory -Force -Path $GeneratedRoot, $AssetsRoot, $GeneratedJava, $ClassesDir, $DexDir | Out-Null

$Python = Find-Python
$ResolvedManifest = Resolve-ProjectPath $ManifestPath
$env:FRIEND_LABELER_PROJECT_ROOT = $ProjectRoot
$env:FRIEND_LABELER_MANIFEST = $ResolvedManifest
$env:FRIEND_LABELER_ASSETS = $AssetsRoot
$env:FRIEND_LABELER_PACKAGE_NAME = $PackageName
$env:FRIEND_LABELER_RECIPIENT_EMAIL = $RecipientEmail
$env:FRIEND_LABELER_MAX_EDGE = [string]$MaxImageEdge
$env:FRIEND_LABELER_JPEG_QUALITY = [string]$JpegQuality
$env:FRIEND_LABELER_IMAGE_LIMIT = [string]$ImageLimit

@'
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

root = Path(os.environ["FRIEND_LABELER_PROJECT_ROOT"])
manifest_path = Path(os.environ["FRIEND_LABELER_MANIFEST"])
assets = Path(os.environ["FRIEND_LABELER_ASSETS"])
package_name = os.environ["FRIEND_LABELER_PACKAGE_NAME"]
recipient_email = os.environ.get("FRIEND_LABELER_RECIPIENT_EMAIL", "")
max_edge = int(os.environ["FRIEND_LABELER_MAX_EDGE"])
quality = int(os.environ["FRIEND_LABELER_JPEG_QUALITY"])
limit = int(os.environ["FRIEND_LABELER_IMAGE_LIMIT"])

if not manifest_path.is_absolute():
    manifest_path = root / manifest_path
if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)

if assets.exists():
    shutil.rmtree(assets)
(assets / "images").mkdir(parents=True, exist_ok=True)

def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe[:72] or "image"

def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

manifest = pd.read_parquet(manifest_path)
if limit > 0:
    manifest = manifest.head(limit)

rows = []
asset_bytes = 0
for index, (_, row) in enumerate(manifest.iterrows()):
    source = project_path(str(row["relative_path"]))
    if not source.exists():
        continue
    data = np.fromfile(str(source), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        continue
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / float(max(width, height)))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        continue
    image_id = str(row["image_id"])
    filename = f"{len(rows):05d}_{image_id}_{safe_name(source.stem)}.jpg"
    target = assets / "images" / filename
    target.write_bytes(encoded.tobytes())
    asset_bytes += target.stat().st_size
    rows.append(
        {
            "image_id": image_id,
            "path": f"images/{filename}",
            "original_relative_path": str(row["relative_path"]),
            "width": int(row.get("width", width) or width),
            "height": int(row.get("height", height) or height),
            "split_group_id": str(row.get("split_group_id", "")),
        }
    )

payload = {
    "version": 2,
    "package_name": package_name,
    "recipient_email": recipient_email,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "images": rows,
}
(assets / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(f"Prepared {len(rows)} Android assets ({asset_bytes / 1024 / 1024:.1f} MB)")
'@ | & $Python -
Check-Exit "asset generation"

$ManifestXml = Join-Path $AppRoot "src\main\AndroidManifest.xml"
$ResDir = Join-Path $AppRoot "src\main\res"
$JavaDir = Join-Path $AppRoot "src\main\java"

& $Aapt2 compile --dir $ResDir -o $ResCompiled
Check-Exit "aapt2 compile"
& $Aapt2 link `
    -I $AndroidJar `
    --manifest $ManifestXml `
    -R $ResCompiled `
    -A $AssetsRoot `
    --java $GeneratedJava `
    --min-sdk-version 23 `
    --target-sdk-version 36 `
    --version-code 1 `
    --version-name "1.0" `
    --auto-add-overlay `
    -o $UnsignedApk
Check-Exit "aapt2 link"

$Sources = @()
$Sources += Get-ChildItem $JavaDir -Recurse -Filter *.java | Select-Object -ExpandProperty FullName
$Sources += Get-ChildItem $GeneratedJava -Recurse -Filter *.java | Select-Object -ExpandProperty FullName
& $Javac -encoding UTF-8 -source 8 -target 8 -classpath $AndroidJar -d $ClassesDir $Sources
Check-Exit "javac"

& $Jar cf $ClassesJar -C $ClassesDir .
Check-Exit "jar classes"
& $D8 --min-api 23 --lib $AndroidJar --output $DexDir $ClassesJar
Check-Exit "d8"
Copy-Item -LiteralPath $UnsignedApk -Destination $ApkWithDex -Force
& $Jar uf $ApkWithDex -C $DexDir classes.dex
Check-Exit "jar apk"
& $Zipalign -p -f 4 $ApkWithDex $AlignedApk
Check-Exit "zipalign"

if (-not (Test-Path $Keystore)) {
    & $Keytool -genkeypair -v `
        -keystore $Keystore `
        -storepass android `
        -alias androiddebugkey `
        -keypass android `
        -keyalg RSA `
        -keysize 2048 `
        -validity 10000 `
        -dname "CN=Cloud Aesthetics Android Debug,O=Cloud Aesthetics,C=DE" | Out-Null
    Check-Exit "keytool"
}

$ResolvedOutput = Resolve-ProjectPath $OutputApk
New-Item -ItemType Directory -Force -Path (Split-Path $ResolvedOutput) | Out-Null
& $Apksigner sign `
    --ks $Keystore `
    --ks-pass pass:android `
    --key-pass pass:android `
    --out $ResolvedOutput `
    $AlignedApk
Check-Exit "apksigner sign"
& $Apksigner verify --verbose --print-certs $ResolvedOutput
Check-Exit "apksigner verify"

$Apk = Get-Item $ResolvedOutput
Write-Host "APK written: $($Apk.FullName)"
Write-Host "APK size: $([math]::Round($Apk.Length / 1MB, 1)) MB"
