# Cloud Aesthetics Research Workbench

Local-first Python tooling for collecting cloud photo ratings, training models that predict group aesthetic scores, and explaining what drives those predictions with interpretable features and image-space attributions.

## What is included

- Dataset ingestion and manifest generation
- Per-rater scalar and pairwise annotation storage
- Rating aggregation, normalization, and pseudo-pair generation
- Leakage-aware dataset split generation
- Hand-crafted feature extraction for color, texture, and composition
- Baseline regressors and pairwise ranking models
- Deep transfer model scaffolding for regression plus pairwise ranking
- Explainability utilities for heatmaps, feature importance, regions, and text summaries
- Streamlit UI for rating, training, inspection, and comparison

## Quickstart

```powershell
python -m venv D:\CloudAI\.venv
D:\CloudAI\.venv\Scripts\python.exe -m pip install -e D:\CloudAI\cloud-aesthetics[dev,deep,explain]
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli download-web-dataset --config D:\CloudAI\cloud-aesthetics\configs\dataset\web_sample.yaml
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli ingest-images
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli aggregate-labels
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli extract-features --config D:\CloudAI\cloud-aesthetics\configs\features\v1.yaml
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli train --config D:\CloudAI\cloud-aesthetics\configs\models\baseline.yaml
D:\CloudAI\.venv\Scripts\python.exe -m streamlit run D:\CloudAI\cloud-aesthetics\src\cloud_aesthetics\app\Home.py
```

This workflow avoids PowerShell execution-policy issues because it does not require `Activate.ps1`.

## Web Test Dataset

The command below downloads a local test dataset of openly licensed cloud images from Wikimedia Commons and stores source/license metadata next to the raw data.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli download-web-dataset --config D:\CloudAI\cloud-aesthetics\configs\dataset\web_sample.yaml
```

Images are saved under `data/raw/images/web_sample`, while attribution and license metadata are written to `data/raw/metadata/web_sample_metadata.parquet` and `.csv`.

## Core philosophy

- Preserve raw data; derive all aggregates and normalized labels.
- Separate interpretable features from learned representations.
- Keep experiments reproducible with explicit config files and cached artifacts.
- Treat explanations as model-attribution evidence, not causal proof.

## Main CLI commands

- `python -m cloud_aesthetics.cli import-images --source <folder> --dataset-name private_clouds`
- `python -m cloud_aesthetics.cli ingest-images`
- `python -m cloud_aesthetics.cli export-friend-package --output data/artifacts/cloud_labeling_friend_package.zip`
- `python -m cloud_aesthetics.cli import-friend-labels --bundle <friend_labels.json>`
- `python -m cloud_aesthetics.cli aggregate-labels`
- `python -m cloud_aesthetics.cli extract-features --config configs/features/v1.yaml`
- `python -m cloud_aesthetics.cli train --config configs/models/baseline.yaml`
- `python -m cloud_aesthetics.cli evaluate --run <run_id>`
- `python -m cloud_aesthetics.cli explain --run <run_id> --image-id <image_id>`

## Friend Standalone Labeling Package

You can export a no-install browser package for friends. It includes the current manifest images, scalar ratings, pairwise preferences, JSON export, and browser-side import/crop support for their own images.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli export-friend-package --output D:\CloudAI\cloud-aesthetics\data\artifacts\cloud_labeling_friend_package.zip --package-name cloud_labeling_friend_package
```

Send the zip file to a friend. They unzip it, open `index.html`, rate images, and send back the exported `*_labels.json` file.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli import-friend-labels --bundle D:\Path\To\friend_labels.json
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli ingest-images
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli aggregate-labels
```

If friends import their own images in the standalone app, the returned JSON contains those images and generated crops as embedded data. The import command restores them under `data/raw/images/friend_imports`; run `ingest-images` afterward so they enter the manifest.

## Private Image Import and Cropping

Private photo folders can be imported directly into `data/raw/images/<dataset-name>`. By default, the import keeps copied originals and generates variable-size sky/cloud crops selected by simple HSV sky/cloud masks.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli import-images --source D:\Path\To\PrivateCloudPhotos --dataset-name private_clouds
```

The importer writes derivative metadata to `data/raw/metadata/image_derivatives.parquet`. Manifest ingestion uses that metadata to keep each original and its crops in the same `split_group_id`, which avoids train/test leakage when models are trained later.

## Directory highlights

- `configs/`: dataset, feature, model, and app configs
- `data/`: raw data, processed tables, cached features and embeddings, run artifacts
- `src/cloud_aesthetics/`: package code
- `notebooks/`: exploratory notebooks
- `tests/`: unit and smoke tests
