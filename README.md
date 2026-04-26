# Cloud Aesthetics Research Workbench

Local-first Python tooling for collecting cloud photo ratings, training models that predict aesthetic scores, and explaining what drives those predictions with interpretable features, image-space attributions, and Grad-CAM for deep CNN runs.

## What Is Included

- Dataset ingestion and manifest generation
- Private image import with copied originals and variable sky/cloud crops
- Import batch IDs for later batch analysis
- Per-rater scalar and pairwise annotation storage
- 0.0-10.0 scalar ratings in 0.5 steps
- Skip/exclusion flags for bad crops, satellite images, airplane shots, duplicates, or non-cloud training data
- Leakage-aware train/validation/test splits that keep originals and derived crops together
- Hand-crafted feature extraction for color, texture, and composition
- Baseline regressors and pairwise ranking models
- Deep transfer model scaffolding for regression
- Grad-CAM preparation for Deep-CNN runs with selectable ResNet layers
- Streamlit UI for importing, rating, training, inspection, batch reporting, and comparison
- Optional standalone no-install friend labeling package

## Quickstart

```powershell
python -m venv D:\CloudAI\.venv
D:\CloudAI\.venv\Scripts\python.exe -m pip install -e D:\CloudAI\cloud-aesthetics[dev,deep,explain]
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli ingest-images
D:\CloudAI\.venv\Scripts\python.exe -m streamlit run D:\CloudAI\cloud-aesthetics\src\cloud_aesthetics\app\Home.py
```

This avoids PowerShell execution-policy issues because it does not require `Activate.ps1`.

## Main CLI Commands

- `python -m cloud_aesthetics.cli import-images --source <folder> --dataset-name private_clouds`
- `python -m cloud_aesthetics.cli ingest-images`
- `python -m cloud_aesthetics.cli aggregate-labels`
- `python -m cloud_aesthetics.cli extract-features --config configs/features/v1.yaml`
- `python -m cloud_aesthetics.cli train --config configs/models/baseline.yaml`
- `python -m cloud_aesthetics.cli train --config configs/models/deep_cnn_gradcam.yaml`
- `python -m cloud_aesthetics.cli analyze-batch --run <run_id> --batch-id <import_batch_id>`
- `python -m cloud_aesthetics.cli explain --run <run_id> --image-id <image_id>`
- `python -m cloud_aesthetics.cli export-friend-package --output data/artifacts/cloud_labeling_friend_package.zip`
- `python -m cloud_aesthetics.cli import-friend-labels --bundle <friend_labels.json>`

## Streamlit Workflow

Start the local app:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m streamlit run D:\CloudAI\cloud-aesthetics\src\cloud_aesthetics\app\Home.py
```

Pages:

- `Import`: import private folders, generate crops, and optionally start batch analysis after import.
- `Rate`: label images, randomize or diversify order, use hotkeys, and skip unsuitable images.
- `Gallery`: browse active originals/crops as tiles with labeled/open filters.
- `Exclusions`: restore skipped images to the active pool.
- `Train`: aggregate labels, refresh features, train models, and inspect split/test metrics.
- `Inspect`: select model runs, pick images from thumbnails, view predictions, feature diagnostics, and heatmaps.
- `Batch Report`: analyze import batches with a local model and review top/low/uncertain predictions.

## Private Image Import And Cropping

Private photo folders can be imported directly into `data/raw/images/<dataset-name>`. The importer keeps copied originals and generates variable-size sky/cloud crops using local HSV sky/cloud heuristics.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli import-images --source D:\Path\To\PrivateCloudPhotos --dataset-name private_clouds
```

The importer writes derivative metadata to `data/raw/metadata/image_derivatives.parquet`. Manifest ingestion uses that metadata to:

- keep each original and its crops in the same `split_group_id`
- avoid train/test leakage
- track `import_batch_id` for later batch analysis

## Labeling And Exclusions

Scalar labels are stored locally under `data/raw/metadata/ratings/` as per-rater CSV files. The current scale is 0.0-10.0 in 0.5 steps. Older 1-10 labels remain valid.

When an image does not belong in training, use `Skip / exclude` in the `Rate` page. Exclusions are stored in `data/raw/metadata/exclusions.csv`; image files are not deleted. `ingest-images` filters active exclusions out of the manifest, and `Exclusions` can restore them.

## Training And Validation

Recommended sequence:

1. `Aggregate labels`
2. `Extract features`
3. `Train selected config`

`aggregate-labels` creates grouped splits:

- roughly 20% held-out `test` data
- remaining `dev` data split into validation folds
- originals and crops stay together via `split_group_id`

Use `configs/models/baseline.yaml` first. It is fast and gives useful signal with fewer labels. Baseline models provide feature diagnostics and fallback/diagnostic maps, but not true Grad-CAM.

## Grad-CAM Analysis

True Grad-CAM requires a Deep-CNN run:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli train --config D:\CloudAI\cloud-aesthetics\configs\models\deep_cnn_gradcam.yaml
```

The current Grad-CAM config is conservative:

- `resnet18`
- `image_size: 224`
- `batch_size: 4`
- `epochs: 5`
- `freeze_backbone: true`

After training, open `Inspect`, choose the deep run, pick an image from the thumbnail grid, and click `Generate explanation`. The analysis stays cached while you switch heatmap sources.

Available heatmap sources can include:

- `input_gradient`
- `gradcam:backbone.layer1`
- `gradcam:backbone.layer2`
- `gradcam:backbone.layer3`
- `gradcam:backbone.layer4`

Layer guidance:

- `layer4`: coarser semantic regions, usually best first view.
- `layer3`: somewhat finer cloud structures.
- `layer1`/`layer2`: low-level edges/textures and often noisier.

Baseline runs do not produce model-native Grad-CAM. They show feature diagnostics such as cloud mask, edge/contrast, brightness, saturation, saliency, or fallback edge maps.

## Batch Reports

Batch reports connect imports and trained models. After importing a folder, use its `import_batch_id`:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli analyze-batch --run <run_id> --batch-id <import_batch_id>
```

The `Batch Report` page shows:

- image count, mean score, top score, uncertainty
- ranked predictions
- top/low/uncertain tile groups
- detail view with explanation overlay
- top features and concepts

This helps triage a new batch before labeling everything.

## Friend Standalone Labeling Package

Export a no-install browser package for friends:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli export-friend-package --output D:\CloudAI\cloud-aesthetics\data\artifacts\cloud_labeling_friend_package.zip --package-name cloud_labeling_friend_package
```

Friends unzip it, open `index.html`, rate images, and send back the exported `*_labels.json`.

Import returned labels:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli import-friend-labels --bundle D:\Path\To\friend_labels.json
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli ingest-images
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli aggregate-labels
```

If friends import their own images in the standalone app, the returned JSON embeds those images and crops. The import command restores them under `data/raw/images/friend_imports`.

## Web Test Dataset

The project can download a local test dataset of openly licensed cloud images from Wikimedia Commons:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli download-web-dataset --config D:\CloudAI\cloud-aesthetics\configs\dataset\web_sample.yaml
```

Images are saved under `data/raw/images/web_sample`; attribution and license metadata are written to `data/raw/metadata/web_sample_metadata.parquet` and `.csv`.

## Current Open Work

- Add stronger real sky/cloud segmentation, possibly SAM or a dedicated semantic segmentation model.
- Improve AMD GPU path. A Radeon 6700 XT is powerful, but PyTorch on Windows usually will not use AMD GPUs directly; WSL2/ROCm or CPU training should be evaluated.
- Add more Deep-CNN configs once labels are stronger, for example EfficientNet-B0 or ConvNeXt-Tiny if local dependencies support them.
- Add early stopping and richer checkpoint metadata for deep training.
- Improve Grad-CAM UX with side-by-side layer comparison and saved report images.
- Add CI/PR automation after GitHub CLI is installed.
- Decide whether generated processed data should stay local-only or move to a data artifact system.
- Collect more labels. Baseline runs are useful around 150-300 labels; Grad-CAM becomes more meaningful once deep models have enough signal, roughly 300-500+ labels.

## Data Privacy Notes

Raw private images, labels, exclusions, processed features, and model artifacts are local data. Be careful before committing or pushing files under `data/raw`, `data/processed`, or `data/artifacts`, because they may contain private or derived information.

Normal code commits should include `configs/`, `src/`, `tests/`, and documentation, not private datasets or generated feature tables.

## Directory Highlights

- `configs/`: dataset, feature, model, and app configs
- `data/`: raw data, processed tables, cached features and embeddings, run artifacts
- `src/cloud_aesthetics/`: package code
- `notebooks/`: exploratory notebooks
- `tests/`: unit and smoke tests
