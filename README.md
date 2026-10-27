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
- Deterministic interpretation cards that translate scores, uncertainty, features, concepts, and cloud-type estimates into review actions
- Optional local Ollama paraphrasing for explanation text; the structured deterministic result remains the source of truth
- Streamlit UI for importing, rating, training, inspection, source-group review, batch reporting, and comparison
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

- `python -m cloud_aesthetics.cli status` (which pipeline stages exist, which are stale)
- `python -m cloud_aesthetics.cli refresh [--features missing|full|skip] [--train-config <yaml>]`
- `python -m cloud_aesthetics.cli import-images --source <folder> --dataset-name private_clouds`
- `python -m cloud_aesthetics.cli ingest-images`
- `python -m cloud_aesthetics.cli aggregate-labels`
- `python -m cloud_aesthetics.cli extract-features --config configs/features/v1.yaml`
- `python -m cloud_aesthetics.cli extract-embeddings [--model clip_vit_b32]`
- `python -m cloud_aesthetics.cli zero-shot`
- `python -m cloud_aesthetics.cli judge-images [--model claude-opus-5-5] [--limit N] [--no-only-labeled]`
- `python -m cloud_aesthetics.cli judge-report --rater llm_deepseek_v4_1_flash`
- `python -m cloud_aesthetics.cli judge-triage`
- `python -m cloud_aesthetics.cli contrast [--band-a 9.5-10] [--band-b 7.5-8.5] [--crop-class original]`
- `python -m cloud_aesthetics.cli extract-context [--no-weather] [--refresh]`
- `python -m cloud_aesthetics.cli counterfactual-sweep [--band 7.5-8.5] [--crop-class original] [--limit 60]`
- `python -m cloud_aesthetics.cli train --config configs/models/embedding_clip.yaml`
- `python -m cloud_aesthetics.cli train --config configs/models/concept_bottleneck_originals.yaml` (readable concept model)
- `python -m cloud_aesthetics.cli train --config configs/models/baseline.yaml`
- `python -m cloud_aesthetics.cli train --config configs/models/deep_cnn_gradcam.yaml`
- `powershell -ExecutionPolicy Bypass -File D:\CloudAI\cloud-aesthetics\scripts\train_wsl_rocm.ps1`
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

- `Home`: pipeline status panel (raw images, manifest, ratings, labels, splits, features, latest run) with stale markers and one-click refresh, plus rating distribution by image kind.
- `Import`: import private folders, generate crops, and optionally start batch analysis after import.
- `Rate`: label images with score buttons and number keys (0–9, `+` for 10, `.` for ±½, Enter saves, Backspace undoes), optional quick mode (digit saves immediately), previous ratings shown per image, judge scores hidden by default, image-kind filter, label orders including the judge queue, source photo shown for crops. The second tab runs duels among your top-rated images (hotkeys A/L/T/N) and shows a Bradley-Terry ranking next to the absolute scores.
- `Gallery`: browse active originals/crops as tiles with labeled/open filters.
- `Exclusions`: restore skipped images to the active pool.
- `Source Groups`: review originals and their derived crops together, then exclude a single image or whole group.
- `Train`: aggregate labels, refresh features, train models, and inspect split/test metrics.
- `Inspect`: select model runs, pick images from thumbnails, view predictions, feature diagnostics, and heatmaps.
- `Batch Report`: analyze import batches or legacy capture sessions with a local model and review top/low/uncertain/review-first predictions.
- `Rater Comparison`: human raters versus model raters (LLM judges), agreement per group and largest disagreements.
- `Triage`: judge-based labeling queue, uncertain images and exclusion proposals to accept or ignore.
- `Was wäre wenn`: counterfactual explorer. Edit a photo with sliders (contrast, saturation, exposure, warmth, zoom, blur), see the CLIP model's score react live, read sensitivity curves per operator and the single edit with the largest gain. `counterfactual-sweep --band 7.5-8.5` averages the same curves over a human score band and the page shows the result.
- `Konzeptmodell`: concept bottleneck model. A Lasso on named concepts (CLIP concepts, light, judge flags, palette, scene fraction) with bootstrap stability, a vocabulary comparison (which family of words explains the rating out-of-fold), per-image waterfall of contributions and a residual gallery of images the concepts cannot explain.
- `10 vs 8`: compare two human score bands across hand-crafted features, CLIP zero-shot attributes and judge attributes (effect sizes, FDR-corrected q-values, top-end trends, example tiles). CLI equivalent: `contrast --band-a 9.5-10 --band-b 7.5-8.5 --crop-class original`.

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

Scalar labels are stored locally under `data/raw/metadata/ratings/` as per-rater CSV files. The current scale is 0.0-10.0 in 0.5 steps. Older 1-10 labels remain valid. Undo never rewrites those files: withdrawn ratings are listed in `ratings/ratings_void.csv` and filtered out when ratings are loaded.

Pairwise duels (`data/raw/metadata/pairwise/<rater>.csv`) feed a Bradley-Terry fit during `aggregate-labels`; `aggregated_labels.parquet` carries `bt_score` (0-10 scaled) and `duels` per image.

When an image does not belong in training, use `Skip / exclude` in the `Rate` page or exclude it from `Source Groups`. Exclusions are stored in `data/raw/metadata/exclusions.csv`; image files are not deleted. The Streamlit pages and training commands filter active exclusions at read time, and `Exclusions` can restore them.

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

Baseline runs also store global SHAP importance per model (`shap_importance_<model>.parquet`, top features in `summary.json`) and explanations use exact SHAP contributions for tree and linear models. The Train page shows a calibration view per run (actual vs predicted with bin means, residuals, spread ratio, expected calibration error, conformal interval coverage) and the SHAP importance chart.

Ranking runs (`kind: ranking`, `configs/models/ranking.yaml` on features or `ranking_clip.yaml` on CLIP embeddings) train a pairwise logistic regression on dev pairs (explicit duels plus pseudo-pairs) and report pairwise accuracy on test pairs plus the Spearman agreement of the induced image scores with `mean_score` and `bt_score`. `configs/models/embedding_clip_bt.yaml` regresses the Bradley-Terry strength instead of the absolute score once at least 40 images have duels.

Baseline runs evaluate with grouped k-fold cross-validation: every dev fold is held out once, `metrics.val` is computed on the pooled out-of-fold predictions, `cv` holds the per-fold mean and standard deviation, and the final model is refit on all dev images before the held-out test evaluation. Out-of-fold and test predictions are written to `<run>/predictions.parquet`.

### Image kind, crop class and dataset columns

The manifest, aggregated labels and feature table carry `kind` (`original` or `crop`), `crop_class` (`original`, `large_crop`, `small_crop`), `crop_scale` (crop edge as fraction of the source short edge) and `dataset_name` (first folder under `data/raw/images`). Crops are rated systematically lower than originals, mostly because small crops show a detail without scene context. All tabular model summaries report metrics per group under `by_group`, and the Train page shows them. Compare models within a group, not only overall.

Model configs can restrict training to subsets with `include_kinds`, `include_crop_classes` or `include_datasets` (see `configs/models/embedding_clip_originals.yaml`).

Crop import defaults are scene crops: scales `0.5,0.7,0.9` of the short edge, minimum edge 768 px, and candidates below the minimum are skipped rather than enlarged (`--crop-scales`, `--min-crop-size`). The Rate and Inspect pages show the source photo with the crop rectangle for every crop.

## Image Embeddings And Zero-Shot Attributes

Pretrained vision backbones give a far stronger signal than hand-crafted features and train in seconds on CPU because the embeddings are computed once and cached:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli extract-embeddings
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli zero-shot
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli train --config configs/models/embedding_clip.yaml
```

- `configs/embeddings/default.yaml` lists the models: CLIP ViT-B/32 (`open_clip`, OpenAI weights) and DINOv2 ViT-S/14 (`timm`, 224 px). Weights are downloaded from the Hugging Face Hub on first use.
- Embeddings are cached per model under `data/processed/embeddings/<model_key>.parquet` and extended incrementally when new images are imported.
- `zero-shot` compares every CLIP image embedding with prompt ensembles for 11 cloud-type classes, 5 light situations and 16 aesthetic concepts (`clip_zero_shot.parquet`). Grouped attributes are softmax probabilities; concepts are similarity gains over a neutral cloud prompt, also z-scored across the dataset. Explanations show the CLIP cloud type as primary estimate and keep the rule-based one for reference.
- Model configs: `embedding_clip.yaml`, `embedding_dinov2.yaml`, `embedding_hybrid.yaml` (both embeddings + features + zero-shot). Regressors: `ridge` (RidgeCV), `knn` (cosine), `svr`. They use the same grouped k-fold evaluation as the baseline and appear in the Train page leaderboard.
- Explanations for embedding runs include the most similar labeled training images with their human scores (`Inspect` → `Neighbors`).

## LLM Judge (Vision Model As An Extra Rater)

A vision model can rate images on the same 0-10 scale and describe what it sees. It is stored as **its own rater** (`llm_<model>`), registered in `data/raw/metadata/raters.csv` with `rater_kind: llm`, and never mixed into the human consensus labels (`include_llm_raters: false` in the dataset config). Two backends exist, selected in the `judge` block of `configs/app/default.yaml` or with `--backend`:

- `gateway`: any OpenAI-compatible server (default: the university LLM gateway with `deepseek-v4.1-flash` or `qwen3.6-35b`). The key is read from the environment variable `CLOUD_JUDGE_GATEWAY_API_KEY` or from a secrets JSON file (`gateway_secrets_file`); it is never copied into the repo.
- `anthropic`: Claude via the Anthropic SDK (`ANTHROPIC_API_KEY` or `ant auth login`), default model `claude-opus-5-5`, structured outputs.

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli judge-images --limit 5                      # probe
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli judge-images                                # all human-labeled images
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli judge-images --model qwen3.6-35b            # second gateway judge
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli judge-report --rater llm_deepseek_v4_1_flash
```

- Images are downscaled to 1024 px before upload; the rubric is in `judge/prompt.py`, the gateway adds a JSON format hint and uses JSON mode.
- Each judgement is a structured JSON object (score, confidence, cloud types, light, colors, strengths, weaknesses, quality flags, one-sentence verdict), cached under `data/processed/judge/<rater_id>/` so re-runs are free; attributes are flattened into `<rater_id>_attributes.parquet`.
- `--only-labeled` (default) restricts the run to images with human labels: validation first, scaling later (`--no-only-labeled`).
- The report (`data/artifacts/judge_reports/`) and the `Rater Comparison` page show Spearman, MAE, bias, agreement within ±1 point, per crop class, agreement at the top of the scale, and the largest disagreements with the judge's verdict next to the photo.

### Model predictions for every image

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli predict-all --run <embedding or baseline run id>
```

Scores all active images with every model of the run (`data/processed/predictions/<run>.parquet`, copied to `latest.parquet`), including the mean and the spread between the run's models. `judge-triage` picks up `latest.parquet` automatically.

### Triage: labeling queue and exclusion proposals

Once the judges have seen the unlabeled images (`judge-images --no-only-labeled`), `judge-triage` combines all model raters into `data/processed/judge/triage.parquet`: mean judge score, spread between judges, majority cloud type and light, raised quality flags, the model prediction from `predict-all`, a labeling priority and a proposed exclusion reason derived from the flags. The priority rewards expected information: unlabeled originals first, estimates near the top end, disagreement between judges, between the run's models, and between models and judges (`queue_reason` says why). A small share of the rater's own labeled images is re-queued as blind retests (`queue` block in `configs/app/default.yaml`); the Rate page hides the earlier score for those and the Home page reports test-retest reliability and the resulting noise ceiling. Exclusion rules (`landscape_dominant` and `airplane_window` need all judges to agree; `satellite_or_map` and `not_a_cloud_photo` propose on one vote; unanimous `blurry_or_noisy` with a mean score of 1 or less proposes `bad_crop`). Nothing is excluded automatically: the `Triage` page shows the queue, the uncertain images and the proposals with checkboxes, and the Rate page offers the queue as label order `Judge-Vorschlag`.

## Capture Context: Sun, Weather, Palette

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli extract-context
```

- Reads EXIF (DateTimeOriginal, OffsetTime, GPS position and GPS UTC stamp) and falls back to file-name patterns (`PXL_…` UTC, `IMG_…` local, `IMG-…-WA…` date only). Every row records `time_quality` and `location_source`; photos without GPS use the default location from `configs/app/default.yaml` (`context`).
- Sun elevation, azimuth, hours from sunset and a light phase (night, twilight, blue hour, golden hour, low sun, day) are computed offline (NOAA approximation).
- Weather comes from the free Open-Meteo archive (cloud cover total/low/mid/high, humidity, pressure, precipitation, temperature, wind) at the nearest hour plus daily aggregates; responses are cached under `data/processed/context/weather_cache/`. Disable with `weather: false` or `--no-weather`.
- Crops inherit the context of their source original. The table lives in `data/processed/context/capture_context.parquet` and feeds the `10 vs 8` page (families "Sonnenstand / Zeit", "Wetter") and the Inspect page.
- Color palette features (k-means in CIELAB: warm/blue/neutral share, hue concentration, palette contrast, chroma, lightness, colorfulness) are part of `features_v1` since this version; run `extract-features` to refresh older tables.

## Heatmaps: Where Does The Score Come From?

For embedding runs on a single CLIP model (`embedding_clip.yaml`, `embedding_clip_originals.yaml`) every explanation now carries two model-faithful maps, computed on the fly (about 2 s per image on CPU):

- `occlusion:clip_vit_b32`: an 8x8 grid of the center square CLIP sees is masked cell by cell, the image is re-embedded and re-scored. The signed map shows the score loss relative to the typical cell: red regions support the rating, blue regions pull it down. The `Inspect` page renders it as a diverging overlay and reports the largest support and detraction in score points.
- `gradcam:clip_vit_b32.last_block`: Grad-CAM over the ViT patch tokens (7x7) for linear heads (Ridge), hooked one block before the end because CLIP pools only the CLS token after the last block.

The ResNet Grad-CAM of the frozen deep run is still available but shows ImageNet-style object attention (trees, dark masses) rather than sky structure; treat it as a diagnostic, not as evidence.

## Grad-CAM Analysis (Deep-CNN runs)

True ResNet Grad-CAM requires a Deep-CNN run:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli train --config D:\CloudAI\cloud-aesthetics\configs\models\deep_cnn_gradcam.yaml
```

The current Grad-CAM config is conservative:

- `resnet18`
- `image_size: 224`
- `batch_size: 4`
- `epochs: 5`
- `freeze_backbone: true`
- bounded 0-10 output
- automatic device selection
- AMP on CUDA
- early stopping and best/last checkpoints

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

## Interpretation Layer

Every explanation now includes a structured `interpretation` object:

- `score_band`: weak, mixed, strong, or excellent
- `confidence_band`: tight, usable, wide, or very wide
- clipped likely score range on the 0-10 scale
- readable main signals and concepts
- scientific cloud-type context when available
- cautions and a recommended action such as `shortlist`, `keep`, `review`, or `deprioritize`

Baseline feature rows are interpreted as approximate prediction deltas. Deep-CNN feature rows are interpreted as distance-from-typical signals, not causal model contributions.

Optional local LLM wording can be enabled through `configs/app/default.yaml` with the `interpretation` settings. The current adapter targets Ollama at `http://localhost:11434/api/generate` and sends only structured local explanation evidence.

## Local GPU Notes

Deep training records the selected device, Torch version, CUDA/HIP availability, AMP state, split sizes, best epoch, and checkpoint paths in `summary.json`.

`device: auto` uses CUDA when PyTorch can see it, then tries DirectML if `torch-directml` is installed, and otherwise falls back to CPU. On Windows with an AMD Radeon GPU, the default PyTorch wheel is commonly CPU-only; practical GPU paths are WSL2/ROCm or an explicit DirectML setup.

For this machine, WSL2 Ubuntu is present and exposes `/dev/dxg`, but ROCm/PyTorch are not installed in Ubuntu yet. The prepared path is documented in `docs/ROCM_WSL.md` and supported by:

- `scripts/probe_wsl_rocm.sh`
- `scripts/setup_wsl_rocm_env.sh`
- `scripts/train_wsl_rocm.ps1`
- `configs/models/deep_cnn_rocm.yaml`

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

## Android Friend Labeler APK

A native Android version is available under `android/friend-labeler` and can be built without Android Studio:

```powershell
powershell -ExecutionPolicy Bypass -File D:\CloudAI\cloud-aesthetics\scripts\build_friend_labeler_apk.ps1 -OutputApk D:\CloudAI\cloud-aesthetics\data\artifacts\cloud_friend_labeler.apk
```

The APK includes the current manifest images, scalar ratings, Duel mode, local image import with sky/cloud crop generation, XP/streak gamification, and JSON export via Android mail/share. See `docs/ANDROID_LABELER.md` for the full phone workflow.

## Web Test Dataset

The project can download a local test dataset of openly licensed cloud images from Wikimedia Commons:

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli download-web-dataset --config D:\CloudAI\cloud-aesthetics\configs\dataset\web_sample.yaml
```

Images are saved under `data/raw/images/web_sample`; attribution and license metadata are written to `data/raw/metadata/web_sample_metadata.parquet` and `.csv`.

## Current Open Work

- Add stronger real sky/cloud segmentation, possibly SAM or a dedicated semantic segmentation model.
- Validate the preferred AMD GPU route on the actual machine: WSL2/ROCm or `torch-directml`.
- Add more Deep-CNN configs once labels are stronger, for example EfficientNet-B0 or ConvNeXt-Tiny if local dependencies support them.
- Add CI/PR automation after GitHub CLI is installed.
- Decide whether generated processed data should stay local-only or move to a data artifact system.
- Collect more labels. Baseline runs are useful around 150-300 labels; Grad-CAM becomes more meaningful once deep models have enough signal, roughly 300-500+ labels.

## Data Privacy Notes

Raw private images, labels, exclusions, processed features, and model artifacts are local data. Be careful before committing or pushing files under `data/raw`, `data/processed`, or `data/artifacts`, because they may contain private or derived information.

Normal code commits should include `configs/`, `src/`, `tests/`, and documentation, not private datasets or generated feature tables. `data/processed/features/features_v1.parquet` is intentionally local-only and ignored by Git.

## Directory Highlights

- `configs/`: dataset, feature, model, and app configs
- `data/`: raw data, processed tables, cached features and embeddings, run artifacts
- `src/cloud_aesthetics/`: package code
- `notebooks/`: exploratory notebooks
- `tests/`: unit and smoke tests
