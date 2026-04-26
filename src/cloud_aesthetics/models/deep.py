from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cloud_aesthetics.eval.metrics import compute_regression_metrics
from cloud_aesthetics.models.base import create_run_context, save_run_json
from cloud_aesthetics.models.uncertainty import conformal_interval
from cloud_aesthetics.preprocessing.datasets import ImageRegressionDataset
from cloud_aesthetics.preprocessing.transforms import build_eval_transform, build_train_transform

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import torchvision.models as tv_models
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    tv_models = None


@dataclass(slots=True)
class DeepTrainResult:
    run_id: str
    metrics: dict[str, dict[str, float]]
    model_path: str


class CloudRatingNet(nn.Module):  # type: ignore[misc]
    def __init__(self, backbone_name: str = "resnet18", freeze_backbone: bool = True) -> None:
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for deep models")
        backbone_fn = getattr(tv_models, backbone_name, None)
        if backbone_fn is None:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        try:
            backbone = backbone_fn(weights="DEFAULT")
        except Exception:  # pragma: no cover
            backbone = backbone_fn(weights=None)
        if hasattr(backbone, "fc"):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:  # pragma: no cover
            raise ValueError(f"Backbone does not expose an fc layer: {backbone_name}")
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(in_features, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.head(self.dropout(features)).squeeze(-1)


def _make_loader(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    batch_size: int,
    image_size: int,
    target_column: str,
    train: bool,
):
    transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
    dataset = ImageRegressionDataset(manifest, labels, target_column=target_column, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def _prepare_frames(manifest: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, fold_holdout: int):
    merged = manifest.merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
    train_manifest = merged[(merged["partition"] == "dev") & (merged["fold"] != fold_holdout)].drop(columns=["partition", "fold"])
    val_manifest = merged[(merged["partition"] == "dev") & (merged["fold"] == fold_holdout)].drop(columns=["partition", "fold"])
    test_manifest = merged[merged["partition"] == "test"].drop(columns=["partition", "fold"])
    return train_manifest, val_manifest, test_manifest


def _pairwise_aux_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if preds.shape[0] < 2:
        return preds.new_tensor(0.0)
    diffs = preds[:, None] - preds[None, :]
    target_diffs = targets[:, None] - targets[None, :]
    mask = torch.abs(target_diffs) >= 2.0
    if not torch.any(mask):
        return preds.new_tensor(0.0)
    pair_targets = (target_diffs[mask] > 0).float()
    pair_scores = diffs[mask]
    return torch.nn.functional.binary_cross_entropy_with_logits(pair_scores, pair_targets)


def _run_epoch(model, loader, optimizer, device, pairwise_weight: float, train: bool):
    losses = []
    preds_all = []
    targets_all = []
    model.train(mode=train)
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        pred = model(image)
        reg_loss = torch.nn.functional.mse_loss(pred, target)
        pair_loss = _pairwise_aux_loss(pred, target)
        loss = reg_loss + pairwise_weight * pair_loss
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds_all.append(pred.detach().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())
    if not preds_all:
        return {"loss": 0.0}, np.array([]), np.array([])
    preds_np = np.concatenate(preds_all)
    targets_np = np.concatenate(targets_all)
    metrics = compute_regression_metrics(targets_np, preds_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics, preds_np, targets_np


def train_deep_model(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    if torch is None:
        raise ImportError("torch and torchvision are required for deep training")
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    fold_holdout = int(config.get("fold_holdout", 0))
    train_manifest, val_manifest, test_manifest = _prepare_frames(manifest, labels, splits, fold_holdout)
    target_column = str(config.get("target_column", "mean_score"))
    train_loader = _make_loader(
        train_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        True,
    )
    val_loader = _make_loader(
        val_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        False,
    )
    test_loader = _make_loader(
        test_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CloudRatingNet(
        backbone_name=str(config.get("backbone", "resnet18")),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    best_val_rmse = float("inf")
    best_state = None
    history = []
    for epoch in range(int(config.get("epochs", 2))):
        train_metrics, _, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(config.get("pairwise_weight", 0.25)),
            train=True,
        )
        val_metrics, val_preds, val_targets = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            float(config.get("pairwise_weight", 0.25)),
            train=False,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics.get("rmse", float("inf")) < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_interval = conformal_interval(val_targets, val_preds) if len(val_preds) else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_preds, test_targets = _run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        float(config.get("pairwise_weight", 0.25)),
        train=False,
    )
    model_path = run.run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    summary = {
        "run_id": run.run_id,
        "kind": "deep",
        "model_path": str(model_path),
        "metrics": {"history": history, "test": test_metrics},
        "conformal_half_width": best_interval if "best_interval" in locals() else 0.0,
        "test_predictions": test_preds.tolist() if len(test_preds) else [],
        "test_targets": test_targets.tolist() if len(test_targets) else [],
        "backbone": str(config.get("backbone", "resnet18")),
    }
    save_run_json(summary, run.run_dir / "summary.json")
    return summary
