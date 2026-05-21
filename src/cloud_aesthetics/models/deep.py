from __future__ import annotations

import random
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


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class CloudRatingNet(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        backbone_name: str = "resnet18",
        freeze_backbone: bool = True,
        *,
        bounded_output: bool = False,
        score_min: float = 0.0,
        score_max: float = 10.0,
    ) -> None:
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
        self.bounded_output = bounded_output
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        raw = self.head(self.dropout(features)).squeeze(-1)
        if not self.bounded_output:
            return raw
        return self.score_min + (self.score_max - self.score_min) * torch.sigmoid(raw)


def resolve_training_device(preferred: str = "auto"):
    if torch is None:
        raise ImportError("torch is required for device resolution")
    requested = str(preferred or "auto").lower()
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        index = torch.cuda.current_device()
        hip_version = getattr(torch.version, "hip", None)
        backend = "rocm" if hip_version else "cuda"
        return torch.device("cuda"), {
            "type": backend,
            "name": torch.cuda.get_device_name(index),
            "cuda_version": getattr(torch.version, "cuda", None),
            "hip_version": hip_version,
        }
    if requested in {"auto", "directml", "dml"}:
        try:
            import torch_directml  # type: ignore[import-not-found]

            device = torch_directml.device()
            return device, {"type": "directml", "name": "DirectML", "cuda_version": None}
        except ImportError:
            if requested in {"directml", "dml"}:
                raise ImportError("torch-directml is required for DirectML training") from None
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), {"type": "mps", "name": "Apple MPS", "cuda_version": None}
    if requested not in {"auto", "cpu"}:
        raise ValueError(f"Requested training device is not available: {preferred}")
    return torch.device("cpu"), {"type": "cpu", "name": "CPU", "cuda_version": None}


def _make_loader(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    batch_size: int,
    image_size: int,
    target_column: str,
    train: bool,
    num_workers: int,
    pin_memory: bool,
):
    transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
    dataset = ImageRegressionDataset(manifest, labels, target_column=target_column, transform=transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": max(0, num_workers),
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _prepare_frames(manifest: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, fold_holdout: int):
    merged = manifest.merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
    train_manifest = merged[(merged["partition"] == "dev") & (merged["fold"] != fold_holdout)].drop(columns=["partition", "fold"])
    val_manifest = merged[(merged["partition"] == "dev") & (merged["fold"] == fold_holdout)].drop(columns=["partition", "fold"])
    test_manifest = merged[merged["partition"] == "test"].drop(columns=["partition", "fold"])
    return train_manifest, val_manifest, test_manifest


def _checkpoint_payload(model, optimizer, epoch: int, metrics: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
        "metrics": metrics,
        "config": config,
    }


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


def _run_epoch(model, loader, optimizer, device, pairwise_weight: float, train: bool, scaler=None, use_amp: bool = False):
    losses = []
    preds_all = []
    targets_all = []
    model.train(mode=train)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(image)
                    reg_loss = torch.nn.functional.mse_loss(pred, target)
                    pair_loss = _pairwise_aux_loss(pred, target)
                    loss = reg_loss + pairwise_weight * pair_loss
            else:
                pred = model(image)
                reg_loss = torch.nn.functional.mse_loss(pred, target)
                pair_loss = _pairwise_aux_loss(pred, target)
                loss = reg_loss + pairwise_weight * pair_loss
        if train:
            if optimizer is None:
                raise ValueError("Optimizer is required for a training epoch")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
    seed = int(config.get("seed", 42))
    set_reproducible_seed(seed)
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    fold_holdout = int(config.get("fold_holdout", 0))
    train_manifest, val_manifest, test_manifest = _prepare_frames(manifest, labels, splits, fold_holdout)
    if train_manifest.empty:
        raise ValueError("Deep training split is empty. Add labels or regenerate grouped splits before training.")
    target_column = str(config.get("target_column", "mean_score"))
    device, device_info = resolve_training_device(str(config.get("device", "auto")))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = device_info["type"] in {"cuda", "rocm"}
    train_loader = _make_loader(
        train_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        True,
        num_workers,
        pin_memory,
    )
    val_loader = _make_loader(
        val_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        False,
        num_workers,
        pin_memory,
    )
    test_loader = _make_loader(
        test_manifest,
        labels,
        int(config.get("batch_size", 8)),
        int(config.get("image_size", 224)),
        target_column,
        False,
        num_workers,
        pin_memory,
    )
    model = CloudRatingNet(
        backbone_name=str(config.get("backbone", "resnet18")),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        bounded_output=bool(config.get("bounded_output", False)),
        score_min=float(config.get("score_min", 0.0)),
        score_max=float(config.get("score_max", 10.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    use_amp = bool(config.get("amp", True)) and device_info["type"] == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device_info["type"] == "cuda" else None
    best_val_rmse = float("inf")
    best_state = None
    best_epoch = -1
    best_interval = 0.0
    early_stopped = False
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 0) or 0)
    min_delta = float(config.get("early_stopping_min_delta", 0.0))
    history = []
    checkpoint_best_path = run.run_dir / "checkpoint_best.pt"
    checkpoint_last_path = run.run_dir / "checkpoint_last.pt"
    for epoch in range(int(config.get("epochs", 2))):
        train_metrics, _, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(config.get("pairwise_weight", 0.25)),
            train=True,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics, val_preds, val_targets = _run_epoch(
            model,
            val_loader,
            None,
            device,
            float(config.get("pairwise_weight", 0.25)),
            train=False,
            scaler=None,
            use_amp=False,
        )
        monitor_rmse = val_metrics.get("rmse", train_metrics.get("rmse", float("inf")))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "monitor_rmse": monitor_rmse})
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics, "monitor_rmse": monitor_rmse},
                config,
            ),
            checkpoint_last_path,
        )
        if monitor_rmse + min_delta < best_val_rmse:
            best_val_rmse = monitor_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_interval = conformal_interval(val_targets, val_preds) if len(val_preds) else 0.0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    {"train": train_metrics, "val": val_metrics, "monitor_rmse": monitor_rmse},
                    config,
                ),
                checkpoint_best_path,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            early_stopped = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_preds, test_targets = _run_epoch(
        model,
        test_loader,
        None,
        device,
        float(config.get("pairwise_weight", 0.25)),
        train=False,
        scaler=None,
        use_amp=False,
    )
    model_path = run.run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    best_model_path = run.run_dir / "model_best.pt"
    if best_state is not None and bool(config.get("save_best_checkpoint", True)):
        torch.save(best_state, best_model_path)
    summary = {
        "run_id": run.run_id,
        "kind": "deep",
        "model_path": str(model_path),
        "metrics": {"history": history, "test": test_metrics},
        "conformal_half_width": best_interval,
        "test_predictions": test_preds.tolist() if len(test_preds) else [],
        "test_targets": test_targets.tolist() if len(test_targets) else [],
        "backbone": str(config.get("backbone", "resnet18")),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse if np.isfinite(best_val_rmse) else None,
        "best_model_path": str(best_model_path) if best_model_path.exists() else str(model_path),
        "best_checkpoint_path": str(checkpoint_best_path) if checkpoint_best_path.exists() else "",
        "last_checkpoint_path": str(checkpoint_last_path) if checkpoint_last_path.exists() else "",
        "early_stopped": early_stopped,
        "epochs_ran": len(history),
        "device": device_info,
        "amp": use_amp,
        "bounded_output": bool(config.get("bounded_output", False)),
        "dataset_sizes": {
            "train": int(len(train_manifest)),
            "val": int(len(val_manifest)),
            "test": int(len(test_manifest)),
        },
        "software": {
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "hip_version": getattr(torch.version, "hip", None),
        },
    }
    save_run_json(summary, run.run_dir / "summary.json")
    return summary
