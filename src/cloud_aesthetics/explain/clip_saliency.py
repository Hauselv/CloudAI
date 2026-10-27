"""Saliency for embedding models: which image regions carry the predicted score?

Two complementary views for a regressor on CLIP image embeddings:

* **Occlusion** (model-agnostic, faithful): mask each cell of a grid, re-embed, re-score.
  ``support = score_full - score_masked``: positive where hiding the region lowers the
  score (the region supports the rating), negative where hiding it raises the score.
* **Token Grad-CAM** (linear models): gradient of the score with respect to the last
  transformer block's patch tokens times the activations, giving a 7x7 map for ViT-B/32.

CLIP's preprocessing resizes the short side to 224 and center-crops, so both maps cover
the central square of the photo; the borders outside that square are unknown and set to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class CropBox:
    """Center square of the original image that CLIP sees (pixel coordinates)."""

    x0: int
    y0: int
    side: int


def center_crop_box(height: int, width: int) -> CropBox:
    side = min(height, width)
    return CropBox(x0=(width - side) // 2, y0=(height - side) // 2, side=side)


def place_in_image(cell_map: np.ndarray, height: int, width: int, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Upsample a small square map into the center crop of an (height, width) canvas; zeros elsewhere."""
    box = center_crop_box(height, width)
    canvas = np.zeros((height, width), dtype=np.float32)
    resized = cv2.resize(cell_map.astype(np.float32), (box.side, box.side), interpolation=interpolation)
    canvas[box.y0 : box.y0 + box.side, box.x0 : box.x0 + box.side] = resized
    return canvas


def _to_pil(image_rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(image_rgb)


def _normalize_rows(matrix):
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def occlusion_support_map(
    backend: Any,
    predict: Callable[[np.ndarray], np.ndarray],
    image_rgb: np.ndarray,
    *,
    grid: int = 8,
    batch_size: int = 16,
) -> tuple[np.ndarray, float]:
    """Signed support map on the full image canvas and the unmasked score.

    ``backend`` exposes ``preprocess`` (PIL -> tensor) and ``model.encode_image``;
    ``predict`` maps L2-normalised embeddings (n, d) to scores (n,).
    """
    if torch is None:
        raise ImportError("torch is required for occlusion maps")
    tensor = backend.preprocess(_to_pil(image_rgb)).unsqueeze(0)
    _, _, size_h, size_w = tensor.shape
    fill = tensor.mean(dim=(2, 3), keepdim=True)
    masked: list[Any] = []
    cells: list[tuple[int, int]] = []
    ys = np.linspace(0, size_h, grid + 1).astype(int)
    xs = np.linspace(0, size_w, grid + 1).astype(int)
    for row in range(grid):
        for col in range(grid):
            variant = tensor.clone()
            variant[:, :, ys[row] : ys[row + 1], xs[col] : xs[col + 1]] = fill
            masked.append(variant)
            cells.append((row, col))
    with torch.no_grad():
        full = _normalize_rows(backend.model.encode_image(tensor).float()).cpu().numpy()
        chunks = []
        for start in range(0, len(masked), batch_size):
            batch = torch.cat(masked[start : start + batch_size], dim=0)
            chunks.append(_normalize_rows(backend.model.encode_image(batch).float()).cpu().numpy())
    embeddings = np.vstack(chunks)
    score_full = float(np.asarray(predict(full), dtype=float)[0])
    scores = np.asarray(predict(embeddings), dtype=float)
    support = np.zeros((grid, grid), dtype=np.float32)
    for (row, col), score in zip(cells, scores, strict=True):
        support[row, col] = score_full - score
    height, width = image_rgb.shape[:2]
    return place_in_image(support, height, width), score_full


def linear_direction(model, columns: list[str]) -> np.ndarray | None:
    """Effective gradient d(score)/d(embedding) for an imputer/scaler/linear pipeline, else None."""
    final = model.named_steps["model"] if hasattr(model, "named_steps") else model
    coef = getattr(final, "coef_", None)
    if coef is None:
        return None
    weights = np.asarray(coef, dtype=np.float64).ravel()
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scale = np.asarray(model.named_steps["scaler"].scale_, dtype=np.float64)
        weights = weights / np.where(scale > 0, scale, 1.0)
    if len(weights) != len(columns):
        return None
    return weights


def token_gradcam_map(backend: Any, direction: np.ndarray, image_rgb: np.ndarray, *, block_index: int = -2) -> tuple[np.ndarray, np.ndarray]:
    """(relu Grad-CAM in [0, 1], signed grad x activation) over the ViT patch tokens, on the image canvas.

    The hook sits on the output of ``block_index`` (default: second-to-last block). After the
    last block CLIP pools only the CLS token, so the last block's patch tokens receive no
    gradient; one block earlier they still feed the CLS token through attention.
    """
    if torch is None:
        raise ImportError("torch is required for token Grad-CAM")
    visual = backend.model.visual
    blocks = visual.transformer.resblocks
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        tokens = output[0] if isinstance(output, tuple) else output
        tokens.retain_grad()
        captured["tokens"] = tokens

    index = block_index if len(blocks) >= abs(block_index) else -1
    handle = blocks[index].register_forward_hook(hook)
    try:
        tensor = backend.preprocess(_to_pil(image_rgb)).unsqueeze(0)
        backend.model.zero_grad(set_to_none=True)
        embedding = _normalize_rows(backend.model.encode_image(tensor).float())
        weight = torch.as_tensor(direction, dtype=embedding.dtype)
        score = (embedding[0] * weight).sum()
        score.backward()
        tokens = captured["tokens"]
        grads = tokens.grad
    finally:
        handle.remove()
    if tokens.dim() != 3:
        raise ValueError(f"Unexpected token tensor shape {tuple(tokens.shape)}")
    # Find the sequence axis (length = 1 + number of patches) regardless of batch_first.
    seq_axis = 1 if tokens.shape[1] != 1 or tokens.shape[0] == 1 else 0
    if tokens.shape[seq_axis] <= 1:
        seq_axis = 0 if seq_axis == 1 else 1
    acts = tokens.detach().movedim(seq_axis, 0)[:, 0, :]  # (seq, d)
    grad = grads.detach().movedim(seq_axis, 0)[:, 0, :]
    patch_acts = acts[1:]
    patch_grad = grad[1:]
    weights = patch_grad.mean(dim=0, keepdim=True)  # Grad-CAM channel weights
    cam = torch.relu((weights * patch_acts).sum(dim=1))
    signed = (patch_grad * patch_acts).sum(dim=1)
    n_patches = cam.numel()
    side = int(round(np.sqrt(n_patches)))
    grid_size = getattr(visual, "grid_size", None)
    if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2 and grid_size[0] * grid_size[1] == n_patches:
        rows, cols = int(grid_size[0]), int(grid_size[1])
    else:
        rows = cols = side
    cam_np = cam.cpu().numpy().reshape(rows, cols)
    signed_np = signed.cpu().numpy().reshape(rows, cols)
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()
    height, width = image_rgb.shape[:2]
    return place_in_image(cam_np, height, width), place_in_image(signed_np, height, width)


def normalize_signed(values: np.ndarray) -> np.ndarray:
    """Scale a signed map to [-1, 1] by its largest magnitude."""
    values = np.asarray(values, dtype=np.float32)
    peak = float(np.abs(values).max())
    return values / peak if peak > 0 else values
