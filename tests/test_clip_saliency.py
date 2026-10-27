from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cloud_aesthetics.explain.clip_saliency import center_crop_box, linear_direction, normalize_signed, occlusion_support_map, place_in_image
from cloud_aesthetics.explain.heatmaps import overlay_signed_heatmap, signed_heatmap_to_rgb


class FakeVisualModel:
    """Embeds an image as its per-channel means over four quadrants (12 dims)."""

    def encode_image(self, tensor):
        _, _, h, w = tensor.shape
        quads = [tensor[:, :, : h // 2, : w // 2], tensor[:, :, : h // 2, w // 2 :], tensor[:, :, h // 2 :, : w // 2], tensor[:, :, h // 2 :, w // 2 :]]
        # A constant channel keeps the embedding direction sensitive to brightness after L2 normalisation.
        return torch.cat([q.mean(dim=(2, 3)) for q in quads] + [torch.ones(tensor.shape[0], 1)], dim=1)


class FakeBackend:
    model = FakeVisualModel()

    @staticmethod
    def preprocess(pil_image):
        array = np.asarray(pil_image.resize((64, 64))).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def test_center_crop_box_and_placement():
    box = center_crop_box(300, 500)
    assert (box.x0, box.y0, box.side) == (100, 0, 300)
    canvas = place_in_image(np.ones((2, 2), dtype=np.float32), 300, 500)
    assert canvas.shape == (300, 500) and canvas[:, :100].max() == 0 and canvas[150, 250] == 1.0


def test_occlusion_map_highlights_the_region_the_score_depends_on():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:100, 100:] = 255  # bright top-right quadrant
    backend = FakeBackend()
    # score = brightness of the top-right quadrant (dims 3-5 of the 12-dim embedding)
    weights = np.zeros(13)
    weights[3:6] = 10.0

    def predict(embeddings):
        # The map normalises embeddings before scoring; a linear scorer on them is what Ridge does.
        return embeddings @ weights

    support, score_full = occlusion_support_map(backend, predict, image, grid=2, batch_size=3)
    assert support.shape == (200, 200) and score_full > 0
    assert support[25, 175] > support[175, 25]  # hiding the bright quadrant hurts most
    assert support[25, 175] > 0
    signed = normalize_signed(support)
    assert -1.0 <= signed.min() and signed.max() == 1.0


def test_linear_direction_and_signed_overlay():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["c"]
    model = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge(alpha=0.01))]).fit(X, y)
    direction = linear_direction(model, ["a", "b", "c"])
    assert direction is not None and direction[0] > 0 and direction[2] < 0 and abs(direction[1]) < 0.2
    assert linear_direction(model, ["a", "b"]) is None
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    signed = np.array([[1.0, -1.0, 0.0, 0.5]] * 4, dtype=np.float32)
    overlay = overlay_signed_heatmap(image, signed, alpha=1.0)
    assert overlay[0, 0, 0] > overlay[0, 0, 2]  # red where positive
    assert overlay[0, 1, 2] > overlay[0, 1, 0]  # blue where negative
    assert (overlay[0, 2] == 128).all()  # untouched where zero
    assert signed_heatmap_to_rgb(signed).shape == (4, 4, 3)
