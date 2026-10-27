from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cloud_aesthetics.explain.counterfactual import EDITS, ClipScorer, apply_edit, apply_edits, band_sweep, best_single_edits, sensitivity_sweep, sweep_values


class FakeVisual:
    def encode_image(self, tensor):
        # mean brightness, mean saturation proxy (channel spread), constant
        mean = tensor.mean(dim=(1, 2, 3))
        spread = tensor.std(dim=1).mean(dim=(1, 2))
        return torch.stack([mean, spread, torch.ones_like(mean)], dim=1)


class FakeBackend:
    model = FakeVisual()

    @staticmethod
    def preprocess(pil_image):
        array = np.asarray(pil_image.resize((32, 32))).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class BrightnessModel:
    def predict(self, frame):
        values = frame.to_numpy()
        return 10.0 * values[:, 0] - 5.0 * values[:, 1]  # likes bright, dislikes colourful


def test_edits_are_neutral_at_neutral_and_change_the_right_thing():
    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, size=(60, 80, 3), dtype=np.uint8)
    for spec in EDITS.values():
        assert np.array_equal(apply_edit(image, spec.key, spec.neutral), image)
    gray = apply_edit(image, "saturation", 0.0)
    assert np.abs(gray.astype(int)[:, :, 0] - gray.astype(int)[:, :, 1]).max() <= 1
    brighter = apply_edit(image, "exposure", 1.0)
    assert brighter.mean() > image.mean()
    zoomed = apply_edit(image, "zoom", 2.0)
    assert zoomed.shape == image.shape
    warm = apply_edit(image, "warmth", 1.0)
    assert warm[:, :, 0].mean() > warm[:, :, 2].mean()
    assert apply_edits(image, {"contrast": 1.2, "blur": 2.0}).shape == image.shape
    assert EDITS["contrast"].neutral in sweep_values(EDITS["contrast"], 7)


def test_sweep_and_best_edit_follow_the_planted_preference():
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    image[:, :32] = [200, 80, 80]
    scorer = ClipScorer(FakeBackend(), BrightnessModel(), ["a", "b", "c"], batch_size=4)
    sweep = sensitivity_sweep(scorer, image, points=5, keys=["exposure", "saturation"])
    assert set(sweep["edit"]) == {"exposure", "saturation"}
    assert sweep[sweep["is_neutral"]]["delta"].abs().max() < 1e-6
    best = best_single_edits(sweep).set_index("edit")
    assert best.loc["exposure", "best_value"] > 0 and best.loc["exposure", "best_delta"] > 0
    assert best.loc["saturation", "best_value"] < 1.0  # less colour is better for this model
    result = band_sweep(scorer, [("x", image), ("y", image[::-1].copy())], points=5, keys=["exposure"])
    assert result.n_images == 2 and set(result.summary["edit"]) == {"exposure"}
    assert result.summary["n"].max() == 2 and len(result.notes["best_per_edit"]) == 1
