from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def build_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4, brightness_limit=0.1, contrast_limit=0.1),
            A.HueSaturationValue(p=0.3, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10),
            A.Normalize(),
        ]
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.Normalize(),
        ]
    )


def apply_transform(transform: A.Compose, image: np.ndarray) -> dict[str, Any]:
    return transform(image=image)
