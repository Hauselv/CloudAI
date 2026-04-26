from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.preprocessing.transforms import apply_transform

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass


@dataclass
class DatasetItem:
    image: np.ndarray
    target: float
    image_id: str


class ImageRegressionDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        labels: pd.DataFrame,
        target_column: str,
        transform=None,
    ) -> None:
        merged = manifest.merge(labels[["image_id", target_column]], on="image_id", how="inner")
        self.frame = merged.reset_index(drop=True)
        self.target_column = target_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = read_rgb_image(row["relative_path"])
        if self.transform is not None:
            transformed = apply_transform(self.transform, image)
            image = transformed["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        target = np.float32(row[self.target_column])
        if torch is not None:
            return {
                "image": torch.tensor(image, dtype=torch.float32),
                "target": torch.tensor(target, dtype=torch.float32),
                "image_id": row["image_id"],
            }
        return {"image": image, "target": target, "image_id": row["image_id"]}
