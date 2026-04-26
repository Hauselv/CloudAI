from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.segmentation import slic

from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask, read_rgb_image


def build_region_table(image_path: str, heatmap: np.ndarray, n_segments: int = 50) -> pd.DataFrame:
    image = read_rgb_image(image_path)
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    rows: list[dict[str, float | int]] = []
    cloud_mask = estimate_cloud_mask(image)
    for region_id in np.unique(segments):
        mask = segments == region_id
        rows.append(
            {
                "region_id": int(region_id),
                "mean_importance": float(heatmap[mask].mean()),
                "area_fraction": float(mask.mean()),
                "cloud_fraction": float(cloud_mask[mask].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
