"""Per-friend label order.

Friends first see the images the core rater has already labeled (so agreement can be
measured), then the images with the most information value (map gaps, triage queue),
then everything else. Within each block the order is shuffled with a per-friend seed, so
no two friends share the same sequence and order effects do not line up.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FriendQueueConfig:
    crop_classes: tuple[str, ...] = ("original", "large_crop")
    core_rater: str = "friend_a"
    core_originals_first: bool = True
    duel_threshold: float = 7.5
    extra_sources: tuple[str, ...] = field(default=("map_queue", "triage"))


def _seed(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def build_friend_queue(
    manifest: pd.DataFrame,
    *,
    token: str,
    core_rated_ids: set[str],
    excluded_ids: set[str] | None = None,
    map_queue: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    config: FriendQueueConfig = FriendQueueConfig(),
) -> list[str]:
    frame = manifest.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    if "crop_class" in frame.columns:
        frame = frame[frame["crop_class"].astype(str).isin(config.crop_classes)]
    if excluded_ids:
        frame = frame[~frame["image_id"].isin(excluded_ids)]
    frame = frame.drop_duplicates("image_id")
    rng = np.random.default_rng(_seed(token))
    ids = frame["image_id"].tolist()
    kind = dict(zip(frame["image_id"], frame["crop_class"].astype(str))) if "crop_class" in frame.columns else {}

    def shuffled(items: list[str]) -> list[str]:
        items = list(items)
        rng.shuffle(items)
        return items

    core = [i for i in ids if i in core_rated_ids]
    if config.core_originals_first and kind:
        core = shuffled([i for i in core if kind.get(i) == "original"]) + shuffled([i for i in core if kind.get(i) != "original"])
    else:
        core = shuffled(core)
    remaining = [i for i in ids if i not in core_rated_ids]
    ranked: list[str] = []
    if map_queue is not None and not map_queue.empty and "map_rank" in map_queue.columns:
        ordered = map_queue.sort_values("map_rank")["image_id"].astype(str).tolist()
        ranked += [i for i in ordered if i in set(remaining) and i not in ranked]
    if triage is not None and not triage.empty and "label_rank" in triage.columns:
        ordered = triage.sort_values("label_rank")["image_id"].astype(str).tolist()
        ranked += [i for i in ordered if i in set(remaining) and i not in ranked]
    rest = shuffled([i for i in remaining if i not in set(ranked)])
    return core + ranked + rest


def next_items(queue: list[str], rated_ids: set[str], *, exclude: set[str] | None = None, limit: int = 6) -> list[str]:
    exclude = exclude or set()
    out = []
    for image_id in queue:
        if image_id in rated_ids or image_id in exclude:
            continue
        out.append(image_id)
        if len(out) >= limit:
            break
    return out


def progress(queue: list[str], rated_ids: set[str], core_rated_ids: set[str]) -> dict[str, int]:
    queue_set = set(queue)
    done = [i for i in queue if i in rated_ids]
    core = [i for i in queue if i in core_rated_ids]
    return {
        "n_total": len(queue),
        "n_done": len(done),
        "n_open": len(queue_set - rated_ids),
        "n_core": len(core),
        "n_core_done": len([i for i in core if i in rated_ids]),
    }
