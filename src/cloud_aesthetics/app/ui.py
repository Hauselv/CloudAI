from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from cloud_aesthetics.settings import resolve_path


def tile_columns(page_size: int) -> int:
    if page_size <= 12:
        return 3
    if page_size >= 72:
        return 5
    return 4


def render_image_grid(
    frame: pd.DataFrame,
    *,
    caption: Callable[[pd.Series], str],
    columns: int | None = None,
    action: Callable[[pd.Series, int], None] | None = None,
    empty_text: str = "No images in this group.",
) -> None:
    if frame.empty:
        st.info(empty_text)
        return
    count = columns or tile_columns(len(frame))
    grid = st.columns(count)
    for index, (_, row) in enumerate(frame.iterrows()):
        with grid[index % count]:
            image_path = resolve_path(row["relative_path"])
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Missing image")
            st.caption(caption(row))
            if action is not None:
                action(row, index)
