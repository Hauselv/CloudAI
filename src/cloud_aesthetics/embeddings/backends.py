"""Image/text embedding backends.

Every backend exposes ``encode_images(paths) -> np.ndarray`` with L2-normalised rows.
CLIP-style backends additionally expose ``encode_texts`` so zero-shot scoring can
compare images with text prompts in the same space. Heavy imports happen lazily so
the rest of the package stays importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from cloud_aesthetics.settings import resolve_path


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingBackend(Protocol):
    key: str
    dim: int
    supports_text: bool

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class ModelSpec:
    key: str
    backend: str
    architecture: str
    pretrained: str | None = None
    image_size: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(cls, key: str, config: dict[str, object]) -> "ModelSpec":
        known = {"backend", "architecture", "pretrained", "image_size"}
        return cls(
            key=key,
            backend=str(config["backend"]),
            architecture=str(config["architecture"]),
            pretrained=str(config["pretrained"]) if config.get("pretrained") else None,
            image_size=int(config["image_size"]) if config.get("image_size") else None,
            extra={name: value for name, value in config.items() if name not in known},
        )


def _load_pil(path: str | Path):
    from PIL import Image

    with Image.open(resolve_path(path)) as handle:
        return handle.convert("RGB")


class OpenClipBackend:
    """CLIP via ``open_clip``; supports image and text encoding."""

    supports_text = True

    def __init__(self, spec: ModelSpec, batch_size: int = 16, device: str = "cpu") -> None:
        import open_clip
        import torch

        self.key = spec.key
        self.spec = spec
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            spec.architecture, pretrained=spec.pretrained or "openai"
        )
        self.model = self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(spec.architecture)
        self.dim = int(self.model.visual.output_dim)

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch = [self.preprocess(_load_pil(path)) for path in paths[start : start + self.batch_size]]
                tensor = torch.stack(batch).to(self.device)
                chunks.append(self.model.encode_image(tensor).float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            tokens = self.tokenizer(list(texts)).to(self.device)
            return l2_normalize(self.model.encode_text(tokens).float().cpu().numpy())


class TimmBackend:
    """Image-only backbones from ``timm`` (DINOv2, ConvNeXt, ...)."""

    supports_text = False

    def __init__(self, spec: ModelSpec, batch_size: int = 16, device: str = "cpu") -> None:
        import timm
        import torch

        self.key = spec.key
        self.spec = spec
        self.batch_size = batch_size
        self.device = torch.device(device)
        kwargs: dict[str, object] = {"pretrained": True, "num_classes": 0}
        if spec.image_size:
            kwargs["img_size"] = spec.image_size
        self.model = timm.create_model(spec.architecture, **kwargs).eval().to(self.device)
        overrides: dict[str, object] = {}
        if spec.image_size:
            # The pretrained config keeps the original resolution (e.g. 518 for DINOv2); follow the requested size.
            overrides = {"input_size": (3, spec.image_size, spec.image_size), "crop_pct": 1.0}
        data_config = timm.data.resolve_data_config(overrides, model=self.model)
        self.preprocess = timm.data.create_transform(**data_config)
        self.dim = int(self.model.num_features)

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch = [self.preprocess(_load_pil(path)) for path in paths[start : start + self.batch_size]]
                tensor = torch.stack(batch).to(self.device)
                chunks.append(self.model(tensor).float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.spec.architecture} has no text encoder")


def create_backend(spec: ModelSpec, batch_size: int = 16, device: str = "cpu") -> EmbeddingBackend:
    if spec.backend == "open_clip":
        return OpenClipBackend(spec, batch_size=batch_size, device=device)
    if spec.backend == "timm":
        return TimmBackend(spec, batch_size=batch_size, device=device)
    raise ValueError(f"Unknown embedding backend: {spec.backend}")
