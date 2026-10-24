from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.embeddings.backends import ModelSpec, l2_normalize
from cloud_aesthetics.embeddings.store import embedding_matrix, extract_embeddings, load_embedding_table, load_embeddings
from cloud_aesthetics.embeddings.zero_shot import ZeroShotTaxonomy, build_text_bank, zero_shot_table
from cloud_aesthetics.models.embedding import build_embedding_input_table, train_embedding_suite


class FakeBackend:
    """Deterministic stand-in: hashes the file name into a vector, texts into word vectors."""

    supports_text = True
    key = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=self.dim).astype(np.float32)

    def encode_images(self, paths):
        self.calls.append(len(paths))
        return l2_normalize(np.stack([self._vector(Path(path).name) for path in paths]))

    def encode_texts(self, texts):
        return l2_normalize(np.stack([self._vector(text.split()[0]) for text in texts]))


def _manifest(tmp_path: Path, n: int = 6) -> pd.DataFrame:
    rows = []
    for index in range(n):
        path = tmp_path / f"img_{index}.jpg"
        path.write_bytes(b"x")
        rows.append({"image_id": f"id{index}", "relative_path": str(path), "kind": "original", "dataset_name": "toy"})
    return pd.DataFrame(rows)


def test_extract_embeddings_is_incremental(tmp_path: Path):
    manifest = _manifest(tmp_path)
    spec = ModelSpec(key="fake", backend="fake", architecture="fake")
    backend = FakeBackend()
    first = extract_embeddings(manifest.head(4), spec, output_dir=tmp_path / "emb", backend=backend)
    assert len(first) == 4
    second = extract_embeddings(manifest, spec, output_dir=tmp_path / "emb", backend=backend)
    assert len(second) == 6
    assert backend.calls == [4, 2]
    third = extract_embeddings(manifest, spec, output_dir=tmp_path / "emb", backend=backend)
    assert len(third) == 6 and backend.calls == [4, 2]
    cached = load_embeddings("fake", tmp_path / "emb")
    assert embedding_matrix(cached).shape == (6, 8)
    assert (tmp_path / "emb" / "fake.json").exists()
    wide = load_embedding_table(["fake"], tmp_path / "emb")
    assert wide.columns[1].startswith("emb_fake_")


def test_zero_shot_table_shapes_and_probabilities():
    backend = FakeBackend()
    taxonomy = ZeroShotTaxonomy(
        cloud_types={"cumulus": ["cumulus clouds"], "cirrus": ["cirrus clouds"]},
        light={"daylight": ["daylight sky"], "night": ["night sky"]},
        concepts={"dramatic": ["dramatic storm"], "soft": ["soft haze"]},
    )
    bank = build_text_bank(backend, taxonomy)
    embeddings = backend.encode_images([f"a{i}.jpg" for i in range(5)])
    table = zero_shot_table([f"id{i}" for i in range(5)], embeddings, bank)
    assert len(table) == 5
    type_cols = [column for column in table.columns if column.startswith("clip_type_")]
    assert np.allclose(table[type_cols].sum(axis=1), 1.0, atol=1e-5)
    assert set(table["clip_cloud_type"]).issubset({"cumulus", "cirrus"})
    assert set(table["clip_light"]).issubset({"daylight", "night"})
    assert abs(float(table["clip_concept_z_dramatic"].mean())) < 1e-5
    assert (table["clip_cloud_type_confidence"] >= 0).all()


def test_embedding_suite_trains_on_cached_embeddings(tmp_path: Path):
    manifest = _manifest(tmp_path, n=24)
    spec = ModelSpec(key="fake", backend="fake", architecture="fake")
    extract_embeddings(manifest, spec, output_dir=tmp_path / "emb", backend=FakeBackend())
    config = {
        "output_dir": str(tmp_path / "runs"),
        "run_name": "embed_smoke",
        "target_column": "mean_score",
        "embedding_models": ["fake"],
        "embeddings_dir": str(tmp_path / "emb"),
        "regressors": ["ridge", "knn"],
        "knn_neighbors": 3,
    }
    table = build_embedding_input_table(config)
    assert table.shape == (24, 9)
    labels = pd.DataFrame({"image_id": manifest["image_id"], "mean_score": np.linspace(2, 9, 24), "kind": "original", "dataset_name": "toy"})
    splits = pd.DataFrame({"image_id": manifest["image_id"], "partition": ["dev"] * 18 + ["test"] * 6, "fold": [i % 3 for i in range(18)] + [-1] * 6})
    summary = train_embedding_suite(table, labels, splits, config)
    assert summary["kind"] == "embedding"
    assert summary["embedding_models"] == ["fake"]
    assert set(summary["models"]) == {"ridge", "knn"}
    assert summary["evaluation"]["n_features"] == 8
    assert "alpha" in summary["models"]["ridge"]["fitted_params"]
    assert len(summary["input_columns"]) == 8
