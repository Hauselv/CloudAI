"""CLIP zero-shot scores for cloud types, lighting and aesthetic concepts.

Images are compared with text prompts in CLIP space. Grouped attributes (cloud type,
light) are turned into probabilities with a softmax over prompt-ensemble similarities.
Open-ended concepts are scored as the similarity gain over a neutral cloud prompt and
additionally z-scored across the dataset so values are comparable between concepts.
All of this is an image-based estimate, not a meteorological observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cloud_aesthetics.embeddings.backends import EmbeddingBackend, l2_normalize

PROMPT_TEMPLATES = [
    "a photo of {}.",
    "a photograph of the sky showing {}.",
    "{}, seen from the ground.",
]

NEUTRAL_PROMPTS = ["a photo of clouds in the sky.", "a photograph of the sky.", "clouds."]

CLOUD_TYPE_PROMPTS: dict[str, list[str]] = {
    "cirrus": ["cirrus clouds, thin wispy high clouds", "feathery white streaks of cirrus clouds"],
    "cirrocumulus": ["cirrocumulus clouds, small high rippled cloudlets", "a mackerel sky of tiny high cloudlets"],
    "cirrostratus": ["cirrostratus clouds, a thin milky high veil over the sky", "a halo around the sun through a thin high cloud veil"],
    "altocumulus": ["altocumulus clouds, mid-level patches of rounded cloudlets", "a sky covered with rows of mid-level cloud patches"],
    "altostratus": ["altostratus clouds, a grey featureless mid-level sheet", "the sun dimly visible through a grey cloud sheet"],
    "nimbostratus": ["nimbostratus clouds, a dark low rain cloud layer", "a dark overcast rainy sky"],
    "stratocumulus": ["stratocumulus clouds, low lumpy grey and white cloud rolls", "a low layer of lumpy clouds with gaps of blue sky"],
    "stratus": ["stratus clouds, a uniform low grey cloud layer", "low fog-like grey cloud covering the sky"],
    "cumulus": ["cumulus clouds, puffy white fair-weather clouds", "cotton-like white clouds with flat bases on a blue sky"],
    "cumulonimbus": ["a cumulonimbus thunderstorm cloud, towering with an anvil top", "a huge towering storm cloud"],
    "clear_sky": ["a clear blue sky without clouds", "an empty cloudless sky"],
}

LIGHT_PROMPTS: dict[str, list[str]] = {
    "daylight": ["clouds in bright midday daylight", "a blue daytime sky with clouds"],
    "golden_hour": ["clouds lit by warm golden hour sunlight", "clouds glowing orange and gold at sunset"],
    "twilight": ["clouds in pink and purple twilight after sunset", "an afterglow sky with pastel clouds"],
    "blue_hour": ["clouds in deep blue dusk light", "a dark blue evening sky with clouds"],
    "night": ["clouds at night", "a dark night sky with clouds"],
}

CONCEPT_PROMPTS: dict[str, list[str]] = {
    "dramatic_storm_sky": ["a dramatic dark storm sky", "ominous threatening storm clouds"],
    "golden_rim_light": ["clouds with glowing edges backlit by the sun", "rim light on cloud edges"],
    "crepuscular_rays": ["sunbeams breaking through clouds", "crepuscular rays of light through clouds"],
    "pastel_sunset": ["a soft pastel pink and orange sunset sky", "gentle pastel colored clouds"],
    "layered_depth": ["many layers of clouds at different heights giving depth", "a sky with layered clouds and depth"],
    "towering_structure": ["a tall towering cloud with strong vertical structure", "a massive cloud tower"],
    "fine_texture": ["finely textured rippled clouds", "clouds with intricate delicate texture"],
    "mammatus": ["mammatus clouds, pouch-like bulges hanging from a cloud base"],
    "lenticular": ["a lenticular lens-shaped cloud", "smooth UFO-shaped lenticular clouds"],
    "rainbow": ["a rainbow in front of clouds"],
    "high_contrast": ["high contrast clouds with bright highlights and deep shadows"],
    "soft_haze": ["a hazy soft washed-out sky", "flat low-contrast milky sky"],
    "landscape_dominant": ["a landscape photo where land and buildings take up most of the frame"],
    "airplane_window": ["clouds seen from an airplane window", "a view from above the clouds out of a plane"],
    "boring_sky": ["a dull boring plain grey sky", "an unremarkable overcast sky"],
    "spectacular_sky": ["a spectacular breathtaking sky", "an awe-inspiring beautiful cloudscape"],
}


@dataclass
class ZeroShotTaxonomy:
    cloud_types: dict[str, list[str]] = field(default_factory=lambda: dict(CLOUD_TYPE_PROMPTS))
    light: dict[str, list[str]] = field(default_factory=lambda: dict(LIGHT_PROMPTS))
    concepts: dict[str, list[str]] = field(default_factory=lambda: dict(CONCEPT_PROMPTS))
    templates: list[str] = field(default_factory=lambda: list(PROMPT_TEMPLATES))
    neutral: list[str] = field(default_factory=lambda: list(NEUTRAL_PROMPTS))


def _class_embedding(backend: EmbeddingBackend, phrases: list[str], templates: list[str]) -> np.ndarray:
    prompts = [template.format(phrase) for phrase in phrases for template in templates]
    vectors = backend.encode_texts(prompts)
    return l2_normalize(vectors.mean(axis=0, keepdims=True))[0]


def build_text_bank(backend: EmbeddingBackend, taxonomy: ZeroShotTaxonomy | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Encode every class of the taxonomy into one prompt-ensemble vector."""
    taxonomy = taxonomy or ZeroShotTaxonomy()
    bank: dict[str, dict[str, np.ndarray]] = {
        "cloud_type": {name: _class_embedding(backend, phrases, taxonomy.templates) for name, phrases in taxonomy.cloud_types.items()},
        "light": {name: _class_embedding(backend, phrases, taxonomy.templates) for name, phrases in taxonomy.light.items()},
        "concept": {name: _class_embedding(backend, phrases, taxonomy.templates) for name, phrases in taxonomy.concepts.items()},
        "neutral": {"neutral": l2_normalize(backend.encode_texts(taxonomy.neutral).mean(axis=0, keepdims=True))[0]},
    }
    return bank


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def score_group(image_embeddings: np.ndarray, group: dict[str, np.ndarray], logit_scale: float = 100.0) -> pd.DataFrame:
    names = list(group)
    text_matrix = np.stack([group[name] for name in names])
    sims = l2_normalize(image_embeddings) @ text_matrix.T
    probs = _softmax(sims * logit_scale)
    return pd.DataFrame(probs, columns=names)


def zero_shot_table(
    image_ids: list[str],
    image_embeddings: np.ndarray,
    bank: dict[str, dict[str, np.ndarray]],
    *,
    logit_scale: float = 100.0,
    prefix: str = "clip",
) -> pd.DataFrame:
    """Per-image zero-shot attributes from cached CLIP image embeddings."""
    embeddings = l2_normalize(image_embeddings)
    out = pd.DataFrame({"image_id": [str(value) for value in image_ids]})

    type_probs = score_group(embeddings, bank["cloud_type"], logit_scale)
    ranked = np.sort(type_probs.to_numpy(), axis=1)
    out[f"{prefix}_cloud_type"] = type_probs.idxmax(axis=1).to_numpy()
    out[f"{prefix}_cloud_type_probability"] = ranked[:, -1]
    # Margin between the best and second-best class: 0 means a toss-up, 1 means unambiguous.
    out[f"{prefix}_cloud_type_confidence"] = ranked[:, -1] - (ranked[:, -2] if ranked.shape[1] > 1 else 0.0)
    for name in type_probs.columns:
        out[f"{prefix}_type_{name}"] = type_probs[name].to_numpy(dtype=np.float32)

    light_probs = score_group(embeddings, bank["light"], logit_scale)
    out[f"{prefix}_light"] = light_probs.idxmax(axis=1).to_numpy()
    for name in light_probs.columns:
        out[f"{prefix}_light_{name}"] = light_probs[name].to_numpy(dtype=np.float32)

    neutral = bank["neutral"]["neutral"]
    neutral_sim = embeddings @ neutral
    for name, vector in bank["concept"].items():
        delta = embeddings @ vector - neutral_sim
        out[f"{prefix}_concept_{name}"] = delta.astype(np.float32)
        std = float(delta.std()) or 1.0
        out[f"{prefix}_concept_z_{name}"] = ((delta - delta.mean()) / std).astype(np.float32)
    return out


def concept_columns(frame: pd.DataFrame, prefix: str = "clip") -> list[str]:
    return [column for column in frame.columns if column.startswith(f"{prefix}_concept_") and not column.startswith(f"{prefix}_concept_z_")]


def type_columns(frame: pd.DataFrame, prefix: str = "clip") -> list[str]:
    return [column for column in frame.columns if column.startswith(f"{prefix}_type_")]
