"""Cloud atlas: the photos grouped by (reasonably reliable) cloud type, with ratings and meteorology.

Three sources name a cloud type per image: CLIP zero-shot (primary type) and each LLM judge
(primary plus up to two secondary types). They agree only moderately (CLIP vs a judge
~0.35-0.4, judge vs judge ~0.63), so the atlas votes: every primary vote counts 1, a
secondary mention 0.5, and the winning type gets a reliability tier from how many
primaries agree. Per type the atlas reports counts, label coverage, mean human score, the
best-rated examples and a short meteorological description.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPE_ORDER = ["cirrus", "cirrocumulus", "cirrostratus", "altocumulus", "altostratus", "nimbostratus", "stratocumulus", "stratus", "cumulus", "cumulonimbus", "lenticular", "mammatus", "contrail", "fog", "clear_sky"]


@dataclass(frozen=True)
class CloudTypeInfo:
    key: str
    name: str
    latin: str
    level: str
    height_km: str
    formation: str
    weather: str
    look: str


CLOUD_TYPES: dict[str, CloudTypeInfo] = {
    "cirrus": CloudTypeInfo("cirrus", "Cirrus (Federwolke)", "Cirrus, Ci", "hoch", "5–13 km", "Eiskristalle in großer Höhe, oft durch Windscherung zu Fasern und Haken ausgezogen; entstehen aus Feuchte, die mit dem Höhenwind vorausläuft.", "Vorbote einer herannahenden Warmfront; zunehmender, dichter werdender Cirrus kündigt oft binnen 12–24 h Wetterwechsel an. Macht das Abendlicht besonders farbig, weil Eiskristalle das rote Licht lange halten.", "feine weiße Fasern, Haken (uncinus), Flecken; Sonne scheint hindurch."),
    "cirrocumulus": CloudTypeInfo("cirrocumulus", "Cirrocumulus (Schäfchenwolke, hoch)", "Cirrocumulus, Cc", "hoch", "5–13 km", "Kleine konvektive Zellen in einer hohen, dünnen Eis-/Unterkühlte-Wasser-Schicht; meist kurzlebiges Übergangsstadium aus Cirrus oder Cirrostratus.", "„Makrelenhimmel“; kündigt häufig instabile Höhenluft und Wetterverschlechterung an.", "winzige Körnchen oder Rippel ohne Eigenschatten, regelmäßig angeordnet."),
    "cirrostratus": CloudTypeInfo("cirrostratus", "Cirrostratus (hoher Schleier)", "Cirrostratus, Cs", "hoch", "5–13 km", "Großflächige Hebung warmer Luft vor einer Warmfront bildet einen dünnen Eisschleier.", "Klassischer Frontvorbote: Halo um Sonne oder Mond, danach meist Altostratus und Regen innerhalb eines Tages.", "milchiger, fast strukturloser Schleier, Halo-Erscheinungen, Schatten bleiben sichtbar."),
    "altocumulus": CloudTypeInfo("altocumulus", "Altocumulus (Schäfchenwolke, mittel)", "Altocumulus, Ac", "mittel", "2–7 km", "Flache Konvektion oder Wellen in einer feuchten mittelhohen Schicht; Wassertröpfchen, teils unterkühlt.", "Oft bei ruhigem Wetter, morgens als Zeichen für Gewitter am Nachmittag (Ac castellanus); Wellenformen zeigen Windscherung.", "Ballen, Rollen oder Felder mit Eigenschatten, Lücken mit blauem Himmel, Irisieren möglich."),
    "altostratus": CloudTypeInfo("altostratus", "Altostratus (mittelhohe Schicht)", "Altostratus, As", "mittel", "2–7 km", "Aufgleiten warmer Luft an einer Warmfront, dichter als Cirrostratus.", "Sonne nur noch als heller Fleck „wie durch Mattglas“; Regen oder Schnee folgen meist innerhalb weniger Stunden.", "graue bis bläuliche gleichmäßige Decke, keine Halos, keine Schatten am Boden."),
    "nimbostratus": CloudTypeInfo("nimbostratus", "Nimbostratus (Regenschicht)", "Nimbostratus, Ns", "mittel bis tief", "0,5–5 km", "Verdichteter Altostratus bei anhaltender Hebung; sehr mächtig, Niederschlag fällt aus der Wolke selbst.", "Landregen oder Schneefall über Stunden; die Wolke verdeckt die Sonne vollständig.", "dunkelgrau, diffus, zerfetzte Fetzen (pannus) darunter, keine Struktur."),
    "stratocumulus": CloudTypeInfo("stratocumulus", "Stratocumulus (Haufenschicht)", "Stratocumulus, Sc", "tief", "0,5–2 km", "Konvektion unter einer Inversion: Quellwolken breiten sich seitlich aus, weil sie nicht höher steigen können; auch aus absinkendem Stratus.", "Meist trocken, allenfalls Niesel; typische Wolke ruhiger Hochdruckränder und kühler Meeresluft.", "große graue Rollen oder Ballen mit dunkler Unterseite, Lücken dazwischen."),
    "stratus": CloudTypeInfo("stratus", "Stratus (Hochnebel)", "Stratus, St", "tief", "0–0,5 km", "Abkühlung feuchter Luft am Boden oder Hebung von Nebel; gleichförmige Schicht ohne Konvektion.", "Trübes, graues Wetter; Niesel möglich. Löst sich oft tagsüber durch Sonneneinstrahlung auf.", "gleichmäßig grau, tief, strukturlos; Horizont verschwimmt."),
    "cumulus": CloudTypeInfo("cumulus", "Cumulus (Haufenwolke)", "Cumulus, Cu", "tief", "0,5–2 km (Basis)", "Thermik: Am Boden erwärmte Luft steigt auf, kondensiert am Kondensationsniveau; flache Basis, blumenkohlartige Oberseite.", "Schönwetterwolke (humilis); wachsen sie zu mediocris/congestus, drohen Schauer am Nachmittag.", "scharf begrenzte weiße Haufen mit flacher Basis, starke Hell-Dunkel-Kontraste."),
    "cumulonimbus": CloudTypeInfo("cumulonimbus", "Cumulonimbus (Gewitterwolke)", "Cumulonimbus, Cb", "tief bis hoch", "1–13 km", "Hochreichende Konvektion bei labiler Schichtung; der Turm vereist oben und breitet sich an der Tropopause zum Amboss aus.", "Schauer, Gewitter, Hagel, Böen; der Amboss zeigt die Zugrichtung des Höhenwinds an.", "massiger Turm, faseriger oder ambossförmiger Oberteil, dunkle Basis, oft Mammatus darunter."),
    "lenticular": CloudTypeInfo("lenticular", "Lenticularis (Linsenwolke)", "Altocumulus lenticularis", "mittel", "2–7 km", "Stehende Leewellen hinter Gebirgen: Luft kondensiert im Wellenberg und verdunstet im Wellental; die Wolke steht trotz starken Windes still.", "Starker Höhenwind, Föhn; Segelflieger nutzen die Wellen. Kein Niederschlag.", "glatte Linsen oder gestapelte „UFO“-Scheiben, oft weit über dem Gebirge."),
    "mammatus": CloudTypeInfo("mammatus", "Mammatus (Beutelwolke)", "Cumulonimbus mammatus", "Sonderform", "Unterseite von Cb/Amboss", "Absinkende, durch Verdunstung gekühlte Luft hängt als Beutel unter der Ambossunterseite.", "Begleiter schwerer Gewitter, meist nach dem Durchzug; spektakulär im Abendlicht.", "hängende Beutel, oft dicht gepackt, mit goldenem Rand bei tiefer Sonne."),
    "contrail": CloudTypeInfo("contrail", "Kondensstreifen", "Cirrus homogenitus", "hoch", "8–12 km", "Wasserdampf und Ruß aus Triebwerken kondensieren und gefrieren in kalter, feuchter Höhenluft.", "Lange haltende Kondensstreifen zeigen feuchte Höhenluft, oft vor Wetterwechsel; kurzlebige zeigen trockene Höhenluft.", "gerade oder vom Wind verzerrte Linien, teils zu Cirrus zerfließend."),
    "fog": CloudTypeInfo("fog", "Nebel", "Nebula", "Boden", "0 km", "Stratus am Boden: Abkühlung in klaren Nächten (Strahlungsnebel) oder warme feuchte Luft über kaltem Grund.", "Windstill und stabil; löst sich mit der Sonne oder Wind auf.", "verschwimmende Konturen, weiches Licht, Sichtweite unter 1 km."),
    "clear_sky": CloudTypeInfo("clear_sky", "Wolkenloser Himmel", "–", "–", "–", "Absinkende Luft im Hochdruckgebiet trocknet und erwärmt sich; keine Kondensation.", "Stabiles Hochdruckwetter; Dämmerungsfarben entstehen allein durch Streuung an Luft und Aerosol (Rayleigh, Mie).", "Farbverläufe ohne Wolkenstruktur, Erdschatten und Venusgürtel nach Sonnenuntergang."),
}
TYPE_NAMES = {key: info.name for key, info in CLOUD_TYPES.items()}
TYPE_NAMES["unknown"] = "unbestimmt"


def _split(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item for item in str(value).split("|") if item and item != "unknown"]


def consensus_cloud_types(zero_shot: pd.DataFrame | None, judge_tables: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """Vote per image: CLIP primary and judge primaries count 1, judge secondaries 0.5.

    Returns image_id, cloud_type, votes, n_sources, n_primary_agree, reliability
    ('sicher' = every primary source agrees, 'wahrscheinlich' = at least two primaries
    agree or the winner is a primary that another source lists, 'unsicher' = the rest).
    """
    votes: dict[str, dict[str, float]] = {}
    primaries: dict[str, list[str]] = {}
    sources: dict[str, int] = {}

    def add(image_id: str, cloud_type: str, weight: float, primary: bool) -> None:
        if not cloud_type or cloud_type == "unknown":
            return
        votes.setdefault(image_id, {})
        votes[image_id][cloud_type] = votes[image_id].get(cloud_type, 0.0) + weight
        if primary:
            primaries.setdefault(image_id, []).append(cloud_type)

    if zero_shot is not None and not zero_shot.empty and "clip_cloud_type" in zero_shot.columns:
        for row in zero_shot[["image_id", "clip_cloud_type"]].itertuples(index=False):
            image_id = str(row.image_id)
            sources[image_id] = sources.get(image_id, 0) + 1
            add(image_id, str(row.clip_cloud_type), 1.0, True)
    for table in (judge_tables or {}).values():
        if table is None or table.empty:
            continue
        for row in table.drop_duplicates("image_id").itertuples(index=False):
            image_id = str(row.image_id)
            sources[image_id] = sources.get(image_id, 0) + 1
            listed = _split(getattr(row, "judge_cloud_types", None)) or [str(getattr(row, "judge_cloud_type", "unknown"))]
            for index, cloud_type in enumerate(listed[:3]):
                add(image_id, cloud_type, 1.0 if index == 0 else 0.5, index == 0)
    rows = []
    for image_id, counts in votes.items():
        winner, score = max(counts.items(), key=lambda item: (item[1], -(TYPE_ORDER.index(item[0]) if item[0] in TYPE_ORDER else 99)))
        prim = primaries.get(image_id, [])
        agree = sum(1 for item in prim if item == winner)
        n_sources = sources.get(image_id, len(prim))
        if n_sources >= 2 and agree == n_sources:
            reliability = "sicher"
        elif agree >= 2 or (agree >= 1 and counts[winner] >= 1.5):
            reliability = "wahrscheinlich"
        else:
            reliability = "unsicher"
        rows.append({"image_id": image_id, "cloud_type": winner, "votes": float(score), "n_sources": int(n_sources), "n_primary_agree": int(agree), "reliability": reliability, "runner_up": sorted(((v, k) for k, v in counts.items() if k != winner), reverse=True)[0][1] if len(counts) > 1 else ""})
    return pd.DataFrame(rows, columns=["image_id", "cloud_type", "votes", "n_sources", "n_primary_agree", "reliability", "runner_up"])


def atlas_table(consensus: pd.DataFrame, manifest: pd.DataFrame, labels: pd.DataFrame | None = None, zero_shot: pd.DataFrame | None = None, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-image atlas rows: consensus type, reliability, paths, kind, human score, light, model score."""
    table = consensus.copy()
    table["image_id"] = table["image_id"].astype(str)
    groups = manifest.copy()
    groups["image_id"] = groups["image_id"].astype(str)
    keep = [column for column in ("relative_path", "kind", "crop_class", "dataset_name", "capture_date") if column in groups.columns]
    table = table.merge(groups[["image_id"] + keep].drop_duplicates("image_id"), on="image_id", how="inner")
    if labels is not None and not labels.empty:
        part = labels.copy()
        part["image_id"] = part["image_id"].astype(str)
        table = table.merge(part[["image_id", "mean_score"]].drop_duplicates("image_id"), on="image_id", how="left")
    else:
        table["mean_score"] = np.nan
    if zero_shot is not None and not zero_shot.empty and "clip_light" in zero_shot.columns:
        part = zero_shot[["image_id", "clip_light"]].copy()
        part["image_id"] = part["image_id"].astype(str)
        table = table.merge(part.drop_duplicates("image_id"), on="image_id", how="left")
    else:
        table["clip_light"] = np.nan
    if predictions is not None and not predictions.empty and "pred_mean" in predictions.columns:
        part = predictions[["image_id", "pred_mean"]].copy()
        part["image_id"] = part["image_id"].astype(str)
        table = table.merge(part.drop_duplicates("image_id").rename(columns={"pred_mean": "model_score"}), on="image_id", how="left")
    else:
        table["model_score"] = np.nan
    table["type_name"] = table["cloud_type"].map(TYPE_NAMES).fillna(table["cloud_type"])
    return table.reset_index(drop=True)


def atlas_summary(table: pd.DataFrame, *, min_reliability: str = "wahrscheinlich", examples: int = 3) -> pd.DataFrame:
    """One row per cloud type: counts, label coverage, mean/max human score, light mix, best examples."""
    tiers = {"sicher": 2, "wahrscheinlich": 1, "unsicher": 0}
    needed = tiers.get(min_reliability, 1)
    reliable = table[table["reliability"].map(tiers).fillna(0) >= needed]
    rows = []
    for cloud_type in TYPE_ORDER + sorted(set(table["cloud_type"]) - set(TYPE_ORDER)):
        part = reliable[reliable["cloud_type"] == cloud_type]
        if part.empty:
            continue
        labeled = part[part["mean_score"].notna()]
        originals = part[part["kind"].astype(str) == "original"] if "kind" in part.columns else part
        labeled_originals = originals[originals["mean_score"].notna()]
        light = part["clip_light"].dropna().astype(str).value_counts(normalize=True) if "clip_light" in part.columns else pd.Series(dtype=float)
        best = labeled.sort_values("mean_score", ascending=False)
        fallback = part.sort_values("model_score", ascending=False) if part["model_score"].notna().any() else part
        example_ids = best["image_id"].head(examples).tolist()
        if len(example_ids) < examples:
            example_ids += [i for i in fallback["image_id"].tolist() if i not in example_ids][: examples - len(example_ids)]
        rows.append(
            {
                "cloud_type": cloud_type,
                "type_name": TYPE_NAMES.get(cloud_type, cloud_type),
                "n": int(len(part)),
                "n_sure": int((part["reliability"] == "sicher").sum()),
                "n_originals": int(len(originals)),
                "n_labeled": int(len(labeled)),
                "mean_score": float(labeled["mean_score"].mean()) if not labeled.empty else np.nan,
                "sd_score": float(labeled["mean_score"].std(ddof=0)) if len(labeled) > 1 else np.nan,
                "max_score": float(labeled["mean_score"].max()) if not labeled.empty else np.nan,
                "mean_score_originals": float(labeled_originals["mean_score"].mean()) if not labeled_originals.empty else np.nan,
                "n_top": int((labeled["mean_score"] >= 9.0).sum()),
                "share_top": float((labeled["mean_score"] >= 9.0).mean()) if not labeled.empty else np.nan,
                "model_mean": float(part["model_score"].mean()) if part["model_score"].notna().any() else np.nan,
                "top_light": str(light.index[0]) if not light.empty else "",
                "share_golden_or_twilight": float(light.get("golden_hour", 0.0) + light.get("twilight", 0.0)) if not light.empty else np.nan,
                "example_ids": example_ids,
            }
        )
    return pd.DataFrame(rows)


def type_by_light(table: pd.DataFrame, *, min_n: int = 3) -> pd.DataFrame:
    """Mean human score per cloud type and light phase (cells with fewer than min_n labels are blank)."""
    labeled = table[table["mean_score"].notna() & table["clip_light"].notna()]
    if labeled.empty:
        return pd.DataFrame()
    grid = labeled.pivot_table(index="type_name", columns="clip_light", values="mean_score", aggfunc=["mean", "count"])
    means = grid["mean"].where(grid["count"] >= min_n)
    order = [TYPE_NAMES[t] for t in TYPE_ORDER if TYPE_NAMES[t] in means.index]
    light_order = [l for l in ("daylight", "golden_hour", "twilight", "blue_hour", "night") if l in means.columns]
    return means.reindex(order)[light_order]


def reliability_report(consensus: pd.DataFrame) -> dict[str, object]:
    if consensus.empty:
        return {"n": 0}
    counts = consensus["reliability"].value_counts()
    return {"n": int(len(consensus)), "sicher": int(counts.get("sicher", 0)), "wahrscheinlich": int(counts.get("wahrscheinlich", 0)), "unsicher": int(counts.get("unsicher", 0)), "mean_sources": float(consensus["n_sources"].mean())}
