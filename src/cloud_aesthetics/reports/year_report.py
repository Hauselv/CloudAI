"""„Wolken des Jahres“: one self-contained HTML report from everything the project knows.

The report is built in two steps so it can be tested without images or models:
``collect_report_data`` gathers whatever artifacts exist (labels, judge attributes, CLIP
zero-shot, the latest concept run, the 10-vs-8 contrast, the cloud atlas, counterfactual
sweeps, rater reliability) into plain dicts and frames, and ``render_report`` turns that
into HTML with inline JPEG thumbnails and Plotly charts. Missing sources simply leave
their section out. Print the page from the browser to get a PDF.
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape

from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

REPORT_DIR = "data/artifacts/reports"


# --------------------------------------------------------------------------- collection


def image_data_uri(path_like: str | Path, max_edge: int = 720, quality: int = 82) -> str | None:
    from PIL import Image

    path = resolve_path(path_like)
    if not path.exists():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _latest_run(prefix: str, artifacts_root: str | Path = "data/artifacts") -> Path | None:
    root = resolve_path(artifacts_root)
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix) and (p / "summary.json").exists()), key=lambda p: p.name)
    return runs[-1] if runs else None


def _year_filter(manifest: pd.DataFrame, year: int | None) -> pd.Series:
    if year is None or "capture_date" not in manifest.columns:
        return pd.Series(True, index=manifest.index)
    dates = pd.to_datetime(manifest["capture_date"], errors="coerce", utc=True)
    return dates.dt.year.eq(year).fillna(False)


def collect_report_data(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ratings: pd.DataFrame | None = None,
    rater_id: str = "friend_a",
    zero_shot: pd.DataFrame | None = None,
    judge_tables: dict[str, pd.DataFrame] | None = None,
    predictions: pd.DataFrame | None = None,
    concept_run: dict[str, object] | None = None,
    analysis_table: pd.DataFrame | None = None,
    counterfactual_dir: str | Path | None = None,
    leaderboard: pd.DataFrame | None = None,
    top_n: int = 12,
    year: int | None = None,
    originals_only: bool = True,
    title: str | None = None,
) -> dict[str, object]:
    """Gather every section's data; each source is optional."""
    from cloud_aesthetics.eval.cloud_atlas import atlas_summary, atlas_table, consensus_cloud_types
    from cloud_aesthetics.eval.contrast import Band, contrast_bands

    manifest = manifest.copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    labels = labels.copy()
    labels["image_id"] = labels["image_id"].astype(str)
    scope = manifest[_year_filter(manifest, year)]
    if originals_only and "kind" in scope.columns:
        scope = scope[scope["kind"].astype(str) == "original"]
    labeled = scope.merge(labels[["image_id", "mean_score"] + [c for c in ("n_raters", "bt_score", "duels") if c in labels.columns]], on="image_id", how="inner")
    data: dict[str, object] = {
        "title": title or (f"Wolken des Jahres {year}" if year else "Wolken des Jahres"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rater_id": rater_id,
        "scope": {
            "year": year,
            "originals_only": originals_only,
            "n_images": int(len(scope)),
            "n_labeled": int(len(labeled)),
            "n_all_images": int(len(manifest)),
            "n_all_labeled": int(labels["image_id"].isin(manifest["image_id"]).sum()),
            "mean_score": float(labeled["mean_score"].mean()) if not labeled.empty else float("nan"),
            "n_top": int((labeled["mean_score"] >= 9.0).sum()) if not labeled.empty else 0,
            "n_ten": int((labeled["mean_score"] >= 9.75).sum()) if not labeled.empty else 0,
        },
    }
    hist = labeled["mean_score"].round(1).value_counts().sort_index() if not labeled.empty else pd.Series(dtype=float)
    data["score_histogram"] = [{"score": float(k), "n": int(v)} for k, v in hist.items()]

    # Per-image enrichments for the cards
    consensus = consensus_cloud_types(zero_shot, judge_tables) if (zero_shot is not None or judge_tables) else pd.DataFrame()
    atlas = atlas_table(consensus, manifest, labels, zero_shot, predictions) if not consensus.empty else pd.DataFrame()
    judge_text: dict[str, dict[str, str]] = {}
    for rater, table in (judge_tables or {}).items():
        if table is None or table.empty:
            continue
        for row in table.drop_duplicates("image_id").itertuples(index=False):
            judge_text.setdefault(str(row.image_id), {})[rater] = {
                "score": float(getattr(row, "judge_score", float("nan"))),
                "verdict": str(getattr(row, "judge_verdict", "") or ""),
                "strengths": [s.strip() for s in str(getattr(row, "judge_strengths", "") or "").split("/") if s.strip()][:3],
                "light": str(getattr(row, "judge_light", "") or ""),
            }
    contributions = concept_run.get("contributions") if concept_run else None
    concept_labels = {}
    if concept_run and isinstance(concept_run.get("concepts"), pd.DataFrame) and not concept_run["concepts"].empty:
        concept_labels = dict(zip(concept_run["concepts"]["concept"], concept_run["concepts"]["label"]))

    top = labeled.sort_values(["mean_score", "image_id"], ascending=[False, True]).head(top_n)
    cards = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        card: dict[str, object] = {"rank": rank, "image_id": image_id, "relative_path": str(row.relative_path), "score": float(row.mean_score), "kind": str(getattr(row, "kind", "")), "capture_date": str(getattr(row, "capture_date", "") or "")[:10]}
        if not atlas.empty:
            hit = atlas[atlas["image_id"] == image_id]
            if not hit.empty:
                card["cloud_type"] = str(hit.iloc[0]["type_name"])
                card["reliability"] = str(hit.iloc[0]["reliability"])
                card["light"] = str(hit.iloc[0].get("clip_light", "") or "")
                card["model_score"] = float(hit.iloc[0]["model_score"]) if pd.notna(hit.iloc[0].get("model_score")) else None
        if zero_shot is not None and not zero_shot.empty:
            z = zero_shot[zero_shot["image_id"].astype(str) == image_id]
            if not z.empty:
                concept_cols = [c for c in z.columns if c.startswith("clip_concept_z_")]
                values = z.iloc[0][concept_cols].astype(float).sort_values(ascending=False)
                card["clip_concepts"] = [(c.removeprefix("clip_concept_z_").replace("_", " "), float(v)) for c, v in values.head(3).items() if v > 0.5]
        card["judges"] = judge_text.get(image_id, {})
        if contributions is not None and not contributions.empty:
            hit = contributions[contributions["image_id"].astype(str) == image_id]
            if not hit.empty:
                r = hit.iloc[0]
                items = [(concept_labels.get(c, c), float(r[c])) for c in contributions.columns if c not in {"image_id", "intercept", "prediction"} and abs(float(r[c])) > 0.05]
                items.sort(key=lambda item: -abs(item[1]))
                card["concept_contributions"] = items[:4]
                card["concept_prediction"] = float(r["prediction"])
        cards.append(card)
    data["cards"] = cards

    # Concept model
    if concept_run and isinstance(concept_run.get("concepts"), pd.DataFrame) and not concept_run["concepts"].empty:
        from cloud_aesthetics.models.concept import sentence

        concepts = concept_run["concepts"]
        active = concepts[concepts["active"]]
        summary = concept_run.get("summary", {})
        info = summary.get("concept_model", {})
        metrics = summary.get("models", {}).get(info.get("primary", "lasso"), {}).get("metrics", {})
        data["concept_model"] = {
            "run_id": summary.get("run_id", ""),
            "sentence": sentence(concepts),
            "n_active": int(len(active)),
            "n_concepts": int(len(concepts)),
            "oof_spearman": metrics.get("val", {}).get("spearman"),
            "test_spearman": metrics.get("test", {}).get("spearman"),
            "trained_on": "/".join(str(v) for v in summary.get("config", {}).get("include_kinds", [])) or "alle Bildarten",
            "coefficients": [{"label": r.label, "coef": float(r.coef_per_sd), "low": float(getattr(r, "ci_low", np.nan)), "high": float(getattr(r, "ci_high", np.nan)), "stable": bool(getattr(r, "sign_agreement", 1.0) >= 0.8)} for r in active.itertuples()],
        }

    # 10 vs 8
    if analysis_table is not None and not analysis_table.empty and "mean_score" in analysis_table.columns:
        table = analysis_table
        if originals_only and "crop_class" in table.columns:
            table = table[table["crop_class"].astype(str) == "original"]
        result = contrast_bands(table, Band(9.5, 10.0, "9,5–10"), Band(7.5, 8.5, "7,5–8,5"))
        frame = result["table"]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            from cloud_aesthetics.explain.interpretation import FEATURE_LABELS

            top_effects = frame.head(12)
            data["contrast"] = {
                "n_a": result["n_a"],
                "n_b": result["n_b"],
                "n_signals": int(len(frame)),
                "n_significant": int((frame["q_mannwhitney"] <= 0.1).sum()),
                "effects": [{"label": FEATURE_LABELS.get(r.feature, r.feature.replace("_", " ")), "family": r.family_label, "d": float(r.cohens_d), "q": float(r.q_mannwhitney)} for r in top_effects.itertuples()],
            }

    # Atlas
    if not atlas.empty:
        atlas_scope = atlas[atlas["image_id"].isin(scope["image_id"])]
        summary = atlas_summary(atlas_scope, min_reliability="wahrscheinlich", examples=1)
        summary = summary[summary["n_labeled"] >= 1]
        if not summary.empty:
            data["atlas"] = [{"type_name": r.type_name, "n": int(r.n), "n_labeled": int(r.n_labeled), "mean_score": float(r.mean_score) if r.mean_score == r.mean_score else None, "n_top": int(r.n_top), "example_id": r.example_ids[0] if r.example_ids else None} for r in summary.sort_values("mean_score", ascending=False).itertuples()]

    # Counterfactual sweeps
    if counterfactual_dir is not None:
        root = resolve_path(counterfactual_dir)
        sweeps = []
        if root.exists():
            for notes_path in sorted(root.glob("*_notes.json")):
                notes = json.loads(notes_path.read_text(encoding="utf-8"))
                best = notes.get("best_per_edit", {})
                summary_path = notes_path.with_name(notes_path.name.replace("_notes.json", "_summary.csv"))
                worst = None
                if summary_path.exists():
                    s = pd.read_csv(summary_path)
                    if "delta" in s.columns and not s.empty:
                        w = s.loc[s["delta"].idxmin()]
                        worst = {"edit": str(w.get("edit", "")), "value": float(w.get("value", np.nan)), "delta": float(w["delta"])}
                sweeps.append({"band": notes_path.name.replace("_notes.json", "").replace("p", ",").replace("_", " "), "n_images": int(notes.get("n_images", 0)), "best": [{"edit": k, **{kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()}} for k, v in best.items()] if isinstance(best, dict) else best, "worst": worst})
        if sweeps:
            data["counterfactual"] = sweeps

    # Rater reliability and behaviour
    if ratings is not None and not ratings.empty:
        from cloud_aesthetics.eval.reliability import reliability_summary, repeat_pairs

        pairs = repeat_pairs(ratings, rater_id)
        data["reliability"] = reliability_summary(pairs)
        own = ratings[ratings["rater_id"].astype(str) == rater_id]
        data["rating_activity"] = {"n_ratings": int(len(own)), "first": str(own["rating_timestamp"].min())[:10] if not own.empty else "", "last": str(own["rating_timestamp"].max())[:10] if not own.empty else ""}

    # Models
    if leaderboard is not None and not leaderboard.empty:
        best = leaderboard.dropna(subset=["val_spearman"]).sort_values("val_spearman", ascending=False).head(5)
        data["models"] = [{"run_id": r.run_id, "model": r.model, "inputs": r.inputs, "subset": r.subset, "val_spearman": float(r.val_spearman), "val_spearman_originals": float(r.val_spearman_originals) if pd.notna(r.val_spearman_originals) else None, "test_spearman": float(r.test_spearman) if pd.notna(r.test_spearman) else None} for r in best.itertuples()]
    return data


# --------------------------------------------------------------------------- rendering

TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ d.title }}</title>
<style>
  :root { --ink:#1f2933; --muted:#6b7a86; --line:#e3e8ee; --accent:#2a6f97; --good:#2a9d8f; --bad:#e76f51; --bg:#fbfcfd; }
  * { box-sizing:border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); line-height:1.45; }
  main { max-width:1100px; margin:0 auto; padding:32px 20px 60px; }
  h1 { font-size:2.1rem; margin:0 0 4px; }
  h2 { font-size:1.35rem; margin:40px 0 12px; border-bottom:2px solid var(--line); padding-bottom:6px; }
  h3 { font-size:1.05rem; margin:0 0 6px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:16px 0; }
  .kpi { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .kpi b { display:block; font-size:1.5rem; }
  .kpi span { color:var(--muted); font-size:.85rem; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:18px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; break-inside:avoid; }
  .card img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }
  .card .body { padding:12px 14px 14px; }
  .rank { position:absolute; margin:10px; background:rgba(0,0,0,.6); color:#fff; border-radius:999px; padding:2px 10px; font-weight:600; }
  .tag { display:inline-block; font-size:.75rem; background:#eef3f7; color:var(--accent); border-radius:6px; padding:1px 7px; margin:2px 4px 2px 0; }
  .score { font-size:1.4rem; font-weight:700; }
  .meta { color:var(--muted); font-size:.85rem; }
  ul.compact { margin:6px 0 0 18px; padding:0; font-size:.9rem; }
  .contrib { font-size:.85rem; margin-top:8px; }
  .contrib .pos { color:var(--good); } .contrib .neg { color:var(--bad); }
  blockquote { margin:8px 0 0; padding:6px 10px; border-left:3px solid var(--line); color:var(--muted); font-size:.85rem; }
  table { border-collapse:collapse; width:100%; background:#fff; font-size:.9rem; }
  th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .sentence { background:#eef6fb; border-left:4px solid var(--accent); padding:10px 14px; border-radius:6px; }
  .note { color:var(--muted); font-size:.85rem; }
  .bar { height:10px; background:var(--line); border-radius:5px; position:relative; }
  .bar i { position:absolute; top:0; height:100%; border-radius:5px; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  @media (max-width:800px){ .two { grid-template-columns:1fr; } }
  @media print { body { background:#fff; } main { padding:0; } h2 { page-break-after:avoid; } .card { page-break-inside:avoid; } }
</style>
</head>
<body><main>
<h1>{{ d.title }}</h1>
<p class="sub">Bewertungen von <b>{{ d.rater_id }}</b> · erzeugt {{ d.generated_at }} · {{ "nur Originale" if d.scope.originals_only else "Originale und Ausschnitte" }}{% if d.scope.year %} · Aufnahmejahr {{ d.scope.year }}{% endif %}</p>

<div class="kpis">
  <div class="kpi"><b>{{ d.scope.n_images }}</b><span>Fotos im Bericht</span></div>
  <div class="kpi"><b>{{ d.scope.n_labeled }}</b><span>davon bewertet</span></div>
  <div class="kpi"><b>{{ "%.2f"|format(d.scope.mean_score) if d.scope.n_labeled else "–" }}</b><span>Ø Note</span></div>
  <div class="kpi"><b>{{ d.scope.n_top }}</b><span>Fotos mit 9 oder besser</span></div>
  <div class="kpi"><b>{{ d.scope.n_ten }}</b><span>Zehner</span></div>
  {% if d.rating_activity %}<div class="kpi"><b>{{ d.rating_activity.n_ratings }}</b><span>Bewertungen insgesamt ({{ d.rating_activity.first }} bis {{ d.rating_activity.last }})</span></div>{% endif %}
</div>

{% if d.score_histogram %}
<h2>Notenverteilung</h2>
{% set maxn = d.score_histogram|map(attribute='n')|max %}
<table><tbody>
{% for h in d.score_histogram %}
<tr><td style="width:60px" class="num">{{ "%.1f"|format(h.score) }}</td><td><div class="bar"><i style="left:0;width:{{ (100*h.n/maxn)|round(1) }}%;background:var(--accent)"></i></div></td><td class="num" style="width:50px">{{ h.n }}</td></tr>
{% endfor %}
</tbody></table>
{% endif %}

<h2>Die besten Fotos</h2>
<p class="note">Sortiert nach deiner Note. Jede Karte zeigt, was die Bildmodelle zu dem Foto sagen: Wolkentyp aus dem Votum von CLIP und den LLM-Judges, Beiträge des Konzeptmodells (Score-Punkte) und das Urteil der Judges. Das sind Lesehilfen, keine Begründung deiner Note.</p>
<div class="cards">
{% for c in d.cards %}
<div class="card">
  {% if c.image %}<span class="rank">#{{ c.rank }}</span><img src="{{ c.image }}" alt="{{ c.image_id }}">{% endif %}
  <div class="body">
    <div><span class="score">{{ "%.1f"|format(c.score) }}</span> <span class="meta">deine Note{% if c.model_score is not none and c.model_score is defined %} · Modell {{ "%.1f"|format(c.model_score) }}{% endif %}{% if c.concept_prediction is defined %} · Konzeptmodell {{ "%.1f"|format(c.concept_prediction) }}{% endif %}</span></div>
    <div class="meta">{{ c.image_id }}{% if c.capture_date %} · {{ c.capture_date }}{% endif %}</div>
    <div style="margin-top:6px">
      {% if c.cloud_type %}<span class="tag">{{ c.cloud_type }}{% if c.reliability %} ({{ c.reliability }}){% endif %}</span>{% endif %}
      {% if c.light %}<span class="tag">{{ c.light }}</span>{% endif %}
      {% for name, z in c.clip_concepts or [] %}<span class="tag">{{ name }} {{ "%+.1f"|format(z) }}σ</span>{% endfor %}
    </div>
    {% if c.concept_contributions %}
    <div class="contrib"><b>Konzeptmodell:</b>
      {% for label, v in c.concept_contributions %}<span class="{{ 'pos' if v > 0 else 'neg' }}">{{ "%+.2f"|format(v) }} {{ label }}</span>{% if not loop.last %} · {% endif %}{% endfor %}
    </div>
    {% endif %}
    {% for rater, j in c.judges.items() %}
    <blockquote><b>{{ rater|replace("llm_", "") }}</b> ({{ "%.1f"|format(j.score) }}): {{ j.verdict }}</blockquote>
    {% endfor %}
  </div>
</div>
{% endfor %}
</div>

{% if d.concept_model %}
<h2>Das Rating in einem Satz: Konzeptmodell</h2>
<p class="sentence">{{ d.concept_model.sentence }}</p>
<p class="note">Lasso auf benannten Konzepten, trainiert auf {{ d.concept_model.trained_on }} ({{ d.concept_model.run_id }}); {{ d.concept_model.n_active }} von {{ d.concept_model.n_concepts }} Konzepten behalten; Rangkorrelation out-of-fold {{ "%.2f"|format(d.concept_model.oof_spearman or 0) }}{% if d.concept_model.test_spearman %}, Test {{ "%.2f"|format(d.concept_model.test_spearman) }}{% endif %}. Balken: Score-Punkte je Standardabweichung; blass = Vorzeichen im Bootstrap nicht stabil.</p>
<table><tbody>
{% set maxc = d.concept_model.coefficients|map(attribute='coef')|map('abs')|max %}
{% for k in d.concept_model.coefficients %}
<tr><td style="width:40%">{{ k.label }}</td>
<td><div class="bar"><i style="{% if k.coef >= 0 %}left:50%;width:{{ (50*k.coef/maxc)|round(1) }}%;background:var(--good){% else %}right:50%;width:{{ (50*(-k.coef)/maxc)|round(1) }}%;background:var(--bad){% endif %};opacity:{{ 1 if k.stable else 0.4 }}"></i></div></td>
<td class="num" style="width:70px">{{ "%+.2f"|format(k.coef) }}</td></tr>
{% endfor %}
</tbody></table>
{% endif %}

{% if d.contrast %}
<h2>10 gegen 8: Was die Ausnahmebilder von den guten trennt</h2>
<p class="note">{{ d.contrast.n_a }} Fotos mit 9,5–10 gegen {{ d.contrast.n_b }} Fotos mit 7,5–8,5; {{ d.contrast.n_signals }} Signale verglichen, {{ d.contrast.n_significant }} davon unter q ≤ 0,1 (FDR). Cohen's d: positiv = höher bei den Ausnahmebildern.</p>
<table><thead><tr><th>Signal</th><th>Familie</th><th class="num">d</th><th class="num">q</th></tr></thead><tbody>
{% for e in d.contrast.effects %}<tr><td>{{ e.label }}</td><td>{{ e.family }}</td><td class="num">{{ "%+.2f"|format(e.d) }}</td><td class="num">{{ "%.2f"|format(e.q) }}</td></tr>{% endfor %}
</tbody></table>
{% endif %}

{% if d.atlas %}
<h2>Wolkenatlas</h2>
<p class="note">Wolkentyp als Votum aus CLIP und den LLM-Judges (mindestens „wahrscheinlich“); Ø Note über die bewerteten Fotos des Typs.</p>
<table><thead><tr><th>Typ</th><th class="num">Fotos</th><th class="num">bewertet</th><th class="num">Ø Note</th><th class="num">≥ 9</th></tr></thead><tbody>
{% for a in d.atlas %}<tr><td>{{ a.type_name }}</td><td class="num">{{ a.n }}</td><td class="num">{{ a.n_labeled }}</td><td class="num">{{ "%.1f"|format(a.mean_score) if a.mean_score is not none else "–" }}</td><td class="num">{{ a.n_top }}</td></tr>{% endfor %}
</tbody></table>
{% endif %}

{% if d.counterfactual %}
<h2>Was wäre wenn: Empfindlichkeit des Modells</h2>
<p class="note">Kontrollierte Bildveränderungen und die Reaktion des CLIP-Modells, gemittelt über ein Notenband. Aussagen über das Modell, nicht über den Geschmack.</p>
<div class="two">
{% for s in d.counterfactual %}
<div><h3>Band {{ s.band }} ({{ s.n_images }} Fotos)</h3>
<table><thead><tr><th>Eingriff</th><th class="num">bester Wert</th><th class="num">Δ Score</th></tr></thead><tbody>
{% for b in s.best %}<tr><td>{{ b.edit }}</td><td class="num">{{ "%.2f"|format(b.best_value) if b.best_value is defined else "" }}</td><td class="num">{{ "%+.2f"|format(b.best_delta) if b.best_delta is defined else "" }}</td></tr>{% endfor %}
{% if s.worst %}<tr><td><i>stärkster Verlust: {{ s.worst.edit }}</i></td><td class="num">{{ "%.2f"|format(s.worst.value) }}</td><td class="num">{{ "%+.2f"|format(s.worst.delta) }}</td></tr>{% endif %}
</tbody></table></div>
{% endfor %}
</div>
{% endif %}

{% if d.models %}
<h2>Modelle</h2>
<table><thead><tr><th>Run</th><th>Modell</th><th>Eingaben</th><th>Teilmenge</th><th class="num">ρ out-of-fold</th><th class="num">ρ Originale</th><th class="num">ρ Test</th></tr></thead><tbody>
{% for m in d.models %}<tr><td>{{ m.run_id }}</td><td>{{ m.model }}</td><td>{{ m.inputs }}</td><td>{{ m.subset }}</td><td class="num">{{ "%.2f"|format(m.val_spearman) }}</td><td class="num">{{ "%.2f"|format(m.val_spearman_originals) if m.val_spearman_originals is not none else "–" }}</td><td class="num">{{ "%.2f"|format(m.test_spearman) if m.test_spearman is not none else "–" }}</td></tr>{% endfor %}
</tbody></table>
{% endif %}

{% if d.reliability and d.reliability.n %}
<h2>Wie verlässlich sind die Noten?</h2>
<p class="note">{{ d.reliability.n }} Fotos wurden von {{ d.rater_id }} mehrfach bewertet: mittlere Abweichung {{ "%.2f"|format(d.reliability.mean_abs_difference) }} Punkte, {{ (100*d.reliability.within_half)|round(0)|int }} % innerhalb einer halben Note{% if d.reliability.noise_ceiling_spearman is defined %}, Rauschgrenze für Modelle ρ ≈ {{ "%.2f"|format(d.reliability.noise_ceiling_spearman) }}{% endif %}.</p>
{% endif %}

<p class="note" style="margin-top:40px">Erzeugt mit cloud-aesthetics. Alle Modellaussagen sind Bildschätzungen auf kleiner Datenbasis; die Noten sind die Referenz.</p>
</main></body></html>
"""


def render_report(data: dict[str, object], *, with_images: bool = True, image_edge: int = 720) -> str:
    """Render the collected data to a self-contained HTML string."""
    payload = json.loads(json.dumps(data, default=_json_default))
    if with_images:
        for card in payload.get("cards", []):
            card["image"] = image_data_uri(card["relative_path"], max_edge=image_edge)
    env = Environment(autoescape=select_autoescape(["html"]))
    return env.from_string(TEMPLATE).render(d=payload)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def save_report(html: str, *, output_dir: str | Path = REPORT_DIR, stem: str | None = None) -> Path:
    directory = resolve_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = stem or datetime.now(timezone.utc).strftime("wolken_des_jahres_%Y%m%dT%H%M%SZ")
    path = directory / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_year_report_impl(
    dataset_config: dict[str, object],
    *,
    rater_id: str | None = None,
    top_n: int = 12,
    year: int | None = None,
    originals_only: bool = True,
    title: str | None = None,
    image_edge: int = 720,
) -> tuple[dict[str, object], str]:
    """Collect from the standard project paths and render. Returns (data, html)."""
    from cloud_aesthetics.data.exclusions import active_excluded_ids
    from cloud_aesthetics.data.manifest import ensure_kind_columns
    from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
    from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
    from cloud_aesthetics.eval.contrast import load_analysis_table
    from cloud_aesthetics.eval.semantic_map import load_judge_tables
    from cloud_aesthetics.models.concept import load_concept_run
    from cloud_aesthetics.pipeline import run_leaderboard

    manifest = ensure_kind_columns(read_table(dataset_config["manifest_path"]))
    excluded = active_excluded_ids()
    if excluded:
        manifest = manifest[~manifest["image_id"].astype(str).isin(excluded)]
    labels = read_table(dataset_config["aggregated_labels_path"])
    ratings = load_raw_scalar_ratings(dataset_config["ratings_dir"])
    rater = rater_id or str(dataset_config.get("default_rater_id", "friend_a"))
    predictions = read_table("data/processed/predictions/latest.parquet")
    concept_dir = _latest_run("concept_bottleneck_originals") or _latest_run("concept_bottleneck")
    concept_run = load_concept_run(concept_dir.name) if concept_dir else None
    try:
        analysis = load_analysis_table(dataset_config)
    except Exception:  # noqa: BLE001 - the contrast section is optional
        analysis = None
    data = collect_report_data(
        manifest,
        labels,
        ratings=ratings,
        rater_id=rater,
        zero_shot=load_zero_shot(),
        judge_tables=load_judge_tables(),
        predictions=predictions,
        concept_run=concept_run,
        analysis_table=analysis,
        counterfactual_dir="data/artifacts/counterfactual",
        leaderboard=run_leaderboard(),
        top_n=top_n,
        year=year,
        originals_only=originals_only,
        title=title,
    )
    return data, render_report(data, image_edge=image_edge)
