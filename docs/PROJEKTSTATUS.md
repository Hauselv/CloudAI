# CloudAI / cloud-aesthetics – Architektur und Status

Stand: 24.10.2026 (Bestandsaufnahme nach Projektübernahme). Letzter Commit: 22.05.2026.

## 1. Worum es geht

Fotos von Wolken und Sonnenuntergängen werden von Michel (und Freunden) auf einer
0–10-Skala in 0,5er-Schritten bewertet. Auf diesen Labels werden Modelle trainiert, die
das Rating ungesehener Bilder vorhersagen. Das eigentliche Forschungsziel ist aber nicht
die Vorhersage, sondern die **Erklärung**: Von welchen Bildeigenschaften hängt das Rating
ab – Kontrast, Wolkentyp, Farbe, Komposition, Kombinationen davon? Und speziell: Was
unterscheidet eine 10/10 von einer „nur“ guten 8/10?

## 2. Technischer Überblick

| Bereich | Umsetzung |
|---|---|
| Sprache / Umgebung | Python 3.12, venv unter `D:\CloudAI\.venv`, Windows 11 |
| Paket | `src/cloud_aesthetics` (ca. 6.200 Zeilen), installierbar via `pip install -e .[dev,deep,explain]` |
| CLI | `python -m cloud_aesthetics.cli` (Typer): import-images, ingest-images, aggregate-labels, extract-features, train, evaluate, explain, analyze-batch, export-friend-package, import-friend-labels, download-web-dataset |
| UI | Streamlit-Multipage-App (`src/cloud_aesthetics/app/Home.py` + 9 Seiten) |
| ML-Stack | scikit-learn 1.8, torch 2.11 (**CPU-only**), torchvision, timm, captum, grad-cam, shap |
| GPU | AMD Radeon RX 6700 XT; kein CUDA. ROCm-unter-WSL ist vorbereitet (`docs/ROCM_WSL.md`), aber nicht installiert und für diese GPU offiziell nicht unterstützt |
| Tests | 22 pytest-Tests, alle grün (ca. 16 s) |
| Friend-Labeling | Standalone-HTML-Paket (`standalone.py`) und native Android-APK (`android/friend-labeler`, Java, ohne Android Studio baubar) |

### Datenfluss

```
G:\Wolken (90 Originale)  --import-images-->  data/raw/images/private_wolken/{originals,crops}
Openverse/Wikimedia       --download-web-dataset-->  data/raw/images/web_sample/
                                       |
                                 ingest-images  -->  data/processed/image_manifest.parquet
                                       |             (image_id = sha256[:16], phash, split_group_id)
Streamlit "Rate" / Friend-Apps -->  data/raw/metadata/ratings/<rater>.csv   (Rohlabels, append-only)
                                       |
                                 aggregate-labels -->  processed/ratings.parquet, aggregated_labels.parquet,
                                       |               pairwise_preferences.parquet, splits/default_splits.parquet
                                 extract-features -->  processed/features/features_v1.parquet (96 Spalten)
                                       |
                                     train  -->  data/artifacts/<run_name>_<timestamp>/{summary.json, *.pkl | model.pt}
                                       |
                                    explain / analyze-batch -->  <run>/explanations/*.json, *_overlay.png, *_gradcam.npy
```

Wichtige Designentscheidungen, die bereits gut gelöst sind:

- **Leakage-sichere Splits**: Original und seine Crops teilen eine `split_group_id`; Test-Split
  (20 %) und 5 Dev-Folds werden gruppiert und nach Score-Bins stratifiziert gezogen.
- **Append-only Rohlabels** pro Rater als CSV; mehrere Rater und Mehrfachbewertungen sind im
  Schema vorgesehen (`n_raters`, `std_score`, `agreement_index`, Rater-z-Normalisierung).
- **Exclusions** statt Löschen: `exclusions.csv` wird beim Lesen gefiltert, Bilder bleiben erhalten.
- **Import-Batches** verbinden Import und spätere Batch-Analyse mit einem Modell.

## 3. Module im Detail

### `data/`
- `manifest.py`: Bild-Inventar (sha256, pHash, Größe, Session, Split-Gruppe).
- `ratings.py`, `pairwise.py`: Schreiben/Lesen von Skalar- und Paarvergleichen; Pseudo-Paare aus
  Skalarlabels (Score-Abstand ≥ 2) für Ranking-Modelle.
- `aggregation.py`: Mittel, Median, getrimmtes Mittel, Std, SEM, Agreement, normalisierter Score.
- `splits.py`: GroupShuffleSplit + StratifiedGroupKFold.
- `web_dataset.py`: Download lizenzfreier Vergleichsbilder (Openverse-API, Lizenzfilter).

### `preprocessing/`
- `importer.py`: Kopiert Originale, erzeugt bis zu 8 quadratische „Sky-Crops“ pro Bild per
  HSV-Heuristik (Himmel-/Wolkenanteil), schreibt `image_derivatives.parquet`.
- `image_ops.py`: HSV-Wolkenmaske (`V>110 & S<120`), einfache Saliency.
- `transforms.py`, `datasets.py`: Albumentations-Pipelines und Torch-Dataset.

### `features/` (96 handgebaute Spalten)
- `color.py`: Helligkeit, Kontrast, Dynamikumfang, Sättigung, Lab-a/b, Blau-Orange-Balance,
  Farbtemperatur-Proxy, Wolken-/Himmelanteil, 8 Hue-Bins.
- `texture.py`: Kantendichte, LBP, GLCM, Entropie, Blobs, Komponenten, DoG-Energie,
  Fraktaldimension, 8 Kantenorientierungs-Bins.
- `composition.py`: Horizontlage, vertikale Massenverteilung, Helligkeit oben/unten, Symmetrie,
  Negativraum, Saliency-Schwerpunkt, Drittel-Regel-Abstand, dominante Wolkenmasse.
- `scientific.py`: 14 „meteo_*“-Features und ein **regelbasierter** Wolkentyp-Klassifikator
  (10 Typen, handgewichtete Sigmoiden). Für Crops wird das Ergebnis des Originals übernommen.
- `concept_bootstrap.py`: 10 „Konzepte“ (towering, dramatic edges, rim lighting, …) als
  handgewichtete Sigmoiden **über andere Features** – also keine gelernten oder unabhängigen Signale.

### `models/`
- `baseline.py`: ElasticNet, RandomForest, HistGradientBoosting auf den Features; plus
  Pairwise-LogReg auf Feature-Differenzen; Konformal-Intervall aus Val-Residuen.
- `deep.py`: `CloudRatingNet` = torchvision-Backbone (ResNet18, **eingefroren**) + Dropout + Linear,
  optional sigmoid-begrenzt auf 0–10; MSE + Pairwise-Hilfsloss innerhalb des Batches; Early Stopping,
  Checkpoints, Geräteauswahl (CUDA/ROCm/DirectML/CPU).
- `hybrid.py`: Features + PCA-„Embeddings“ der Features (kein echtes Bild-Embedding) → HistGB.
- `ranking.py`: Pairwise-LogReg.
- `concept.py`: Konzept-Flaschenhals (Lasso auf benannten Konzepten, Bootstrap-Stabilität, Beiträge pro Bild,
  Vokabular-Vergleich); Ergebnis des Runs liegt als `concepts.parquet`, `contributions.parquet`, `vocabulary.parquet` im Run-Ordner.

### `friends/`
- `registry.py` (Tokens ↔ Rater), `queue.py` (Reihenfolge pro Freund), `server.py` (FastAPI), `ui.py` (Handy-Seite).

### `reports/year_report.py`
- Datensammlung über alle Artefakte und Jinja2-Rendering des HTML-Berichts mit eingebetteten Bildern.

### `eval/cloud_atlas.py`
- Wolkentyp-Votum (CLIP + Judges) mit Zuverlässigkeitsstufen, Atlas-Tabelle, Typ × Licht, Meteorologie-Texte.

### `eval/rater_analytics.py`
- Sitzungen, Residuen gegen Referenz, Drift, Anker-Effekt, Skalennutzung, Streitbilder, Rater-Matrix.

### `eval/semantic_map.py`
- UMAP/PCA-Projektion der Embeddings, Kartentabelle mit allen Signalen, k-Means-Regionen mit Lücken-Score,
  Thumbnail-Data-URIs für die Karte.

### `explain/`
- `feature_importance.py`: Permutation Importance; lokale Beiträge per „Median-Baseline, ein Feature
  austauschen“ (Ein-Feature-Sensitivität, keine SHAP-Werte, obwohl shap installiert ist).
- `heatmaps.py`: Input-Gradient, Grad-CAM pro ResNet-Layer (eigene Hook-Implementierung),
  Feature-Diagnosekarten (Maske, Kanten, Helligkeit …), Kanten-Fallback.
- `regions.py`: SLIC-Superpixel × Heatmap → Regionentabelle.
- `interpretation.py`: Deterministische „Interpretationskarte“ (Score-Band, Konfidenzband,
  Treiber, Konzepte, Wolkentyp, Cautions, Empfehlung) + optionaler Ollama-Textgenerator.

### `app/` (Streamlit)
Import · Rate (Skalar + Paarvergleich, Hotkeys, Reihenfolge-Modi, Skip) · Train · Inspect
(Heatmaps, Signale, Nachbarn) · Compare · Gallery · Batch Report · Exclusions · Source Groups.

## 4. Datenlage (24.10.2026)

| Kennzahl | Wert |
|---|---|
| Bilder im Manifest | 939 nach Deduplizierung (229 Originale: 89 privat + 140 Web; 710 Crops; 10 ausgeschlossen) |
| Rohbewertungen | 321 von einem Rater (`friend_a`), alle am 26.04.2026 |
| Bewertete Bilder | 312 (161 private Crops, 18 private Originale, 133 Web-Originale), 8 doppelt bewertet |
| Split (neu, 24.10.2026) | 265 Dev (5 Folds) / 47 Test |
| Noch unbewertet | ~630 Bilder, davon ~78 Originale |
| Quelle der privaten Fotos | `G:\Wolken`, 90 Dateien (alle bereits importiert) |

Score-Verteilung: Mittel 6,3, linksschief (Häufungen bei 8–9,5; Spitze 8,5). Die 8 Doppelbewertungen
liegen meist 0–1 Punkt auseinander (zwei Fälle 1,5 und 2,5) – ein erster Hinweis auf die
Test-Retest-Zuverlässigkeit, aber viel zu wenig für eine belastbare Rauschgrenze.

### Zentraler Befund: Crop-vs-Original-Confound

| Gruppe | n | Mittelwert | Std |
|---|---|---|---|
| private Originale | 18 | 8,7 | 1,8 |
| Web-Originale | 135 | 7,5 | 2,0 |
| private Crops | 161 | 5,0 | 2,5 |

Ob ein Bild ein Crop ist, korreliert mit dem Score mit Spearman ρ = −0,51. Die am stärksten
korrelierten Features (DoG-Energie 0,62; Entropie 0,56; Kantendichte 0,55; LBP −0,54) messen
im Kern **Detailreichtum/Schärfe** – genau das, was kleinen, hochskalierten Crops fehlt.
Innerhalb der Crops bleiben diese Korrelationen stark (ρ ≈ 0,6–0,68), innerhalb der Web-Originale
sind sie deutlich schwächer (ρ ≈ 0,3–0,37). Das aktuelle Modell lernt also zu einem erheblichen Teil
„Crop oder nicht“ bzw. „scharf oder matschig“, nicht die ästhetische Feinunterscheidung.
Für die 10-vs-8-Frage ist das der wichtigste zu klärende Punkt (siehe Roadmap, P1).

## 5. Modellstand

Beide vorhandenen Runs stammen vom 26.04.2026 (mit damaligem Labelstand):

| Run | Val (Fold 0, n=21) | Test (n=48) |
|---|---|---|
| baseline RandomForest | MAE 1,83 · ρ 0,52 | MAE 1,39 · ρ 0,67 · R² 0,54 |
| baseline HistGB | MAE 1,76 · ρ 0,53 | MAE 1,50 · ρ 0,69 · R² 0,51 |
| baseline ElasticNet | MAE 2,08 · ρ 0,48 | MAE 2,81 · R² −0,55 (kollabiert) |
| deep ResNet18 (frozen, 5 Ep., CPU) | MAE 1,70 · ρ 0,54 | MAE 1,62 · ρ 0,50 · R² 0,30 |
| pairwise LogReg | – | Paar-Genauigkeit 0,70 · AUC 0,75 |

### Neuer Baseline-Run mit 5-fach-CV (24.10.2026, `baseline_v1_20261024T115937Z`)

| Modell | OOF (265 Dev) | Fold-Streuung | Test (47) | OOF nur Originale (116) | OOF nur Crops (149) |
|---|---|---|---|---|---|
| ElasticNet | MAE 1,50 · ρ 0,64 | ±0,21 · ±0,15 | MAE 1,39 · ρ 0,57 | ρ 0,32 | ρ 0,70 |
| HistGB | MAE 1,51 · ρ 0,63 | ±0,22 · ±0,13 | MAE 1,45 · ρ 0,57 | ρ 0,33 | ρ 0,67 |
| RandomForest | MAE 1,52 · ρ 0,64 | ±0,19 · ±0,14 | MAE 1,38 · ρ 0,59 | ρ 0,37 | ρ 0,66 |

Die Fold-Streuung von ±0,15 bei Spearman zeigt, wie wenig eine einzelne Fold aussagt. Entscheidend:
**innerhalb der Originale** liegt die Rangkorrelation nur bei 0,32–0,37, innerhalb der Crops bei 0,66–0,70.
Das Gesamtergebnis (0,64) wird vom Crop-Confound getragen. Für die 10-vs-8-Frage bei echten Fotos ist
das aktuelle Feature-Modell also noch schwach – genau das motiviert Roadmap P2 und P3.

### Embedding-Runs (24.10.2026, P3)

Alle Bilder einmal mit CLIP ViT-B/32 und DINOv2-S encodiert (je ~40 ms pro Bild auf CPU, Cache unter
`data/processed/embeddings/`). Darauf Ridge/kNN/SVR mit derselben 5-fach-CV:

| Run / Modell | OOF ρ | Fold-Std | Test ρ | OOF ρ Originale | OOF ρ Crops | Test ρ Originale |
|---|---|---|---|---|---|---|
| **CLIP · Ridge (α=300)** | **0,72** | 0,13 | **0,76** | **0,47** | 0,75 | **0,68** |
| CLIP · SVR | 0,72 | 0,10 | 0,70 | 0,50 | 0,71 | 0,57 |
| Hybrid (CLIP+DINO+Features+Zero-Shot) · Ridge | 0,72 | 0,16 | 0,74 | 0,48 | 0,73 | 0,64 |
| CLIP · kNN (k=10, cosine) | 0,67 | 0,14 | 0,68 | 0,32 | 0,68 | 0,55 |
| DINOv2 · SVR | 0,65 | 0,13 | 0,64 | 0,38 | 0,67 | 0,47 |
| Baseline Features · RandomForest | 0,64 | 0,14 | 0,59 | 0,37 | 0,66 | 0,41 |
| DINOv2 · Ridge | 0,63 | 0,12 | 0,70 | 0,28 | 0,68 | 0,61 |

Lesart: CLIP-Embeddings sind der klar beste Einzelinput. Der Sprung ist dort am größten, wo es zählt:
innerhalb der Originale (0,37 → 0,47 OOF, 0,41 → 0,68 auf dem kleinen Test-Set mit 35 Originalen).
DINOv2 (rein visuell, ohne Sprachbezug) bringt weniger; die Kombination aller Inputs bringt gegenüber
CLIP allein nichts – bei 265 Trainingsbildern dominiert Regularisierung (Ridge wählt das größte α).
MAE bleibt bei ~1,35 Punkten; die Konformal-Halbbreiten (~3) sind weiterhin breit, weil Michels
Labels selbst eine Test-Retest-Streuung von etwa ±1 Punkt haben (siehe Doppelbewertungen).

### Crop-Confound aufgelöst (24.10.2026, P2)

Die Crops sind nicht „matschig“: Median der kurzen Kante 1058 px, nur 136 von 710 unter 768 px, und die
kleinen Crops wurden sogar *besser* bewertet (6,1) als die großen (4,8). Entscheidend ist der **Szenenanteil**:
527 Crops zeigen nur 35 % der kurzen Kante (≈ 12 % der Fläche), also Detailausschnitte ohne Komposition,
Horizont und Lichtverlauf. Deshalb gibt es jetzt `crop_scale` und `crop_class`:

| crop_class | Definition | Bilder | gelabelt | Ø Score |
|---|---|---|---|---|
| original | Vollbild | 229 | 151 | 7,60 |
| large_crop | Crop ≥ 50 % der kurzen Kante und ≥ 768 px | 139 | 34 | 5,41 |
| small_crop | Crop < 50 % der kurzen Kante oder < 768 px | 571 | 127 | 4,88 |

Was sich geändert hat:
- Rate- und Inspect-Seite zeigen bei Crops das Quellbild mit markiertem Ausschnitt und dem Szenenanteil;
  beim Labeln lässt sich nach Bildart filtern (für die 10-vs-8-Frage: „Nur Originale“).
- Import-Defaults: Crop-Skalen 0,5 / 0,7 / 0,9 statt 0,35–1,0, Mindestkante 768 px, Kandidaten unter der
  Mindestgröße werden übersprungen statt vergrößert. Bestehende Crops bleiben, sind aber klassifiziert.
- Alle Modell-Summaries berichten Metriken pro `crop_class`; Modell-Configs können per `include_kinds`,
  `include_crop_classes`, `include_datasets` auf Teilmengen trainieren.

Kontrollexperiment „nur Originale“ (`embedding_clip_originals.yaml`, 116 Dev / 35 Test):
CLIP-Ridge OOF ρ 0,43 (Test 0,60), SVR 0,47 (Test 0,44). Das entspricht der Originale-Teilmetrik des
Modells, das auf *allen* Bildern trainiert wurde (ρ 0,47, Test 0,68). Die Crops schaden dem Modell für
Originale also nicht, sie helfen eher leicht; der Confound saß in der Bewertung des Gesamtergebnisses,
nicht im Training. Konsequenz: weiter auf allem trainieren, aber **immer nach `crop_class` berichten**, und
für die Feinanalyse Originale labeln. Innerhalb der Crops erreicht CLIP-Ridge ρ 0,74–0,79, innerhalb der
Originale 0,47: Der schwierige Teil ist das Ranking unter echten Fotos.

### LLM-Judge (24.10.2026, P4)

`cloud_aesthetics/judge/`: Ein Vision-Modell bewertet Bilder mit derselben 0–10-Rubrik und liefert ein
strukturiertes Urteil (Score, Konfidenz, Wolkentypen, Licht, Farben, Stärken, Schwächen, Qualitätsflags,
Ein-Satz-Fazit). Jedes Modell ist ein eigener Rater `llm_<modell>`, in `raters.csv` als `llm` registriert und
aus dem menschlichen Konsens ausgeschlossen. Urteile werden pro Bild gecacht, Attribute in eine Tabelle
geflacht, der Vergleich mit den menschlichen Labels (Spearman, MAE, Bias, ±1-Trefferquote, pro `crop_class`,
oberes Ende der Skala, größte Abweichungen) läuft per `judge-report` und auf der Seite „Rater Comparison“.

Zwei Backends: der **Uni-LLM-Server** (OpenAI-kompatibles Gateway `ki-chat.uni-mainz.de`, Modelle
`deepseek-v4.1-flash` und `qwen3.6-35b`, beide bildfähig, ca. 3 s pro Bild, Key aus der Secrets-Datei des
Spider-Projekts) und die **Anthropic-API** (Claude Opus 5.5 mit strukturierter Ausgabe, braucht
`ANTHROPIC_API_KEY`). Der Uni-Server ist der Standard, weil er kostenfrei ist und bereits genutzt wird.

### Judge-Validierung gegen Michels 312 Labels (24.10.2026)

Beide Uni-Modelle haben alle gelabelten Bilder bewertet (ca. 3 s pro Bild; Gateway-Limit 100 Anfragen pro
Fenster, der Judge wartet automatisch bis zum Reset). Ergebnis gegen Michels Mittelwert pro Bild:

| Rater | n | ρ gesamt | MAE | Bias | ±1 Punkt | ρ Originale | ρ bei Mensch ≥ 7,5 | ρ Originale ≥ 7,5 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 311 | 0,70 | 1,57 | −0,66 | 44 % | 0,39 | 0,42 (n=139) | 0,28 (n=100) |
| Qwen3.6 35B | 312 | 0,69 | 1,46 | −0,14 | 48 % | 0,35 | 0,41 | 0,29 |
| Mittel beider Judges | 311 | 0,71 | 1,49 | | | 0,40 | 0,42 | 0,29 |
| zum Vergleich: CLIP-Ridge (OOF, Dev) | 265 | 0,72 | 1,34 | | | 0,47 | | |
| CLIP-Ridge + Judge-Mittel | 265 | 0,75 | | | | 0,50 | | |

Lesart:
- **Grob ja, fein nein.** Die Judges trennen gute von schlechten Bildern sauber (Mensch < 5 → Judge ≈ 4,0–4,5;
  Mensch ≥ 9 → Judge ≈ 7,0–7,4), komprimieren aber die Skala (Std 1,6 statt 2,6) und erreichen innerhalb der
  Originale nur ρ ≈ 0,35–0,40, am oberen Ende 0,28. Für die 10-vs-8-Frage ersetzen sie Michels Labels nicht.
- **Die Judges sind sich einig (ρ = 0,90 untereinander, 0,81 innerhalb der Originale), aber nicht mit Michel.**
  Das ist das eigentlich interessante Ergebnis: Der Unterschied ist Geschmack, nicht Rauschen.
- **Woran sich der Geschmack unterscheidet** (größte Abweichungen): Michel gibt 9,5–10 für klare, saubere Himmel
  mit verstreuten Cumulus und Kondensstreifen oder für pastellfarbene Strand-Dämmerungen, die beide Judges als
  „ordinary“ einstufen (5,5–6). Umgekehrt werten die Judges düstere, landschaftslastige Szenen mit Wolkenschelf
  oder Glühen hinter Bergen auf (6,5–7), die Michel mit 2,5–3 abtut. Bei Michels 9+-Originalen monieren die
  Judges systematisch „ausgefressene Sonne“, „überprozessierte Farben“ und „unruhiger Vordergrund“ – Kriterien,
  die für Michels Rating offenbar keine Rolle spielen. Hypothese: Michels Skala belohnt Helligkeit, Klarheit,
  Pastell und Lichtstimmung; die Modelle belohnen Struktur, Drama und fotografische Technik.
- **Als Attribut-Layer taugen die Judges sofort.** Mittlerer Michel-Score nach DeepSeek-Wolkentyp: Lenticular 8,2,
  Cumulus 7,3, Cumulonimbus 7,2, Cirrus 7,0, Altocumulus 6,9, Stratocumulus 6,7, Altostratus 3,2, Stratus 0,8.
  Nach Licht: goldene Stunde 7,4 (n=51) vs. Tageslicht 6,0 (n=221); Sonnenuntergangs-Flag 6,9 vs. 6,0.
  Qualitätsflags (52× unscharf/verrauscht, 41× landschaftsdominant, 13× Flugzeugfenster) sind brauchbare
  Kandidaten für Exclusions.
- **Konsequenz für die Skalierung:** Judge-Labels eignen sich als günstiges Vortraining/Vorfilter für die
  ~630 ungelabelten Bilder (Grobsortierung, Attribute, Exclusion-Vorschläge), nicht als Ersatz für menschliche
  Labels am oberen Ende. Die Kombination CLIP-Ridge + Judges hebt die Originale-Korrelation leicht (0,47 → 0,50).

Reports: `data/artifacts/judge_reports/<rater>_vs_human.{md,json}` plus Paar-Tabelle; UI: Seite „Rater Comparison“.

**Qualitätsflags vs. Michels Geschmack** (Kalibrierung der Exclusion-Vorschläge an den gelabelten Bildern):
`not_a_cloud_photo` traf 3 Bilder, alle von Michel ≤ 1 bewertet – das einzige Flag, das seinen bisherigen
Ausschlüssen (10× `not_cloud_training_data`) entspricht. `landscape_dominant` (69 Bilder, Michel im Mittel 6,9,
43 davon ≥ 7, drei 10er) und `airplane_window` (13 Bilder, Mittel 7,2) markieren dagegen *Lieblingsbilder*:
Sonnenuntergänge mit Landschaft und Fensterblicke gehören für Michel zum Genre. Deshalb sind diese Flags in der
Triage nur Hinweise; Vorschläge gibt es für „kein Wolkenfoto“, einstimmige Satelliten-/Objekt-Flags bei
Judge-Score ≤ 4 und einstimmig „unscharf“ bei Judge-Score ≤ 3 (9 von 11 solcher Bilder hatte Michel ≤ 3).

### Triage der ungelabelten Bilder (P4, Schritt 2/3)

`judge-triage` kombiniert alle Modell-Rater pro Bild (Mittel, Spannweite, Mehrheits-Wolkentyp und -Licht, Flags)
zu einer Label-Warteschlange (ungelabelte Originale zuerst, hoher Judge-Score zuerst, Uneinigkeit als
Tie-Breaker) und zu Exclusion-Vorschlägen. Die Seite „Triage“ zeigt Warteschlange, unsichere Bilder (Judges
≥ 2 Punkte auseinander) und Vorschläge mit Häkchen; die Rate-Seite kennt die Reihenfolge als „Judge-Vorschlag“.
Nichts wird automatisch ausgeschlossen oder gelabelt.

Stand 24.10.2026 nach dem Lauf beider Judges über alle 939 Bilder (627 ungelabelt):
- Die Judges stimmen auf den ungelabelten Bildern untereinander mit ρ = 0,87 überein.
- **Label-Warteschlange:** 78 ungelabelte Originale, davon 30 mit Judge-Mittel ≥ 7 (keines ≥ 8). Die Spitze ist fast
  durchweg „goldene Stunde“ mit Altocumulus/Stratocumulus über Landschaft, dazu ein Lenticularis über einem See –
  genau der Bildtyp, bei dem Michels Rating und das der Judges am weitesten auseinanderliegen. Diese 30 Bilder zu
  labeln ist der schnellste Weg zu mehr Signal am oberen Ende.
- Ungelabelte Crops: 105 große (Judge-Mittel 5,2), 444 kleine (5,0); die kleinen Crops sind zu gut 60 % Cumulus,
  Altocumulus und Cirrus, mit 83 Altostratus- und 14 Stratus-Fällen (Judge-Mittel 4,1 bzw. 1,8) als Kandidaten für
  das untere Ende.
- **Exclusion-Vorschläge:** 9 ungelabelte (8× „kein Wolkenfoto“/Objekte, 1× unscharfer Crop) plus 4 gelabelte,
  die Michel ohnehin mit 0–1 bewertet hat. Hinweise ohne Vorschlag: 141× unscharf, 109× landschaftsdominant,
  90× Detailcrop, 47× überprozessiert, 9× Flugzeugfenster.
- 14 ungelabelte Bilder, bei denen die Judges ≥ 2 Punkte auseinanderliegen (Tab „Unsicher“).

### „10 vs 8“-Analyse (24.10.2026, P5)

`eval/contrast.py`, CLI `contrast`, Seite „10 vs 8“: Zwei Bänder von Michels Score werden über alle Signale
verglichen – 90 handgebaute Features, 52 CLIP-Zero-Shot-Spalten, Judge-Attribute (Mittel, Konfidenz, Typ- und
Licht-Anteile, Flags) – mit Cohen's d, Mann-Whitney-Test, Benjamini-Hochberg-q-Werten, dazu Spearman-Trends
innerhalb der guten Bilder (≥ 7) und Beispielkacheln beider Bänder.

Ergebnis für **Originale, 9,5–10 (n=34) vs. 7,5–8,5 (n=51)**: Kein einziges der 149 Signale überlebt die
FDR-Korrektur (bestes q = 0,19); 6 Signale haben |d| ≥ 0,5. Die größten Effekte:

| Signal | Ø Band A (10er) | Ø Band B (8er) | d |
|---|---|---|---|
| Anzahl Judge-Flags | 0,31 | 0,58 | −0,70 |
| Judge-Typ Cumulus | 0,22 | 0,48 | −0,62 |
| Judge-Typ Lenticular | 0,10 | 0,00 | +0,61 |
| Judge-Konfidenz | 0,84 | 0,82 | +0,60 |
| vertikale Wolkenmasse Mitte | 0,54 | 0,68 | −0,52 |
| Judge-Score | 7,3 | 6,9 | +0,45 |
| Helligkeit unten | 0,38 | 0,32 | +0,44 |
| Sättigung | 70 | 83 | −0,33 |

Trend innerhalb der Originale ≥ 7 (n=112): Dynamikumfang +0,29, CLIP-Cirrocumulus +0,28, CLIP-Cirrus +0,26,
CLIP „towering structure“ −0,28, CLIP-Cumulonimbus −0,26, Judge-Score +0,25, LBP-Mittel −0,24 (keines unter q 0,14).

Lesart: Die 10er sind für die Judges „saubere“ Bilder (weniger Qualitätsflags, höhere Konfidenz), eher
Lenticularis/Cirrus-artig als Cumulus, mit weniger Wolkenmasse in der Bildmitte, hellerem unterem Bilddrittel
und *geringerer* Sättigung – also eher luftig-hell als bunt-dramatisch. Das passt zur Geschmacks-Hypothese aus
der Judge-Validierung. Statistisch ist das bei 34 gegen 51 Bildern aber alles noch Hypothese: Mit ~100 weiteren
gelabelten Originalen am oberen Ende (die Triage-Warteschlange) werden Effekte dieser Größe signifikant.

Kontrollrechnung über **alle Bildarten** (36 vs. 86): Dort werden Dynamikumfang, Kontrast, Entropie und
Judge-Score hochsignifikant (q < 0,01) – das ist wieder der Crop-Confound (die 8er enthalten viele Crops), nicht
die Feinunterscheidung. Die Seite hat deshalb den Bildart-Filter standardmäßig auf Originale.

### Wissenschaftlicher Layer: Aufnahmekontext, Sonnenstand, Wetter, Farbpalette (24.10.2026, S3)

- **Aufnahmekontext** (`features/capture.py`): EXIF DateTimeOriginal/OffsetTime, GPS-Position und GPS-UTC-Zeitstempel;
  Fallback auf Dateinamen (`PXL_…` = UTC, `IMG_…` = Ortszeit, `IMG-…-WA…` = nur Datum). Jede Zeile trägt
  `time_quality` (exact_utc, local_with_offset, local_estimated_offset, date_only, unknown) und `location_source`
  (exif, default = Mainz, none). Datenlage der 229 Originale: 67 exakt UTC, 7 Ortszeit, 19 nur Datum, 136 unbekannt
  (alle Web-Bilder); 70 mit GPS.
- **Sonnenstand** (`features/solar.py`, NOAA-Näherung ohne Fremdbibliothek, getestet gegen Sonnenuntergang Mainz):
  Elevation, Azimut, Stunden bis/seit Sonnenuntergang, Phase (night … blue_hour, golden_hour, low_sun, day).
  Unter den privaten Originalen mit Zeit: 43 Tag, 18 goldene Stunde, 5 tiefe Sonne, 4 blaue Stunde, 3 nautische Dämmerung.
- **Wetter** (`features/weather.py`): Open-Meteo-Archiv (kostenlos, ohne Key), pro Ort/Tag gecacht; nächste Stunde
  für Bewölkung gesamt/tief/mittel/hoch, Feuchte, Druck, Niederschlag, Temperatur, Wind, plus Tagesaggregate für
  Nur-Datum-Bilder. 93 Originale mit Wetter.
- **Farbpalette** (`features/palette.py`, jetzt Teil von `features_v1`): k-Means (k=5) in CIELAB → Warm-/Blau-/
  Neutralanteil, Farbton-Konzentration (zirkulär), Paletten-Kontrast (ΔE), Chroma, Helligkeit und -spanne,
  dominante Farbe, Hasler-Süsstrunk-Colorfulness. 13 Spalten.
- Crops erben den Kontext ihres Originals; Tabelle `data/processed/context/capture_context.parquet`, CLI `extract-context`.

**Palette in der 10-vs-8-Analyse (Originale, 34 vs. 51):** Neutralanteil der Palette höher bei den 10ern
(0,37 vs. 0,24, d = +0,44), Helligkeitsspanne höher (d = +0,39), Blauanteil niedriger (d = −0,30); im Trend
ab Score 7 (n=112) Neutralanteil +0,23, Colorfulness −0,18, Chroma-Maximum −0,17. Das deckt sich mit „luftig-hell
statt bunt“: Michels Spitzenbilder haben mehr Weiß/Grau/Pastell und weniger satte Farbe als die 8er. Für die
Baseline-Modelle ändert die Palette fast nichts (RandomForest OOF ρ 0,642 → 0,644, Originale 0,37 → 0,37).

**Ehrliche Grenze:** Von den 151 gelabelten Originalen haben nur 14 eine Aufnahmezeit, weil 133 davon Web-Bilder
ohne EXIF sind. Damit ist jede Aussage zu Sonnenstand oder Wetter und Rating derzeit anekdotisch (z. B. Temperatur
ρ 0,49 bei n=14). Der Layer wird in dem Moment aussagekräftig, in dem die 78 ungelabelten privaten Originale –
fast alle mit EXIF und GPS – gelabelt sind. Das ist dieselbe Warteschlange wie für die 10-vs-8-Frage.

### Labeling-UX (24.10.2026, S1)

Rate-Seite: Score-Buttons 0–10 und Zifferntasten, Schnellmodus, Undo über eine Tombstone-Datei
(`data/raw/metadata/ratings/ratings_void.csv`; die Rater-CSVs bleiben append-only, der Lader filtert), Anzeige
bisheriger Bewertungen pro Bild, Judge-Scores standardmäßig ausgeblendet. Zweiter Tab „Duelle & Paarvergleich“:
Pool aus eigenen Bildern ab Schwelle (Standard 8,0, nur Originale), optional ungelabelte Triage-Kandidaten mit
Judge ≥ 7; Paarauswahl bevorzugt Bilder mit wenigen Duellen und ähnlichem Score, kein Paar wird wiederholt.
Bradley-Terry-Fit (`data/pairwise.py`, MM-Algorithmus mit kleinem Prior) liefert `bt_score`/`duels`, auch in
`aggregated_labels.parquet`. Die Rangfolge aus Duellen ist das geeignete Zielmaß für die Feinunterscheidung am
oberen Ende; sobald genug Duelle vorliegen, sollten die Embedding-Modelle auch darauf trainiert werden.

### Modell-Diagnostik: SHAP, Kalibrierung, Ranking (24.10.2026, S2)

- **SHAP** (`explain/shap_values.py`): exakte TreeExplainer-Beiträge für RandomForest/HistGB, LinearExplainer für
  ElasticNet; im Explain ersetzen sie die Ein-Feature-Sensitivität (Attributions-Art `tree_shap`/`linear_shap`),
  pro Baseline-Run liegt die globale Wichtigkeit (mittlerer |SHAP| und Richtung) als Parquet und auf der Train-Seite.
  Befund: `dog_mean_abs` (Feinstruktur-Energie) dominiert den RandomForest mit 0,83 Score-Punkten mittlerem |SHAP|,
  alle anderen Features liegen unter 0,1 – das Modell lebt vom Schärfe-/Crop-Signal. Danach folgen Entropie,
  Kantenorientierung, Lab-a, Blob-Größe (negativ) und die neue Paletten-Helligkeit (negativ: hellere Palette →
  niedrigerer Score im Gesamtbild, weil helle flache Himmel oft Crops sind).
- **Kalibrierung** (`eval/calibration.py`, Train-Seite): Ist-vs-Vorhersage mit Bin-Mitteln, Residuen-Histogramm,
  Bias, Streuungsverhältnis Vorhersage/Ist, ECE und empirische Abdeckung des Konformal-Intervalls, wählbar pro
  Modell und Partition (OOF/Test).
- **Ranking-Suite** (`models/ranking.py`, CLI `train` mit `kind: ranking`): logistische Regression auf
  Eingabedifferenzen, trainiert auf Dev-Paaren (explizite Duelle + Pseudo-Paare aus Scores, max. 20.000),
  evaluiert auf Test-Paaren; jedes Bild bekommt einen latenten Rang-Score. Ergebnis mit CLIP-Embeddings:
  Paar-Genauigkeit auf 598 Test-Paaren 0,85, AUC 0,94, Spearman der Bildscores zum menschlichen Mittel 0,68 (Test);
  mit handgebauten Features 0,77 / 0,84 / 0,42. Sobald Duelle vorliegen, kann `pair_sources: [explicit]` die
  Rangordnung ausschließlich aus Michels Duellen lernen.
- **BT-Ziel**: `embedding_clip_bt.yaml` regressiert `bt_score` statt `mean_score`; Zeilen ohne Duelle werden
  übersprungen, der Run startet erst ab 40 Bildern mit Duellen (`min_rows`). Derzeit gibt es noch keine Duelle.

### Aktive Label-Warteschlange und Zuverlässigkeit (25.10.2026)

- `predict-all` bewertet alle 939 Bilder mit einem trainierten Run (Standard: CLIP-Ridge/kNN/SVR); unter den
  627 ungelabelten Bildern liegen 86 bei Modell ≥ 7,5, bei 106 sind sich die drei Modelle um ≥ 1,5 Punkte uneinig.
- Die Warteschlange (`judge-triage`) kombiniert jetzt Judge-Mittel und Modellvorhersage zu einer Schätzung und
  ordnet nach erwartetem Informationsgewinn: Originale zuerst, Schätzung nahe dem oberen Ende, Uneinigkeit
  zwischen Judges, zwischen den Modellen und zwischen Modell und Judges. Jede Zeile erklärt sich (`queue_reason`,
  in der Rate-Seite als „Warum jetzt“). Ergebnis: 29 ungelabelte Originale mit Schätzung ≥ 7,5, 45 Bilder mit
  Modell-Judge-Abstand ≥ 2 – die Spitze der Liste sind Originale, bei denen das CLIP-Modell 8,5–9 sieht, die Judges
  aber nur 6–7 geben.
- **Blinde Wiederholungen**: 5 % (mindestens 5, höchstens 20) der eigenen Bewertungen werden nach ≥ 1 Tag erneut
  in die Warteschlange gelegt, über die Skala gestreut; die frühere Note bleibt verborgen, nach dem Speichern
  zeigt die Seite die Differenz. Home zeigt daraus mittlere Abweichung, Anteil innerhalb ±1, Drift und die
  geschätzte Rausch-Obergrenze für Spearman. Stand: 8 Paare aus April, Ø |Differenz| 0,9, 75 % innerhalb ±1,
  Drift +0,3, geschätzte Obergrenze ρ ≈ 0,92 – bei n=8 noch nicht belastbar (wird ab ~30 Paaren stabil).

### Heatmaps: Grad-CAM-Bewertung und CLIP-Saliency (27.10.2026)

**Bewertung des bestehenden Grad-CAM** (Deep-Run ResNet18, eingefrorenes Backbone, April): technisch korrekt,
inhaltlich kaum brauchbar. Layer 4 hebt ImageNet-Objekte hervor (bei einer 10 den Baum, bei einer 3 die dunkle
Baummasse), Layer 3 zeigt verstreute Blobs, der Input-Gradient ist Rauschen; nur 30–37 % der Kartenmasse liegen
in den obersten 20 % der Pixel. Dazu ist das Modell selbst schwach (eine 3,0 wird als 6,4 vorhergesagt). Für die
Frage „woher kommt der Score“ taugt es nicht.

**Neu: Saliency für das beste Modell** (`explain/clip_saliency.py`, in `explain` und Inspect integriert):
- *Occlusion*: 8×8-Raster über dem Mittelquadrat, das CLIP sieht; jede Zelle wird verdeckt, neu eingebettet und
  neu bewertet. Angezeigt wird der Score-Verlust relativ zur typischen Zelle als divergierende Karte (rot stützt,
  blau senkt). Modellagnostisch und treu zum tatsächlichen Regressor; etwa 2 s pro Bild.
- *Token-Grad-CAM*: Gradient des Ridge-Scores nach den Patch-Tokens des vorletzten Transformer-Blocks
  (der letzte Block hat an den Patch-Tokens keinen Gradienten, weil CLIP nur das CLS-Token poolt).
- Befund auf vier Originalen: Bei der 10 liegt die Stützung auf dem leuchtenden Wolkenband, bei der 9 auf der
  Sonne am Horizont, bei der 8,5 auf den warm beleuchteten Wolkenschichten. Bei der 3,0 stützt überraschend die
  dunkle Baummasse den (zu hohen) Score; das ist ein echter Hinweis auf ein Modellproblem, nicht ein Artefakt.
  Verdecken irgendeiner Zelle senkt den Score fast immer etwas (typischer Effekt 0,5–0,9 Punkte), darum die
  Zentrierung.

### Kontrafaktischer Explorer (27.10.2026, T1)

`explain/counterfactual.py`, Seite „Was wäre wenn“, CLI `counterfactual-sweep`. Sechs Operatoren (Kontrast,
Sättigung, Belichtung, Wärme, Zoom, Unschärfe) verändern ein Foto, CLIP bettet es neu ein, der Ridge-Regressor
bewertet es. Pro Bild: Live-Score, Sensitivitätskurven, bester Einzel-Eingriff. Pro Score-Band: gemittelte Kurven.

Gemittelt über die Originale (51 Achter 7,5–8,5 und 34 Zehner 9,5–10), Δ Modell-Score gegenüber dem Original:

| Eingriff | 8er | 10er | Lesart |
|---|---|---|---|
| Unschärfe 3 px / 6 px | −1,8 / −2,7 | −2,2 / −3,4 | Schärfe ist das stärkste Modellsignal (wieder der Crop-Confound) |
| Zoom 1,5× / 2× | −0,9 / −2,0 | −1,1 / −1,9 | Ausschnitt verkleinern kostet, Szenenkontext zählt |
| Belichtung −1 / +1 EV | −1,0 / −1,4 | −1,1 / −2,0 | Original liegt nahe am Optimum, beide Richtungen schaden |
| Wärme −1 / +1 | −0,6 / −0,5 | −0,8 / −0,7 | dito |
| Kontrast 0,5× / 1,6× | −1,5 / +0,3 | −1,7 / +0,1 | weniger Kontrast schadet, mehr hilft minimal |
| Sättigung 0× / 2× | −0,3 / +0,35 | −1,0 / +0,3 | Entsättigen schadet, bei 10ern dreimal so stark |

Zwei Befunde, die zählen:
1. **Das Modell würde Sättigung und Kontrast erhöhen** (bester Einzel-Eingriff bei den 8ern: Sättigung 1,44× für
   +0,49, Kontrast 1,4× für +0,41). Die 10-vs-8-Analyse auf deinen Labels zeigt das Gegenteil: Deine 10er sind
   *weniger* gesättigt als die 8er. Hier liegen Modell und Mensch messbar auseinander – das CLIP-Modell hat aus
   dem Gesamtdatensatz (viele Crops, Web-Bilder) „bunt = gut“ gelernt, dein Geschmack am oberen Ende ist das nicht.
2. **Unschärfe und Zoom dominieren** die Modellreaktion. Auch das ist der Crop-Confound in anderer Form: Das Modell
   hat gelernt, dass detailarme, ausschnitthafte Bilder schlechte Noten bekommen.

Konsequenz: Die Eingriffe, die das Modell empfiehlt, sind Aussagen über das Modell, nicht über Michels Geschmack.
Sobald Labels am oberen Ende und Duelle vorliegen (und ein Modell auf `bt_score` oder nur Originalen trainiert
ist), lohnt derselbe Sweep erneut; die Differenz der Kurven zwischen den Modellen zeigt dann, was die neuen Labels
dem Modell beigebracht haben.

### Konzept-Flaschenhals-Modell (27.10.2026, T2)

`models/concept.py`, Configs `concept_bottleneck.yaml` (alle Bilder) und `concept_bottleneck_originals.yaml` (nur
Originale), Seite „Konzeptmodell“, Block „Konzeptmodell“ im Inspect. Ein Lasso auf *benannten* Konzepten (CLIP-Konzepte,
CLIP-Licht, Judge-Flags/-Licht, Farbpalette, Szenenanteil) statt auf 512 anonymen Embedding-Dimensionen; gleiche
gruppierte 5-fach-CV wie alle Tabellenmodelle, dazu Koeffizienten je Standardabweichung, 200 Bootstrap-Refits
(Vorzeichenstabilität, 90-%-Intervall), additive Beiträge pro Bild (`contributions.parquet`), Residuen-Galerie und
ein **Vokabular-Vergleich** (dasselbe Modell mit verschiedenen Konzeptfamilien, out-of-fold).

| Vokabular | ρ OOF alle (n=265) | ρ OOF Originale (n=116) | behalten |
|---|---|---|---|
| nur 16 CLIP-Konzepte | **0,68** | **0,39** | 10 |
| CLIP-Konzepte + Licht | 0,66 | 0,35 | 9 |
| alles (54 Konzepte) | 0,67 | 0,29 | 19 |
| alles + Judge-Score | 0,68 | 0,35 | 37 |
| nur Szene (crop_scale, Himmel-/Wolkenanteil) | 0,57 | 0,22 | 3 |
| nur Farbpalette | 0,55 | 0,12 | 8 |
| nur Judge-Attribute | 0,49 | 0,28 | 19 |

Lesart: 16 Wörter erklären fast so viel wie das 512-dimensionale CLIP-Ridge (OOF 0,68 vs. 0,72 gesamt; 0,41 vs. 0,47
innerhalb der Originale, wenn nur auf Originalen trainiert wird). Palette und Judge-Attribute bringen out-of-fold
nichts zusätzlich; bei nur 116 Originalen überfittet das breite Vokabular sogar (ρ 0,11), deshalb nutzt die
Originale-Config ausschließlich die CLIP-Konzepte.

Das Originale-Modell in einem Satz (Score-Punkte je Standardabweichung, Vorzeichen in ≥ 95 % der Bootstraps stabil,
sofern nicht anders vermerkt): **−0,82 weicher Dunst, −0,72 Flugzeugfenster, −0,58 aufgetürmte Struktur,
+0,52 spektakulärer Himmel**, dann −0,39 Pastell-Sonnenuntergang (84 %), −0,34 dramatischer Sturmhimmel (74 %),
+0,29 Schichtung/Tiefe (83 %), +0,27 hoher Kontrast (88 %). Verworfen: Strahlenbüschel, feine Textur, Lenticularis,
Landschaft dominiert.

Was das für die Leitfrage heißt: Innerhalb der Originale senken *Dunst* und *aufgetürmte Masse* die Note am
stärksten, nicht fehlende Farbe. „Pastell“ als CLIP-Konzept ist negativ, obwohl die 10-vs-8-Analyse geringere
Sättigung bei den 10ern zeigt; vermutlich fängt der Prompt eher flaue, kontrastarme Himmel ein als klare, helle.
Das passt zu „klar, strukturiert, hell“ als Arbeitshypothese und widerspricht erneut dem Gesamtmodell, das
Sättigung belohnt. Ein Modell mit 0,41 Rangkorrelation ist aber kein Beweis; die Residuen-Galerie (unterschätzte
Originale) ist die Liste der Bilder, für die noch ein Wort fehlt – dort lohnen neue CLIP-Prompts.

### Semantische Karte (27.10.2026, T3)

`eval/semantic_map.py`, CLI `semantic-map`, Seite „Karte“. UMAP (Cosinus, Fallback PCA) der gecachten Embeddings
(CLIP ViT-B/32 und DINOv2-S, Projektionen gecacht unter `data/processed/embeddings/projection_*.parquet`), pro Punkt
alle bekannten Signale (eigener Score, Modell-Vorhersage, Judge-Mittel, CLIP-Typ/-Licht, Bildart, gelabelt ja/nein).
Einfärbung frei wählbar, optional Vorschaubilder direkt auf der Karte (Layout-Images, bis 300 Stück), Lasso-/Box-Auswahl
→ Bildkacheln und CSV-Export. **Regionen**: k-Means-Zellen auf der Karte mit Label-Abdeckung, Ø Score, Anzahl ≥ 9,
Ø Judge/Modell, häufigstem CLIP-Typ, typischen Bildern und ungelabelten Originalen als Label-Kandidaten;
`gap_score = (ungelabelt) × (0,5 + Schätzung/10)` sortiert die Lücken nach Nutzen.

Befund auf CLIP (12 Regionen, 939 Bilder, 312 Labels): Keine Region ist völlig ohne Label, aber die Labels sind sehr
ungleich verteilt. **Region 5 ist das Zehner-Nest**: 69 Bilder, 54 davon Originale, 51 gelabelt, Ø 8,1, 22 der Bilder
≥ 9 (CLIP-Typ Cumulonimbus, Tageslicht). Die größte Lücke ist **Region 1** (114 Bilder, nur 30 gelabelt, 42 Originale,
Ø 7,6 und 8 Bilder ≥ 9 unter den gelabelten): dort liegen vermutlich weitere hohe Noten, also der beste Ort für die
nächsten Labels. Regionen 0, 7 und 8 (Altostratus/Nimbostratus, Ø 2–4,5) sind die „graue Suppe“ und fast nur Crops;
dort sind weitere Labels wenig informativ. Die 10er verteilen sich über 10 von 12 Regionen, es gibt also nicht *den*
einen Bildtyp für eine 10, aber eine klare Verdichtung.

Nachtrag: `semantic-map` schreibt zusätzlich `data/processed/embeddings/map_queue.parquet`, eine Label-Reihenfolge
„Kartenlücke“ (ungelabelte Originale, Regionen nach Lückenwert, je vier Bilder pro Region im Rundlauf, innerhalb der
Region die vielversprechendsten zuerst). Die Rate-Seite bietet sie als Label order an und zeigt pro Bild „Warum jetzt:
Kartenlücke R1 …“. Nach neuen Labels `semantic-map` erneut ausführen, damit die Reihenfolge den Labelstand kennt.

### Rater-Analytik (27.10.2026, T5)

`eval/rater_analytics.py`, Seite „Rater-Analytik“. Sitzungen aus Zeitstempeln (Pause > 30 min), Verhaltensfragen auf
Residuen (eigene Note minus pro Bild kalibrierte Referenz: Judge-Mittel oder Out-of-fold-Modell), Drift innerhalb und
zwischen Sitzungen, Anker-Effekt des Vorgängerbilds (Paare gleicher Split-Gruppe oder Aufnahmesitzung ausschließbar),
Skalennutzung, Streitbilder (Mensch vs. Judges, später Mensch vs. Mensch), Rater-Übereinstimmungsmatrix.

Befund für `friend_a` (321 Bewertungen, zwei Sitzungen am 26.4.2026 mit 35 und 60 Minuten, Median 8 s je Bild):
- **Keine Drift**: Spearman(Position, Residuum) = −0,02; Residuenänderung je 50 Bilder ≈ 0; die beiden Sitzungen
  unterscheiden sich im mittleren Residuum um 0,4. Die ersten 20 Bilder einer Sitzung liegen 0,6–0,7 Punkte über dem
  Rest (Aufwärm-Effekt oder Zufall bei n = 40).
- **Deutlicher Anker-Effekt (Assimilation)**: Residuum vs. Vorgänger-Residuum r = +0,28 (Judge-Referenz) bzw. +0,27
  (Modell-Referenz), p < 0,001, 275 Paare nach Ausschluss von 43 Paaren gleicher Split-Gruppe. Nach einem Bild mit
  9+ fällt das nächste im Mittel **≈ 1 Punkt** milder aus als nach einem Bild mit ≤ 4 (+1,06 bzw. +0,92). Ein Teil
  kann inhaltliche Ähnlichkeit aufeinanderfolgender Bilder sein (gleicher Abend), aber die Modellreferenz kennt
  Michels Geschmack und reduziert den Effekt kaum. Konsequenz: Label-Reihenfolge „Random“/„Diverse groups“, Duelle für
  feine Unterschiede, und bei der Rauschschätzung ±0,5 Punkte Anker-Anteil einrechnen.
- **Skala**: 41 % bzw. 55 % halbe Noten in den beiden Sitzungen, Modus 8,5 (39×), 10 nur 9×, volle Spanne 0–10 genutzt.

### Wolkenatlas (27.10.2026, T4)

`eval/cloud_atlas.py`, Seite „Wolkenatlas“. Wolkentyp pro Bild als Mehrheitsvotum aus CLIP-Zero-Shot und beiden Judges
(Primärtyp 1 Stimme, Nebentyp ½), Zuverlässigkeit „sicher“ (alle drei Primärtypen einig, 255 Bilder), „wahrscheinlich“
(666) und „unsicher“ (18). Die Quellen sind sich nur mäßig einig (CLIP vs. Judge 0,35–0,39, Judge vs. Judge 0,63, alle
drei 0,27), deshalb das Votum. Pro Typ: Bestand, Labelabdeckung, Ø Note, Anteil ≥ 9, häufigstes Licht, beste Beispiele
und ein kurzer meteorologischer Text (Stockwerk, Höhe, Entstehung, Wetterbedeutung, Aussehen) für 15 Typen inklusive
Lenticularis, Mammatus, Kondensstreifen, Nebel und wolkenlos. Dazu ein Typ × Licht-Raster der mittleren Note.

Befund (Originale, mindestens „wahrscheinlich“; bei Stimmengleichstand zählt der CLIP-Typ): Die höchsten Mittelwerte
haben Mammatus (9,0; n = 2), Lenticularis (8,4; n = 5, 3 × ≥ 9), **Cirrus (8,4; n = 19, 10 × ≥ 9)** und
**Altocumulus (8,2; n = 24, 11 × ≥ 9)**; Cumulus 7,7 (n = 44, 12 × ≥ 9), Cumulonimbus 7,3, Stratocumulus 7,2 (n = 32),
Altostratus 4,8, wolkenlos 3,0 (n = 1). Typ × Licht: Altocumulus in der Dämmerung 9,2, Cirrus am Tag 8,8, Cumulus am
Tag 7,8, Cumulonimbus am Tag 6,1. Das passt zur Konzept-Lesart „fein, geschichtet, hell“: hohe und mittelhohe
Strukturwolken (Cirrus, Altocumulus, Linsen) liegen über den massigen Haufen- und Gewitterwolken. Die Zahlen sind
klein und das Typ-Votum ist unscharf (nur 255 von 939 Bildern „sicher“); der Atlas ist Hypothesengenerator und
Lesehilfe, kein Beweis.

### Bericht „Wolken des Jahres“ (27.10.2026, T6)

`reports/year_report.py`, CLI `year-report`, Seite „Bericht“. Ein eigenständiges HTML (ca. 0,8 MB bei 12 Karten, Bilder
als eingebettete JPEGs mit 720 px) aus allem, was das Projekt weiß: Kennzahlen und Notenverteilung, die besten Fotos mit
Erklärkarten (Wolkentyp-Votum mit Zuverlässigkeit, CLIP-Lichtphase, auffällige CLIP-Konzepte in σ, Konzeptmodell-
Beiträge, Judge-Urteile), das Konzeptmodell in einem Satz mit Koeffizientenbalken, die 10-gegen-8-Tabelle, der
Wolkenatlas, die kontrafaktischen Band-Sweeps, die Modell-Rangliste und die Test-Retest-Verlässlichkeit. Fehlende Quellen
lassen ihren Abschnitt aus. Filter: Rater, Anzahl Karten, Aufnahmejahr (über `capture_date`), nur Originale. PDF über
den Druckdialog des Browsers (Druck-CSS ist enthalten). Ausgabe unter `data/artifacts/reports/` (git-ignoriert, enthält
private Bilder).

### UI-Konsolidierung (27.10.2026, T8)

`app/Home.py` ist jetzt nur noch der Einstieg: `st.navigation` bündelt die 19 Seiten in drei Arbeitsbereiche
**Daten & Labeln** (Start & Status, Import, Bewerten, Triage, Ausschlüsse, Quellgruppen, Galerie), **Trainieren &
Vergleichen** (Train & Leaderboard, Runs vergleichen, Batch Report, Rater-Vergleich, Rater-Analytik) und **Verstehen**
(Inspect, 10 vs 8, Was wäre wenn, Konzeptmodell, Karte, Wolkenatlas, Bericht). Die URL-Pfade sind unverändert
(`/Rate`, `/Karte`, …), die alte Home-Seite heißt „Start & Status“ (`pages/00_Start.py`). Gemeinsame Sidebar auf jeder
Seite: Rater-Auswahl (`st.session_state["rater_id"]`, von Bewerten, Inspect, Galerie, Quellgruppen, Rater-Analytik,
Bericht und Start als Vorgabe übernommen) und ein Pipeline-Ampelhinweis (120 s gecacht). `safe_read_table` cached
Tabellen nach Pfad, Änderungszeit und Größe, damit Manifest und Labels nicht auf jeder Seite neu gelesen werden.

### Freunde-Links: Labeln ohne Installation (27.10.2026)

`friends/` (Registry, Warteschlange, FastAPI-Server, Handy-Seite), CLI `friends-add` / `friends-list` / `friends-revoke` /
`friends-serve` / `friends-tunnel`, Seite „Freunde“. Jeder Freund bekommt einen Link `…/f/<token>`; der Token ist die
Rater-ID (`friend_<name>`, in `raters.csv` als human registriert, Registry `data/raw/metadata/friends.csv`, git-ignoriert).
Bewertungen und Duelle landen sofort über die bestehenden Append-Funktionen in den Rater-Dateien, Undo per Tombstone.
Reihenfolge pro Freund: erst die Bilder, die `friend_a` schon bewertet hat (Vergleichsset für Übereinstimmung), dann
Karten-Lücken und Triage-Reihenfolge, dann der Rest; Originale und große Ausschnitte, kleine Ausschnitte nicht; jeder
Freund mit eigenem Zufalls-Seed. Bilder werden auf 1280 px verkleinert und gecacht (`data/processed/friend_cache/`).

Sicherheit: Server bindet nur an localhost, alles ohne gültigen aktiven Token ist 404, Bilder nur über Manifest-IDs
(keine Pfade aus der URL), einzige Schreibzugriffe sind Append-Zeilen und Tombstones des eigenen Raters, Rate-Limit pro
Token, keine Doku-/Admin-Endpunkte, Security-Header. Zugang von außen über Cloudflare Quick Tunnel (`friends-tunnel`,
keine Portfreigabe, wechselnde Adresse). Ausbaustufe mit Cloudflare-Konto und Domain: fester Hostname plus Cloudflare
Access (E-Mail-Einmalcode) vor dem Tunnel; langfristig als Docker-Container auf dem NAS.

### CLIP-Zero-Shot (Wolkentyp, Licht, Konzepte)

Textprompt-Ensembles im CLIP-Raum ersetzen den regelbasierten Klassifikator als primäre Schätzung.
Verteilung über alle 939 Bilder: Cumulus 229, Altocumulus 128, klarer Himmel 121, Altostratus 109,
Stratocumulus 72, Nimbostratus 70, Cirrus 66, Cirrostratus 58, Cumulonimbus 46, Stratus 25,
Cirrocumulus 15 – deutlich plausibler als „70 % Cumulonimbus“. Mittleres Rating nach CLIP-Typ (gelabelt):
Altocumulus 7,4, Cumulonimbus 7,3, Cirrus 7,3, Cumulus 6,8, Cirrostratus 6,4, Nimbostratus 4,8,
Altostratus 3,1. Nach Licht: goldene Stunde 7,2 (n=27), Tageslicht 6,0 (n=205).

Konzept-Korrelationen mit dem Rating (Spearman, alle 312 / nur Originale):
`boring_sky` −0,44 / −0,14 · `spectacular_sky` +0,41 / +0,16 · `airplane_window` −0,40 / −0,17 ·
`soft_haze` −0,37 / −0,13. Die Zero-Shot-Konzepte tragen also vor allem die Grob-Trennung; für die
Feinunterscheidung unter guten Bildern sind sie (noch) schwach – ein Hinweis, dass die 10-vs-8-Frage
eher über Nachbarschaft im Embedding-Raum und gezielte Prompts/Labels am oberen Ende zu knacken ist.
Caveat bleibt: Zero-Shot ist eine Bild-Text-Ähnlichkeit, keine meteorologische Beobachtung.

Einordnung der alten Runs:
- Baumbasierte Baselines schlagen das Deep-Modell. Das eingefrorene ResNet18 mit linearem Kopf ist
  faktisch eine lineare Sonde auf ImageNet-Features, mit Batchgröße 4 trainiert – kein Wunder.
- Die Konformal-Intervalle (90 %) haben Halbbreiten von 3,0–3,6 Punkten, sind also praktisch
  nicht informativ. Ursache: nur eine Val-Fold mit 21 Bildern, hohes Label-Rauschen.
- Metriken beruhen auf einem einzelnen Fold statt 5-fach-CV; die Streuung zwischen Folds ist unbekannt.
- Grad-CAM funktioniert technisch (4 Layer wählbar), liefert bei einem eingefrorenen Backbone aber
  nur ImageNet-Aufmerksamkeit, nicht Rating-spezifische Evidenz.

## 6. Qualität der Erklärungsschicht (ehrliche Einschätzung)

- **Wolkentyp-Klassifikator**: 218 von 314 Originalen werden als *cumulonimbus* klassifiziert –
  das regelbasierte System ist stark verzerrt und aktuell nicht verwertbar. Das Feld
  `vision_model_vote` steht überall auf `not_configured`.
- **Konzepte** sind deterministische Funktionen der übrigen Features; ihre „Erklärkraft“ ist zirkulär.
- **Lokale Feature-Beiträge** sind Ein-Feature-Sensitivitäten um den Median, keine additive
  Zerlegung (SHAP wäre installiert, wird aber nicht genutzt).
- **Interpretationskarte** ist sauber strukturiert und ehrlich beschriftet (Cautions, Caveats) –
  gutes Gerüst, das mit besseren Signalen sofort wertvoller wird.
- Der **Ollama-Textlayer** ist optional und korrekt als Paraphrase (nicht als Quelle) implementiert.

## 7. Gefundene Probleme / technische Schulden

Erledigt in P1 (24.10.2026):

1. ~~Verarbeitete Label-Tabellen leer~~ → `aggregate-labels` neu gelaufen: 312 Labels, 265 Dev / 47 Test.
2. ~~Unkommittierte Mai-Arbeit~~ → committed.
3. ~~QWK clippt auf 1–10~~ → Skala 0–10.
4. ~~Kein Pipeline-Zustand sichtbar~~ → `cloud_aesthetics/pipeline.py`, Home-Panel, CLI `status` / `refresh`.
5. ~~Metriken nur auf einer Fold~~ → Baseline nutzt gruppierte 5-fach-CV (OOF-Metriken, Fold-Streuung,
   Metriken nach `kind`/`dataset_name`, `predictions.parquet` pro Run).
6. Neu gefunden und behoben: 3 byte-identische Bilder lagen unter zwei Pfaden und standen doppelt im
   Manifest (gleiche `image_id`), was Labels und Splits duplizierte. Manifest dedupliziert jetzt nach `image_id`.

Offen:

7. ~~`hybrid`-Modell ohne Bild-Embedding~~ → P3: echte CLIP/DINOv2-Embeddings (`embedding_*.yaml`); das alte
   `hybrid.yaml` (PCA der Features) ist damit obsolet und kann entfernt werden.
8. **Deep-Runs** werden weiterhin nur auf einer Fold evaluiert (CPU-Zeit).
9. **pyproject** nennt als Autor „OpenAI Codex“; Lizenz MIT. Ggf. anpassen.
10. Streamlit-Seiten lesen Manifest und Ratings-CSV bei jedem Rerun neu –
    bei ~1.000 Bildern noch okay, skaliert aber nicht.
11. Notebooks sind nur 2-Zellen-Platzhalter.

## 8. Was gut ist und bleiben sollte

Datenmodell (Rater-CSV, Manifest, Gruppen-Splits, Exclusions), die Import-/Crop-Pipeline, die
CLI-Struktur mit `*_impl`-Funktionen (UI und CLI teilen Code), die Interpretationskarte als
deterministisches Rückgrat, das Friend-Labeling (Web + Android), die Testbasis.
