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
