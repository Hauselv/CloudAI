# cloud-aesthetics – Hinweise für Claude Code

Lokales Forschungs-Workbench-Projekt: Wolken-/Sonnenuntergangsfotos bewerten (0–10), Rating
vorhersagen und vor allem **erklären**, welche Bildeigenschaften das Rating treiben.
Überblick und Ist-Zustand: `docs/PROJEKTSTATUS.md`. Priorisierte Vorhaben: `docs/ROADMAP.md`.

## Umgebung
- Python: `D:\CloudAI\.venv\Scripts\python.exe` (3.12). Kein `Activate.ps1` nötig.
- Torch ist CPU-only (AMD RX 6700 XT, kein CUDA). Deep-Training ist langsam; Embedding-Ansätze bevorzugen.
- Tests: `D:\CloudAI\.venv\Scripts\python.exe -m pytest -q` (ca. 16 s, alle grün).
- UI: `D:\CloudAI\.venv\Scripts\python.exe -m streamlit run src\cloud_aesthetics\app\Home.py`
- CLI: `D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli <befehl>`

## Konventionen
- Logik gehört in `*_impl`-Funktionen in `cli.py` oder in die Fachmodule; Streamlit-Seiten rufen diese nur auf.
- `app/Home.py` registriert alle Seiten per `st.navigation` (drei Arbeitsbereiche, feste `url_path`s); neue Seiten dort
  eintragen, der Ordner `pages/` wird nicht mehr automatisch gelesen. Rater kommt aus der gemeinsamen Sidebar (`current_rater()`).
- Rohlabels (`data/raw/metadata/ratings/*.csv`) sind append-only und die einzige Wahrheit; nie umschreiben.
- Jeder Labelgeber ist ein eigener `rater_id`. LLM-/Modell-Labels bekommen eigene rater_ids und werden
  nie unmarkiert mit menschlichen Labels gemischt.
- Originale und ihre Crops teilen `split_group_id`; Splits immer gruppiert ziehen.
- `data/raw`, `data/processed`, `data/artifacts` enthalten private Bilder und Ableitungen: nicht committen
  (bis auf die in `.gitignore` freigegebenen `web_sample`-Dateien).
- Code und Code-Kommentare auf Englisch, Doku für Michel auf Deutsch.

## Embeddings
- `extract-embeddings` (CLIP ViT-B/32, DINOv2-S; Cache in `data/processed/embeddings/`) und `zero-shot` laufen in ~2 Minuten
  auf CPU; Embedding-Modelle (`configs/models/embedding_*.yaml`) trainieren in Sekunden. Bevorzugter Weg vor Deep-Training.
- Streamlit cached importierte Paketmodule: nach Änderungen an `pipeline.py` o. ä. den Server neu starten.

## LLM-Judge
- `judge-images` hat zwei Backends (`configs/app/default.yaml`, Block `judge`): `gateway` = Uni-LLM-Server
  (OpenAI-kompatibel, Key aus `CLOUD_JUDGE_GATEWAY_API_KEY` oder der Secrets-Datei des Spider-Projekts, nie im Repo)
  und `anthropic` (SDK `client.messages.parse`, `ANTHROPIC_API_KEY`). Urteile werden gecacht (`data/processed/judge/`),
  Scores landen als eigener Rater `llm_<modell>` in den Ratings-CSVs; `raters.csv` markiert sie als `llm`, die
  Aggregation ignoriert sie (`include_llm_raters: false`). Key-Werte nie ausgeben, auch nicht in Logs.
- Das Uni-Gateway limitiert auf ca. 100 Anfragen pro Fenster; `GatewayJudge` wartet bis zum angegebenen Reset.
  Für ~300 Bilder mit zwei Modellen rund eine Stunde einplanen. Judge-Cache (`data/processed/judge/`) ist git-ignoriert.

## Karte und Konzeptmodell
- `semantic-map` cached UMAP-Projektionen unter `data/processed/embeddings/projection_*.parquet` und schreibt die
  Label-Reihenfolge „Kartenlücke“ (`map_queue.parquet`); nach neuen Labels erneut ausführen.
- Konzeptmodelle (`configs/models/concept_bottleneck*.yaml`) sind Lasso-Läufe mit `kind: concept`; Artefakte
  `concepts.parquet`, `contributions.parquet`, `vocabulary.parquet` liegen im Run-Ordner.

## Freunde-Server
- `friends-serve` (FastAPI, localhost:8765) + `friends-tunnel` (Cloudflare Quick Tunnel). Tokens in `data/raw/metadata/friends.csv`,
  nie ausgeben oder committen. Server nie an 0.0.0.0 binden; Zugriffsschutz ist der Token (später Cloudflare Access).

## Bekannte Stolpersteine
- `data/processed/aggregated_labels.parquet` kann leer sein, obwohl Rohlabels existieren → `aggregate-labels` ausführen.
- Nach neuem Labeln immer: `aggregate-labels` → `extract-features` → `train`, sonst passen Splits und Runs nicht zusammen.
- Crops sind systematisch schlechter bewertet als Originale (Confound); Auswertungen nach `kind` und `crop_class`
  trennen. Ursache ist der Szenenanteil (`crop_scale`), nicht die Pixelgröße. Für die 10-vs-8-Frage zählen Originale.
