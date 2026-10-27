# Roadmap – Vorschlag nach Bestandsaufnahme (24.10.2026)

Leitfrage des Projekts: **Wovon hängt das Rating ab, und was trennt eine 10 von einer 8?**
Alles unten ist danach priorisiert, wie direkt es diese Frage voranbringt. Details zum Ist-Zustand
stehen in `PROJEKTSTATUS.md`.

Legende: **P** = primär (zuerst, Fundament), **S** = sekundär, **T** = tertiär / Kür.
Aufwand grob: S = Stunden, M = 1–2 Tage, L = mehrere Tage.

---

## P – Primär: Fundament reparieren, Datenbasis ehrlich machen

### P1 · Pipeline-Hygiene und Zustand sichtbar machen (Aufwand S) — erledigt 24.10.2026
- `aggregate-labels` laufen lassen, damit die leeren Label-Tabellen wieder gefüllt sind; danach
  Features/Runs neu erzeugen, damit alles zu einem Labelstand passt.
- Unkommittierte Mai-Arbeit committen (Tests sind grün).
- **Pipeline-Status-Panel** auf der Home-Seite: Zeitstempel und Zeilenzahlen von Rohlabels →
  Aggregat → Features → Splits → letzter Run, mit „veraltet“-Markierung, plus ein Knopf
  „Pipeline aktualisieren“ (aggregate → features → optional train).
- `kind` (original/crop) und `dataset` (privat/web) als Spalten in Labels und Features mitführen.
- QWK auf 0–10 korrigieren; Metriken als 5-fach-CV mit Mittelwert ± Std statt einer Fold.

### P2 · Crop-Confound auflösen (Aufwand S–M, aber entscheidend) — erledigt 24.10.2026
Umgesetzt: `crop_scale`/`crop_class` (original · large_crop · small_crop) in Manifest, Labels, Features und
allen Modell-Metriken; Quellbild mit Ausschnittsrahmen in Rate und Inspect; Bildart-Filter beim Labeln;
Import-Defaults auf Szenen-Crops (50/70/90 % der kurzen Kante, min. 768 px, nie vergrößern);
`include_kinds`-Filter in Modell-Configs und ein Originale-only-Run. Kernbefund: Crops wurden nicht wegen
Unschärfe schlechter bewertet, sondern weil 527 von 710 nur 35 % der kurzen Kante zeigen (Details ohne
Komposition). Details in `PROJEKTSTATUS.md`.
Der stärkste Prädiktor ist derzeit „Crop oder Original“ (ρ = −0,51). Optionen, die sich nicht
ausschließen:
1. **Metriken und Erklärungen getrennt nach `kind`** ausweisen (Minimum).
2. **Primäranalyse nur auf Originalen**; Crops bleiben für Augmentation/Robustheit, werden aber nicht
   als eigenständige Labels in die 10-vs-8-Analyse gemischt.
3. Crops **ohne Hochskalieren** mit Mindestgröße (z. B. ≥ 768 px) erzeugen, damit Schärfe kein
   Artefakt mehr ist; bestehende kleine Crops als `derivative_kind = small_crop` kennzeichnen.
4. In der Rate-Seite bei Crops das **Original mit markiertem Ausschnitt** daneben zeigen, damit das
   Rating bewusst „Ausschnitt-Ästhetik“ bewertet.

### P3 · Bild-Embeddings als Rückgrat (Aufwand M) — erledigt 24.10.2026
Umgesetzt: `cloud_aesthetics/embeddings/` (CLIP ViT-B/32 + DINOv2-S, Cache, Zero-Shot), `models/embedding.py`
(Ridge/kNN/SVR, gleiche CV), Leaderboard auf der Train-Seite, Embedding-Nachbarn und CLIP-Attribute im Inspect.
Ergebnis: CLIP-Ridge hebt die OOF-Rangkorrelation von 0,64 auf 0,72 und innerhalb der Originale von 0,37 auf 0,47
(Test-Originale 0,41 → 0,68). Details in `PROJEKTSTATUS.md`. Noch offen aus P3: GP-Unsicherheit, UMAP-Karte (T3).

Statt ResNet18 auf CPU zu trainieren: einmalig **CLIP (ViT-B/32 oder ViT-L/14)** und/oder
**DINOv2-small** Embeddings für alle Bilder berechnen und cachen (`data/processed/embeddings/`).
Auf CPU dauert das für ~1.000 Bilder Minuten, nicht Stunden. Darauf:
- Ridge / kNN / GP-Regression → in Sekunden trainierbar, 5-fach-CV trivial, saubere
  Unsicherheiten (GP oder kNN-Streuung) statt der aktuellen 3-Punkte-Intervalle.
- **Zero-Shot-Wolkentypen** per CLIP-Textprompts („a photo of cumulus clouds“ …) ersetzen den
  regelbasierten Klassifikator, der aktuell 70 % als Cumulonimbus einstuft.
- **Echte Konzept-Scores** per CLIP-Prompts („dramatic storm sky“, „golden hour rim light“,
  „crepuscular rays“, „pastel sunset“ …) statt handgewichteter Sigmoiden über andere Features.
- Das `hybrid`-Modell wird damit zum echten Hybrid: Features + Embedding.
- Nachbarschaft/Prototypen („ähnlichste 10er“) werden semantisch statt feature-basiert.

### P4 · LLM-Judge als zusätzlicher Rater (Aufwand M) — erledigt 24.10.2026
Umgesetzt: `cloud_aesthetics/judge/` (Rubrik-Prompt, strukturiertes JSON-Urteil, Cache, Rater-Registry,
Vergleichsreport), Backends Uni-Gateway (DeepSeek V4.1 Flash, Qwen3.6 35B) und Anthropic, CLI `judge-images` /
`judge-report`, Seite „Rater Comparison“. Validiert gegen alle 312 Labels: ρ ≈ 0,70 gesamt, aber nur ≈ 0,35–0,40
innerhalb der Originale und 0,28 am oberen Ende; die Judges sind untereinander mit ρ 0,90 einig, der Unterschied
zu Michel ist also Geschmack. Brauchbar als Vorfilter und Attribut-Layer, nicht als Ersatz für die feinen Labels.
Details und Geschmacks-Hypothesen in `PROJEKTSTATUS.md`. Schritt 2/3 ebenfalls umgesetzt: `judge-triage` und die
Seite „Triage“ (Label-Warteschlange, unsichere Bilder, Exclusion-Vorschläge mit Bestätigung), Label-Reihenfolge
„Judge-Vorschlag“ in der Rate-Seite.
Michels Idee, mit einem Vision-Modell schnell einen großen Mock-Datensatz zu labeln, passt sauber
ins bestehende Schema: **jedes Modell ist ein eigener `rater_id`** (z. B. `llm_claude_sonnet_5_5`),
mit zusätzlicher Spalte `rater_kind = human | llm` in den Rohlabels, damit Mensch und Modell nie
unbemerkt gemischt werden.
- Schritt 1 (Validierung zuerst): die 312 bereits von Michel gelabelten Bilder vom LLM bewerten lassen
  und Spearman/MAE gegen Michel messen. Erst wenn die Übereinstimmung brauchbar ist (Ziel ρ ≥ 0,6),
  den Rest labeln.
- Schritt 2: strukturierte Ausgabe pro Bild als JSON: Score, Konfidenz, Wolkentyp(en), Tageszeit/Licht,
  dominante Farben, 3 Begründungssätze, Qualitätsflags (Flugzeugfenster, Satellit, Landschaftsanteil).
  Das liefert gleichzeitig den **semantischen Feature-Layer** (siehe S3) und Vorschläge für Exclusions.
- Schritt 3: UI-Seite „Rater-Vergleich“: Mensch vs. LLM, Streudiagramm, größte Abweichungen als Kacheln
  (dort lernt man am meisten über das eigene Geschmacksmodell).
- Kosten vorab mit aktueller Preisliste rechnen; Bilder auf ~1024 px verkleinern; Ergebnisse cachen.
- Wichtig für die Feinanalyse: LLM-Labels dienen für Vortraining/Skalierung; Michels und Freundes-
  Labels bleiben die Referenz für 10-vs-8.

### P5 · „10 vs 8“-Analyseseite (Aufwand S–M) — erledigt 24.10.2026
Umgesetzt: `eval/contrast.py`, CLI `contrast`, Seite „10 vs 8“ (Bänder, Bildart-/Datensatzfilter, Merkmalsfamilien,
Effekt-Balken mit Cohen's d und FDR-q, Merkmal-Explorer mit Strip-/Scatter-Plot und Beispielbildern, Trend am
oberen Ende, Beispielkacheln). Befund: Bei 34 vs. 51 Originalen überlebt kein Signal die FDR-Korrektur; die größten
Effekte (weniger Judge-Flags, weniger Cumulus, mehr Lenticular/Cirrus, weniger Wolkenmasse in der Mitte, geringere
Sättigung) stützen die Hypothese „luftig-hell statt bunt-dramatisch“. Mehr Labels am oberen Ende sind der Hebel.
`eval/analysis.compare_groups` existiert bereits (Welch-t, Cohen's d), wird aber nirgends in der UI genutzt.
- Zwei frei wählbare Score-Bänder (z. B. 9,5–10 vs. 7,5–8,5), optional nur Originale, optional ein Rater.
- Tabelle der Features/Konzepte/Embedding-Konzepte nach Effektstärke, mit Verteilungs-Violinplots.
- Darunter die Beispielkacheln beider Gruppen nebeneinander – das ist der Moment, in dem man
  Hypothesen bekommt („die 10er haben fast alle Gegenlicht + Mehrschichtigkeit“).
- Feature-vs-Score-Streudiagramm mit Thumbnail-Hover (Plotly), damit man Ausreißer anklicken kann.

---

## S – Sekundär: Labeling beschleunigen, Modelle und Unsicherheit verbessern

### S1 · Labeling-UX (Aufwand S–M) — erledigt 24.10.2026
Umgesetzt: Score-Buttons 0–10 mit Zifferntasten (`+` = 10, `.` = ±½), Schnellmodus (Ziffer speichert sofort),
Undo per Tombstone-Datei (`ratings_void.csv`, Rohdateien bleiben append-only), bisherige Bewertungen pro Bild,
Judge-Scores nur auf Wunsch sichtbar (Anker-Vermeidung), Duell-Modus am oberen Ende (Pool = eigene Bilder ab
Schwelle, optional Triage-Kandidaten; Paare mit wenigen Duellen und ähnlichem Score zuerst; Hotkeys A/L/T/N) mit
Bradley-Terry-Rangfolge; `aggregate-labels` schreibt `bt_score` und `duels`. Nachtrag 25.10.2026: aktive Auswahl
(`predict-all` + Warteschlange nach Informationsgewinn: oberes Ende, Modell-/Judge-Uneinigkeit) und blinde
Test-Retest-Einstreuung mit Zuverlässigkeitsanzeige auf Home sind ebenfalls umgesetzt.
- Zifferntasten 0–9 (+ `.` für ,5) setzen den Score direkt; Enter speichert. Vorherige Bewertung
  des aktuellen Raters einblenden; Undo für die letzte Bewertung.
- Bei Crops das Quellbild mit Rahmen zeigen (siehe P2.4).
- **Aktive Auswahl**: „Als nächstes labeln“ = Bilder, die das Modell ≥ 7,5 schätzt oder bei denen
  Modell-Unsicherheit hoch ist. Für die 10-vs-8-Frage braucht es viele Labels am oberen Ende.
- **Duelle am oberen Ende**: Paarvergleiche nur zwischen Kandidaten ≥ 8, ausgewertet per
  Bradley-Terry/Elo. Feine Unterschiede (10 vs 8,5) sind im Paarvergleich deutlich zuverlässiger
  als in absoluten 0,5er-Schritten. Die Android-App hat „Duel mode“ bereits.
- Test-Retest einbauen: 5 % der Bilder zufällig erneut vorlegen → persönliche Rauschgrenze
  (kein Modell kann konsistenter sein als der Rater selbst).

### S2 · Modell- und Evaluations-Infrastruktur (Aufwand M) — Kern erledigt 24.10.2026
Umgesetzt: SHAP (TreeExplainer/LinearExplainer) für lokale Beiträge im Explain und globale Wichtigkeit pro
Baseline-Run (Train-Seite), Kalibrierungsdiagnostik (Ist-vs-Vorhersage mit Bin-Mitteln, Residuen, Streuungs-
verhältnis, ECE, Intervall-Abdeckung), eigenständige Ranking-Suite auf Paaren (explizite Duelle + Pseudo-Paare,
Dev/Test-Trennung, Bildscores, Vergleich mit mean_score und bt_score; Configs `ranking.yaml`, `ranking_clip.yaml`),
BT-Ziel vorbereitet (`embedding_clip_bt.yaml`, startet erst ab 40 Bildern mit Duellen). 5-fach-CV und Leaderboard
waren schon in P1/P3. Offen: Deep-Fine-Tuning bei mehr Labels und GPU.
- 5-fach-CV als Standard; Run-Vergleichsseite (Leaderboard über alle Runs, nach `kind`, Rater, Datensatz).
- Unsicherheit: Konformal über alle Folds poolen oder GP/Quantil-Regression; Kalibrierungsplot.
- Mehrere Zielgrößen: `mean_score`, Rang (Spearman-Ziel), Top-Band-Klassifikation (≥ 9 ja/nein).
- SHAP für die Baummodelle statt Ein-Feature-Sensitivität (Bibliothek ist installiert).
- Deep-Pfad erst wieder anfassen, wenn Labels ≥ 500 und ein GPU-Weg steht; dann Fine-Tuning statt
  eingefrorenem Backbone, ggf. ConvNeXt-Tiny/EfficientNet.

### S3 · Wissenschaftlicher Layer, der den Namen verdient (Aufwand M–L) — Kern erledigt 24.10.2026
Umgesetzt: Aufnahmekontext aus EXIF/Dateinamen (`features/capture.py`), Sonnenstand und Lichtphase ohne
Fremdbibliothek (`features/solar.py`), Open-Meteo-Reanalyse mit Cache (`features/weather.py`), Kontexttabelle mit
Vererbung auf Crops (`features/context.py`, CLI `extract-context`), 13 Farbpaletten-Features in `features_v1`
(`features/palette.py`); alles in der 10-vs-8-Analyse als Familien „Farbpalette“, „Sonnenstand / Zeit“, „Wetter“
und im Inspect als Kontextblock. Grenze: Nur 14 gelabelte Originale haben EXIF-Zeit, weil die gelabelten Originale
überwiegend Web-Bilder sind; der Layer wird erst mit Labels für die privaten Originale aussagekräftig. Offen aus S3:
bessere Himmelssegmentierung (SegFormer/CLIPSeg).
- Wolkentyp: CLIP-Zero-Shot (P3) oder LLM-Judge-Attribute (P4) statt Regeln; Konfidenz ehrlich ausweisen.
- **Sonnenstand aus Zeit und Ort**: EXIF-Datum/GPS (bzw. Datum aus dem Dateinamen bei WhatsApp-Bildern)
  → Sonnenhöhe/Azimut (`astral`/`pysolar`) → Phasen „Tag / goldene Stunde / bürgerliche Dämmerung /
  blaue Stunde“. Hypothese prüfbar: Hängt die 10 am Licht oder an der Wolke?
- **Wetter-Reanalyse**: Open-Meteo-Historien-API (kostenlos) liefert pro Zeitpunkt Bewölkung nach
  Schicht (niedrig/mittel/hoch), Feuchte, Druck, Niederschlag. Damit lassen sich Ratings erstmals mit
  echter Meteorologie verknüpfen statt mit Bildheuristiken.
- Bessere Himmelssegmentierung: SegFormer (ADE20K hat die Klasse „sky“) oder CLIPSeg statt HSV-Schwellen;
  verbessert Crops, Masken-Features und Regionenanalyse zugleich.
- Farbpaletten-Features: k-Means in Lab → 5 dominante Farben, Palettenkontrast, Warm-Kalt-Verhältnis,
  Farbharmonie-Maße (komplementär/analog). Für Sonnenuntergänge vermutlich aussagekräftiger als Hue-Bins.

### S4 · Datenbeschaffung (Aufwand S)
- `G:\Wolken` enthält nur 90 Fotos, alle bereits importiert. Die „vielen weiteren Wolkenbilder“ müssen
  noch gefunden/importiert werden (Handy, Backup-Ordner, Freunde). Import-Workflow existiert.
- Freundes-Labels einsammeln (APK/Web-Paket existieren) → mehrere Rater → Agreement-Analyse,
  Krippendorffs Alpha, „persönliches Geschmacksmodell vs. Konsens“.

---

## T – Tertiär: Kür und kreative Erweiterungen

### T0 · Modelltreue Heatmaps — erledigt 27.10.2026
Occlusion- und Token-Grad-CAM-Karten für CLIP-Embedding-Runs (`explain/clip_saliency.py`); das ResNet-Grad-CAM des
Deep-Runs ist als Diagnostik eingestuft (zeigt ImageNet-Objekte, nicht Himmelsstruktur).

### T1 · Kontrafaktischer Explorer („Was macht aus der 8 eine 10?“) — erledigt 27.10.2026
`explain/counterfactual.py`, Seite „Was wäre wenn“, CLI `counterfactual-sweep`. Befund siehe `PROJEKTSTATUS.md`:
Das CLIP-Modell würde Sättigung und Kontrast erhöhen, was der Beobachtung aus der 10-vs-8-Analyse widerspricht,
dass Michels 10er *weniger* gesättigt sind als die 8er – ein konkreter Hinweis, wo Modell und Mensch auseinanderliegen.
Bild kontrolliert verändern (Kontrast ±, Sättigung ±, Wärme ±, Beschnitt, Himmelanteil) und die
Modellvorhersage live beobachten. Zeigt Sensitivitäten pro Bild und ist die direkteste Antwort auf
die Leitfrage – vorausgesetzt, das Modell ist gut genug (nach P3).

### T2 · Konzept-Flaschenhals-Modell — erledigt 27.10.2026
`models/concept.py`, Seite „Konzeptmodell“, Configs `concept_bottleneck*.yaml`. Befund in `PROJEKTSTATUS.md`: 16 CLIP-Konzepte
erklären out-of-fold fast so viel wie das volle CLIP-Embedding (0,68 vs. 0,72; Originale 0,41 vs. 0,47); innerhalb der
Originale lautet das Modell „−0,82 weicher Dunst, −0,72 Flugzeugfenster, −0,58 aufgetürmte Struktur, +0,52 spektakulärer
Himmel“. Offen: neue Konzept-Prompts aus der Residuen-Galerie ableiten, Modell auf `bt_score`, sobald genug Duelle da sind.
Rating = lineare Funktion aus ~15 benannten Konzepten (aus CLIP oder LLM-Attributen). Weniger genau als
ein Black-Box-Modell, aber vollständig lesbar: „+1,2 für Gegenlicht, +0,8 für Mehrschichtigkeit,
−0,6 für Landschaftsanteil“.

### T3 · Semantische Karte — erledigt 27.10.2026
`eval/semantic_map.py`, CLI `semantic-map`, Seite „Karte“ (UMAP/PCA, Einfärbung, Thumbnails, Lasso-Auswahl, Regionen mit
Lücken-Score und Label-Kandidaten). Befund in `PROJEKTSTATUS.md`: Region 5 ist das Zehner-Nest (22 von 69 Bildern ≥ 9),
Region 1 die größte Lücke (114 Bilder, 30 Labels, Ø 7,6). Offen: Lücken-Kandidaten direkt als Label-Reihenfolge in Rate.
UMAP der Embeddings mit Thumbnails, eingefärbt nach Rating, Rater, Wolkentyp, Tageszeit. Cluster von
10ern werden sichtbar; Lücken zeigen, welche Bildtypen noch gelabelt werden sollten.

### T4 · Wolkenatlas
Galerie gruppiert nach (zuverlässigem) Wolkentyp mit mittlerem Rating, bestem Beispiel und Kurztext zur
Meteorologie des Typs (Entstehung, Höhe, Wetterbedeutung). Verbindet Ästhetik und Wissenschaft sichtbar.

### T5 · Rater-Analytik
Drift über Zeit (wird man im Laufe einer Session strenger?), Reihenfolge-Effekte (Anker durch das
vorherige Bild), Unterschiede zwischen Freunden, „Streitbilder“ mit höchster Uneinigkeit.

### T6 · Berichte und Teilen
Export eines Batch-Reports als HTML/PDF („Wolken des Jahres“), inklusive Erklärkarten.

### T7 · GPU-Weg
ROCm unter WSL für die RX 6700 XT ist experimentell; DirectML wäre die Windows-Alternative. Erst nötig,
wenn echtes Fine-Tuning ansteht (S2). Bis dahin trägt der Embedding-Ansatz auf CPU.

### T8 · UI-Konsolidierung
Streamlit reicht für lokal. Wenn die Seitenzahl weiter wächst: Seiten zu drei Arbeitsbereichen
bündeln (Daten & Labeln · Trainieren & Vergleichen · Verstehen), gemeinsame Sidebar mit Rater,
Datensatzfilter und Pipeline-Status; Caching (`st.cache_data`) für Manifest und Labels.

---

## Vorgeschlagene Reihenfolge für die nächsten Sitzungen

1. P1 + P2.1 + P2.4 (Aufräumen, Status-Panel, Confound sichtbar machen) – eine Sitzung.
2. P3 (Embeddings + Ridge/kNN + CLIP-Zero-Shot für Typen und Konzepte) – eine bis zwei Sitzungen.
3. P5 („10 vs 8“-Seite) – profitiert sofort von P3.
4. P4 (LLM-Judge, zunächst nur Validierung auf den 312 gelabelten Bildern).
5. S1 (Labeling-UX und Duelle am oberen Ende), parallel S4 (weitere Fotos finden).
6. S3 (Sonnenstand, Wetter, Palette) – der Teil, der das Projekt wissenschaftlich interessant macht.
