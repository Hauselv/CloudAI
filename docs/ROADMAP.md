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

### P2 · Crop-Confound auflösen (Aufwand S–M, aber entscheidend)
Der stärkste Prädiktor ist derzeit „Crop oder Original“ (ρ = −0,51). Optionen, die sich nicht
ausschließen:
1. **Metriken und Erklärungen getrennt nach `kind`** ausweisen (Minimum).
2. **Primäranalyse nur auf Originalen**; Crops bleiben für Augmentation/Robustheit, werden aber nicht
   als eigenständige Labels in die 10-vs-8-Analyse gemischt.
3. Crops **ohne Hochskalieren** mit Mindestgröße (z. B. ≥ 768 px) erzeugen, damit Schärfe kein
   Artefakt mehr ist; bestehende kleine Crops als `derivative_kind = small_crop` kennzeichnen.
4. In der Rate-Seite bei Crops das **Original mit markiertem Ausschnitt** daneben zeigen, damit das
   Rating bewusst „Ausschnitt-Ästhetik“ bewertet.

### P3 · Bild-Embeddings als Rückgrat (Aufwand M)
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

### P4 · LLM-Judge als zusätzlicher Rater (Aufwand M)
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

### P5 · „10 vs 8“-Analyseseite (Aufwand S–M)
`eval/analysis.compare_groups` existiert bereits (Welch-t, Cohen's d), wird aber nirgends in der UI genutzt.
- Zwei frei wählbare Score-Bänder (z. B. 9,5–10 vs. 7,5–8,5), optional nur Originale, optional ein Rater.
- Tabelle der Features/Konzepte/Embedding-Konzepte nach Effektstärke, mit Verteilungs-Violinplots.
- Darunter die Beispielkacheln beider Gruppen nebeneinander – das ist der Moment, in dem man
  Hypothesen bekommt („die 10er haben fast alle Gegenlicht + Mehrschichtigkeit“).
- Feature-vs-Score-Streudiagramm mit Thumbnail-Hover (Plotly), damit man Ausreißer anklicken kann.

---

## S – Sekundär: Labeling beschleunigen, Modelle und Unsicherheit verbessern

### S1 · Labeling-UX (Aufwand S–M)
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

### S2 · Modell- und Evaluations-Infrastruktur (Aufwand M)
- 5-fach-CV als Standard; Run-Vergleichsseite (Leaderboard über alle Runs, nach `kind`, Rater, Datensatz).
- Unsicherheit: Konformal über alle Folds poolen oder GP/Quantil-Regression; Kalibrierungsplot.
- Mehrere Zielgrößen: `mean_score`, Rang (Spearman-Ziel), Top-Band-Klassifikation (≥ 9 ja/nein).
- SHAP für die Baummodelle statt Ein-Feature-Sensitivität (Bibliothek ist installiert).
- Deep-Pfad erst wieder anfassen, wenn Labels ≥ 500 und ein GPU-Weg steht; dann Fine-Tuning statt
  eingefrorenem Backbone, ggf. ConvNeXt-Tiny/EfficientNet.

### S3 · Wissenschaftlicher Layer, der den Namen verdient (Aufwand M–L)
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

### T1 · Kontrafaktischer Explorer („Was macht aus der 8 eine 10?“)
Bild kontrolliert verändern (Kontrast ±, Sättigung ±, Wärme ±, Beschnitt, Himmelanteil) und die
Modellvorhersage live beobachten. Zeigt Sensitivitäten pro Bild und ist die direkteste Antwort auf
die Leitfrage – vorausgesetzt, das Modell ist gut genug (nach P3).

### T2 · Konzept-Flaschenhals-Modell
Rating = lineare Funktion aus ~15 benannten Konzepten (aus CLIP oder LLM-Attributen). Weniger genau als
ein Black-Box-Modell, aber vollständig lesbar: „+1,2 für Gegenlicht, +0,8 für Mehrschichtigkeit,
−0,6 für Landschaftsanteil“.

### T3 · Semantische Karte
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
