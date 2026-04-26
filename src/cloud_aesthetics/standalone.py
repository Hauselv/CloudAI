from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from cloud_aesthetics.data.ratings import record_pairwise_preference, record_rating
from cloud_aesthetics.settings import ensure_parent, resolve_path
from cloud_aesthetics.utils.io import read_table


FRIEND_PACKAGE_VERSION = 1


def _json_script_payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _safe_package_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
    return safe.strip("_") or "cloud_labeling_package"


def _package_readme() -> str:
    return """Cloud Aesthetics Freundes-Paket

Start:
1. Ordner entpacken.
2. index.html im Browser öffnen.
3. Namen/Rater-ID eintragen, Bilder bewerten.
4. Am Ende "Export labels" klicken und die erzeugte JSON-Datei zurückschicken.

Optional:
- Eigene Bilder können über "Import images" hinzugefügt werden.
- Die App erzeugt daraus lokale Sky/Cloud-Crops im Browser.
- Beim JSON-Export werden eigene importierte Bilder als Daten mitgespeichert.

Es ist keine Installation nötig. Die Bewertungen bleiben nur lokal im Browser und in der Export-Datei.
"""


def build_friend_package(
    output: str | Path,
    *,
    manifest_path: str | Path = "data/processed/image_manifest.parquet",
    package_name: str = "cloud_labeling_friend_package",
    rater_hint: str = "friend",
    zip_package: bool = True,
) -> Path:
    manifest = read_table(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty. Run ingest-images before exporting a friend package.")

    output_path = resolve_path(output)
    package_dir = output_path.with_suffix("") if output_path.suffix.lower() == ".zip" else output_path
    package_dir.mkdir(parents=True, exist_ok=True)
    image_dir = package_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        source = resolve_path(row["relative_path"])
        if not source.exists():
            continue
        target = image_dir / f"{row['image_id']}_{source.name}"
        if not target.exists():
            shutil.copy2(source, target)
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "path": f"images/{target.name}",
                "original_relative_path": str(row["relative_path"]),
                "width": int(row.get("width", 0) or 0),
                "height": int(row.get("height", 0) or 0),
                "split_group_id": str(row.get("split_group_id", "")),
            }
        )

    payload = {
        "version": FRIEND_PACKAGE_VERSION,
        "package_name": _safe_package_name(package_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rater_hint": rater_hint,
        "images": rows,
    }
    (package_dir / "index.html").write_text(_standalone_html(payload), encoding="utf-8")
    (package_dir / "README_FREUNDE.txt").write_text(_package_readme(), encoding="utf-8")
    (package_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_package:
        zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
        ensure_parent(zip_path)
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in package_dir.rglob("*"):
                archive.write(path, path.relative_to(package_dir))
        return zip_path
    return package_dir


def import_friend_label_bundle(
    bundle_path: str | Path,
    *,
    ratings_dir: str | Path = "data/raw/metadata/ratings",
    pairwise_dir: str | Path = "data/raw/metadata/pairwise",
    imported_images_root: str | Path = "data/raw/images/friend_imports",
) -> dict[str, int]:
    bundle = json.loads(resolve_path(bundle_path).read_text(encoding="utf-8"))
    ratings = bundle.get("ratings", [])
    pairwise = bundle.get("pairwise", [])
    imported_images = bundle.get("imported_images", [])

    imported_count = 0
    for image in imported_images:
        data_url = str(image.get("data_url", ""))
        if "," not in data_url:
            continue
        _, encoded = data_url.split(",", 1)
        suffix = ".jpg" if "jpeg" in data_url or "jpg" in data_url else ".png"
        target = resolve_path(imported_images_root) / f"{image.get('image_id', 'friend_image')}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))
        imported_count += 1

    rating_count = 0
    for item in ratings:
        record_rating(
            ratings_dir,
            image_id=str(item["image_id"]),
            rater_id=str(item.get("rater_id") or bundle.get("rater_id") or "friend"),
            score=float(item["score"]),
            rating_session_id=str(item.get("rating_session_id") or bundle.get("session_id") or "friend_export"),
            note=item.get("note"),
        )
        rating_count += 1

    pairwise_count = 0
    for item in pairwise:
        record_pairwise_preference(
            pairwise_dir,
            left_image_id=str(item["left_image_id"]),
            right_image_id=str(item["right_image_id"]),
            rater_id=str(item.get("rater_id") or bundle.get("rater_id") or "friend"),
            winner=item.get("winner"),
            tie_flag=bool(item.get("tie_flag", False)),
            preference_strength=float(item.get("preference_strength", 0.5)),
        )
        pairwise_count += 1

    return {"ratings": rating_count, "pairwise": pairwise_count, "imported_images": imported_count}


def _standalone_html(payload: dict[str, object]) -> str:
    manifest_json = _json_script_payload(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cloud Rating</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fb; --fg:#17202a; --muted:#6a7280; --line:#d8dde6; --accent:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111318; --fg:#eef2f8; --muted:#a5adba; --line:#303642; --accent:#60a5fa; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--fg); }}
header {{ display:flex; gap:12px; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid var(--line); }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 320px; min-height:calc(100vh - 58px); }}
.stage {{ display:flex; flex-direction:column; align-items:center; justify-content:center; padding:14px; min-width:0; }}
.stage img {{ max-width:100%; max-height:calc(100vh - 220px); object-fit:contain; background:#101114; border-radius:6px; }}
aside {{ border-left:1px solid var(--line); padding:14px; overflow:auto; }}
button, input, select, textarea {{ font:inherit; }}
button {{ border:1px solid var(--line); border-radius:6px; padding:8px 10px; background:transparent; color:var(--fg); cursor:pointer; }}
button.primary {{ background:var(--accent); border-color:var(--accent); color:white; }}
button:disabled {{ opacity:.45; cursor:not-allowed; }}
input, select, textarea {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; background:transparent; color:var(--fg); }}
label {{ display:block; margin:10px 0 4px; color:var(--muted); font-size:13px; }}
.row {{ display:flex; gap:8px; align-items:center; }}
.row > * {{ flex:1; }}
.tabs {{ display:flex; gap:8px; margin-bottom:12px; }}
.tabs button.active {{ border-color:var(--accent); color:var(--accent); }}
.meta {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
.progress {{ height:8px; background:var(--line); border-radius:99px; overflow:hidden; margin:8px 0 12px; }}
.progress span {{ display:block; height:100%; background:var(--accent); }}
.hidden {{ display:none !important; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; width:100%; }}
.pair img {{ max-height:calc(100vh - 185px); width:100%; object-fit:contain; }}
@media (max-width: 860px) {{ main {{ grid-template-columns:1fr; }} aside {{ border-left:0; border-top:1px solid var(--line); }} }}
</style>
</head>
<body>
<header>
  <strong>Cloud Rating</strong>
  <div class="row" style="max-width:560px">
    <input id="rater" placeholder="Rater ID">
    <button id="export" class="primary">Export labels</button>
  </div>
</header>
<main>
  <section class="stage">
    <div id="scalarStage">
      <img id="image" alt="">
      <p id="caption" class="meta"></p>
    </div>
    <div id="pairStage" class="pair hidden">
      <div><img id="leftImage" alt=""><p id="leftCaption" class="meta"></p></div>
      <div><img id="rightImage" alt=""><p id="rightCaption" class="meta"></p></div>
    </div>
  </section>
  <aside>
    <div class="tabs">
      <button id="scalarTab" class="active">Rating</button>
      <button id="pairTab">Pairwise</button>
      <button id="importTab">Import</button>
    </div>
    <div id="scalarPanel">
      <div class="meta" id="count"></div>
      <div class="progress"><span id="bar"></span></div>
      <label>Image</label><select id="picker"></select>
      <label>Score: <span id="scoreText">8</span></label><input id="score" type="range" min="0" max="10" step="0.5" value="8">
      <label>Notes</label><textarea id="note" rows="3"></textarea>
      <div class="row" style="margin-top:12px"><button id="prev">Previous</button><button id="next">Next</button></div>
      <button id="save" class="primary" style="width:100%; margin-top:8px">Save & next</button>
    </div>
    <div id="pairPanel" class="hidden">
      <label>Winner</label><select id="winner"><option value="">Tie</option></select>
      <label>Strength: <span id="strengthText">0.5</span></label><input id="strength" type="range" min="0" max="1" step="0.05" value="0.5">
      <div class="row" style="margin-top:12px"><button id="newPair">New pair</button><button id="savePair" class="primary">Save pair</button></div>
    </div>
    <div id="importPanel" class="hidden">
      <label>Import images</label><input id="files" type="file" accept="image/*" multiple>
      <div class="row"><input id="maxCrops" type="number" min="0" max="24" value="8"><button id="clearImports">Clear imports</button></div>
      <p class="meta">Imported images and browser-generated crops are included in the JSON export.</p>
    </div>
  </aside>
</main>
<script>
const manifest = {manifest_json};
const storageKey = "cloud-rating-" + manifest.package_name;
const state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
state.ratings ??= [];
state.pairwise ??= [];
state.importedImages ??= [];
state.importedImages = state.importedImages.filter((item) => item.data_url);
state.index ??= 0;
state.mode ??= "scalar";
state.sessionId ??= new Date().toISOString().slice(0,10) + "_friend_session";

const $ = (id) => document.getElementById(id);
const allImages = () => manifest.images.concat(state.importedImages);
const saveState = () => {{
  const compact = {{...state, importedImages: state.importedImages.map(({{data_url, ...rest}}) => rest)}};
  try {{ localStorage.setItem(storageKey, JSON.stringify(compact)); }} catch (error) {{ console.warn("Local save skipped", error); }}
}};
const ratedIds = () => new Set(state.ratings.map((r) => r.image_id));
const imageSrc = (item) => item.data_url || item.path;

function setMode(mode) {{
  state.mode = mode; saveState();
  $("scalarTab").classList.toggle("active", mode === "scalar");
  $("pairTab").classList.toggle("active", mode === "pair");
  $("importTab").classList.toggle("active", mode === "import");
  $("scalarPanel").classList.toggle("hidden", mode !== "scalar");
  $("pairPanel").classList.toggle("hidden", mode !== "pair");
  $("importPanel").classList.toggle("hidden", mode !== "import");
  $("scalarStage").classList.toggle("hidden", mode === "pair");
  $("pairStage").classList.toggle("hidden", mode !== "pair");
  render();
}}

function renderPicker() {{
  const ids = ratedIds();
  $("picker").innerHTML = "";
  allImages().forEach((item, index) => {{
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${{ids.has(item.image_id) ? "[rated]" : "[open]"}} ${{item.image_id}}`;
    $("picker").appendChild(option);
  }});
  $("picker").value = String(state.index);
}}

function renderScalar() {{
  const images = allImages();
  if (!images.length) return;
  state.index = Math.max(0, Math.min(state.index, images.length - 1));
  const item = images[state.index];
  $("image").src = imageSrc(item);
  $("caption").textContent = item.original_relative_path || item.name || item.path;
  const rating = state.ratings.find((r) => r.image_id === item.image_id);
  $("score").value = rating?.score ?? 8;
  $("scoreText").textContent = $("score").value;
  $("note").value = rating?.note ?? "";
  renderPicker();
  const done = ratedIds().size;
  $("count").textContent = `Labeled ${{done}} / ${{images.length}}`;
  $("bar").style.width = `${{images.length ? (done / images.length) * 100 : 0}}%`;
}}

function pickPair() {{
  const images = allImages();
  if (images.length < 2) return;
  let left = Math.floor(Math.random() * images.length);
  let right = Math.floor(Math.random() * images.length);
  while (right === left) right = Math.floor(Math.random() * images.length);
  state.currentPair = [left, right]; saveState();
}}

function renderPair() {{
  const images = allImages();
  if (images.length < 2) return;
  if (!state.currentPair) pickPair();
  const [li, ri] = state.currentPair;
  const left = images[li], right = images[ri];
  $("leftImage").src = imageSrc(left); $("rightImage").src = imageSrc(right);
  $("leftCaption").textContent = left.image_id; $("rightCaption").textContent = right.image_id;
  $("winner").innerHTML = `<option value="">Tie</option><option value="${{left.image_id}}">Left</option><option value="${{right.image_id}}">Right</option>`;
}}

function render() {{
  $("rater").value = state.raterId || manifest.rater_hint || "friend";
  renderScalar();
  renderPair();
}}

function saveRating() {{
  const item = allImages()[state.index];
  const row = {{ image_id:item.image_id, rater_id:$("rater").value || "friend", rating_session_id:state.sessionId, score:Number($("score").value), note:$("note").value, timestamp:new Date().toISOString() }};
  state.raterId = row.rater_id;
  state.ratings = state.ratings.filter((r) => r.image_id !== item.image_id);
  state.ratings.push(row);
  nextUnrated();
}}

function nextUnrated() {{
  const images = allImages();
  const ids = ratedIds();
  for (let offset = 1; offset <= images.length; offset++) {{
    const candidate = (state.index + offset) % images.length;
    if (!ids.has(images[candidate].image_id)) {{ state.index = candidate; saveState(); render(); return; }}
  }}
  state.index = (state.index + 1) % images.length; saveState(); render();
}}

async function sha256Hex(buffer) {{
  if (!globalThis.crypto?.subtle) return `${{Date.now().toString(16)}}${{Math.random().toString(16).slice(2)}}`;
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}}

function rgbToHsv(r, g, b) {{
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r,g,b), min = Math.min(r,g,b), d = max - min;
  let h = 0; if (d) h = max === r ? ((g-b)/d)%6 : max === g ? (b-r)/d+2 : (r-g)/d+4;
  h = Math.round(h * 30); if (h < 0) h += 180;
  return [h, max ? d / max * 255 : 0, max * 255];
}}

function cropStats(data, width, x, y, size) {{
  let sky = 0, cloud = 0, total = 0;
  const step = Math.max(1, Math.floor(size / 64));
  for (let yy = y; yy < y + size; yy += step) for (let xx = x; xx < x + size; xx += step) {{
    const i = (yy * width + xx) * 4, hsv = rgbToHsv(data[i], data[i+1], data[i+2]);
    const blueSky = hsv[0] >= 85 && hsv[0] <= 135 && hsv[1] >= 20 && hsv[2] >= 70;
    const brightCloud = hsv[1] <= 95 && hsv[2] >= 115;
    const hazySky = hsv[1] <= 70 && hsv[2] >= 145;
    if (blueSky || brightCloud || hazySky) sky++;
    if (hsv[1] <= 85 && hsv[2] >= 125) cloud++;
    total++;
  }}
  return {{ sky: sky / total, cloud: cloud / total }};
}}

async function importFile(file) {{
  const buffer = await file.arrayBuffer();
  const hash = await sha256Hex(buffer);
  const dataUrl = await new Promise((resolve) => {{ const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsDataURL(file); }});
  const img = await new Promise((resolve) => {{ const image = new Image(); image.onload = () => resolve(image); image.src = dataUrl; }});
  state.importedImages.push({{ image_id:hash.slice(0,16), name:file.name, original_relative_path:file.name, data_url:dataUrl, imported:true }});
  const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d", {{ willReadFrequently:true }});
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; ctx.drawImage(img, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const shortEdge = Math.min(canvas.width, canvas.height), maxCrops = Number($("maxCrops").value || 8);
  const candidates = [];
  [0.35,0.5,0.7,0.9,1.0].forEach((scale) => {{
    const size = Math.max(384, Math.round(shortEdge * scale));
    if (size > canvas.width || size > canvas.height) return;
    const step = Math.max(1, Math.round(size * 0.5));
    for (let y = 0; y <= canvas.height - size; y += step) for (let x = 0; x <= canvas.width - size; x += step) {{
      const stats = cropStats(pixels, canvas.width, x, y, size);
      if (stats.sky >= 0.72 && stats.cloud >= 0.08) candidates.push({{x,y,size,...stats}});
    }}
  }});
  candidates.sort((a,b) => (b.sky + b.cloud * 0.5 + b.size / 10000) - (a.sky + a.cloud * 0.5 + a.size / 10000));
  for (const [index, crop] of candidates.slice(0, maxCrops).entries()) {{
    const c = document.createElement("canvas"), cx = c.getContext("2d");
    c.width = crop.size; c.height = crop.size; cx.drawImage(img, crop.x, crop.y, crop.size, crop.size, 0, 0, crop.size, crop.size);
    state.importedImages.push({{ image_id:`${{hash.slice(0,16)}}_crop${{String(index+1).padStart(2,"0")}}`, name:file.name, original_relative_path:file.name, data_url:c.toDataURL("image/jpeg", 0.9), imported:true, crop_x:crop.x, crop_y:crop.y, crop_width:crop.size, crop_height:crop.size }});
  }}
}}

$("save").onclick = saveRating;
$("prev").onclick = () => {{ state.index = Math.max(0, state.index - 1); saveState(); render(); }};
$("next").onclick = () => {{ state.index = Math.min(allImages().length - 1, state.index + 1); saveState(); render(); }};
$("picker").onchange = (event) => {{ state.index = Number(event.target.value); saveState(); render(); }};
$("score").oninput = () => $("scoreText").textContent = $("score").value;
$("strength").oninput = () => $("strengthText").textContent = $("strength").value;
$("scalarTab").onclick = () => setMode("scalar");
$("pairTab").onclick = () => setMode("pair");
$("importTab").onclick = () => setMode("import");
$("newPair").onclick = () => {{ pickPair(); render(); }};
$("savePair").onclick = () => {{
  const images = allImages(), [li, ri] = state.currentPair, left = images[li], right = images[ri], winner = $("winner").value;
  state.pairwise.push({{ left_image_id:left.image_id, right_image_id:right.image_id, rater_id:$("rater").value || "friend", winner:winner || null, tie_flag:!winner, preference_strength:Number($("strength").value), timestamp:new Date().toISOString() }});
  pickPair(); render();
}};
$("files").onchange = async (event) => {{ for (const file of event.target.files) await importFile(file); saveState(); render(); }};
$("clearImports").onclick = () => {{ state.importedImages = []; saveState(); render(); }};
$("export").onclick = () => {{
  state.raterId = $("rater").value || "friend"; saveState();
  const bundle = {{ version:manifest.version, package_name:manifest.package_name, rater_id:state.raterId, session_id:state.sessionId, exported_at:new Date().toISOString(), ratings:state.ratings, pairwise:state.pairwise, imported_images:state.importedImages }};
  const blob = new Blob([JSON.stringify(bundle, null, 2)], {{type:"application/json"}});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${{manifest.package_name}}_${{state.raterId}}_labels.json`; link.click();
}};
document.addEventListener("keydown", (event) => {{
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowRight") $("next").click();
  if (event.key === "ArrowLeft") $("prev").click();
  if (event.key === "Enter") $("save").click();
}});
setMode(state.mode);
</script>
</body>
</html>"""
