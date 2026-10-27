"""The friend labeling page: one HTML file, no build step, phone first."""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex, nofollow">
<title>Wolken bewerten</title>
<style>
  :root { --bg:#0f141a; --panel:#182028; --ink:#eef2f5; --muted:#94a3b3; --line:#2a3542; --accent:#4cc3ff; --good:#2a9d8f; --bad:#e76f51; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html, body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; height:100%; }
  header { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--line); gap:10px; }
  header b { font-size:1rem; }
  header .prog { color:var(--muted); font-size:.85rem; text-align:right; }
  main { padding:10px 12px calc(16px + env(safe-area-inset-bottom)); max-width:980px; margin:0 auto; }
  .stage { position:relative; background:#000; border-radius:12px; overflow:hidden; min-height:40vh; display:flex; align-items:center; justify-content:center; }
  .stage img { max-width:100%; max-height:62vh; display:block; object-fit:contain; }
  .badge { position:absolute; top:8px; left:8px; background:rgba(0,0,0,.55); color:#fff; font-size:.75rem; padding:2px 8px; border-radius:999px; }
  .scores { display:grid; grid-template-columns:repeat(11,1fr); gap:6px; margin-top:12px; }
  .scores button { font-size:1.05rem; padding:12px 0; border-radius:10px; border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  .scores button.sel { background:var(--accent); color:#06202c; border-color:var(--accent); font-weight:700; }
  .row { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
  .row button { flex:1 1 auto; padding:12px 10px; border-radius:10px; border:1px solid var(--line); background:var(--panel); color:var(--ink); font-size:.95rem; }
  .row button.primary { background:var(--good); border-color:var(--good); color:#fff; font-weight:600; }
  .row button.half.on { background:var(--accent); color:#06202c; border-color:var(--accent); }
  .hint { color:var(--muted); font-size:.85rem; margin-top:10px; line-height:1.4; }
  .toast { position:fixed; left:50%; bottom:18px; transform:translateX(-50%); background:#223041; color:#fff; padding:8px 14px; border-radius:999px; font-size:.9rem; opacity:0; transition:opacity .2s; pointer-events:none; }
  .toast.show { opacity:1; }
  .duel { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:4px; }
  .duel .stage { min-height:30vh; cursor:pointer; border:3px solid transparent; }
  .duel .stage img { max-height:48vh; }
  .duel .stage.sel { border-color:var(--good); }
  @media (max-width:640px) { .duel { grid-template-columns:1fr; } .scores button { padding:11px 0; font-size:.95rem; } }
  .tabs { display:flex; gap:6px; margin-bottom:10px; }
  .tabs button { flex:1; padding:8px; border-radius:999px; border:1px solid var(--line); background:transparent; color:var(--muted); }
  .tabs button.on { background:var(--panel); color:var(--ink); border-color:var(--accent); }
  .done { text-align:center; padding:40px 10px; color:var(--muted); }
  .intro { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin-bottom:10px; font-size:.92rem; line-height:1.45; }
  .intro summary { cursor:pointer; color:var(--accent); }
</style>
</head>
<body>
<header><div><b id="name">…</b><div class="prog" id="prog" style="text-align:left"></div></div><div class="prog" id="core"></div></header>
<main>
  <details class="intro" id="intro">
    <summary>Wie bewerten?</summary>
    <p>Gib jedem Himmel eine Note von <b>0</b> (gar nichts) bis <b>10</b> (das beste Wolkenfoto, das du dir vorstellen kannst). Mit <b>½</b> machst du halbe Noten. Es gibt kein richtig oder falsch, es geht um deinen Geschmack. Bewerte zügig aus dem Bauch, du kannst jederzeit aufhören und später weitermachen, der Link merkt sich deinen Stand.</p>
    <p>Manche Bilder sind Ausschnitte eines größeren Fotos (Badge „Ausschnitt“). Bewerte dann nur, was du siehst. Mit <b>Duell</b> vergleichst du zwei deiner guten Bilder direkt: tippe auf das bessere.</p>
    <p>Tastatur: Ziffern 0–9, <b>+</b> = 10, <b>.</b> = halbe Note, <b>Leertaste</b> = überspringen, <b>Backspace</b> = letzte Note zurück.</p>
  </details>
  <div class="tabs"><button id="tabRate" class="on">Bewerten</button><button id="tabDuel">Duell</button></div>

  <section id="rate">
    <div class="stage"><span class="badge" id="badge"></span><img id="img" alt=""></div>
    <div class="scores" id="scores"></div>
    <div class="row">
      <button class="half" id="half">½</button>
      <button id="undo">↶ zurück</button>
      <button id="skip">überspringen</button>
    </div>
    <div class="hint" id="hint"></div>
  </section>

  <section id="duel" style="display:none">
    <div class="duel">
      <div class="stage" id="left"><span class="badge">A</span><img alt=""></div>
      <div class="stage" id="right"><span class="badge">B</span><img alt=""></div>
    </div>
    <div class="row"><button id="tie">unentschieden</button><button id="duelSkip">anderes Paar</button></div>
    <div class="hint" id="duelHint"></div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const token = location.pathname.split("/").filter(Boolean).pop();
const api = (p) => `/api/${token}/${p}`;
const $ = (id) => document.getElementById(id);
let queue = [], skipped = new Set(), half = false, lastRating = null, current = null, mode = "rate", pair = null, busy = false;

function toast(text) { const t = $("toast"); t.textContent = text; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 1400); }
function imgUrl(id) { return api(`image/${id}`); }
function prefetch() { queue.slice(1, 4).forEach(item => { const i = new Image(); i.src = imgUrl(item.image_id); }); }

async function loadNext(force) {
  if (!force && queue.length > 2) return;
  const exclude = [...skipped].join(",");
  const r = await fetch(api(`next?limit=8&exclude=${encodeURIComponent(exclude)}`));
  if (!r.ok) { toast("Link ungültig oder Server nicht erreichbar"); return; }
  const data = await r.json();
  $("name").textContent = data.display_name;
  renderProgress(data.progress);
  const known = new Set(queue.map(q => q.image_id));
  for (const item of data.items) if (!known.has(item.image_id)) queue.push(item);
  if (!current) showCurrent();
}
function renderProgress(p) {
  $("prog").textContent = `${p.n_done} von ${p.n_total} bewertet`;
  $("core").textContent = p.n_core ? `Vergleichsset ${p.n_core_done}/${p.n_core}` : "";
}
function showCurrent() {
  current = queue[0] || null;
  if (!current) { $("rate").innerHTML = '<div class="done">Alles bewertet. Danke! 🎉<br>Schau später wieder vorbei, es kommen neue Bilder dazu.</div>'; return; }
  $("img").src = imgUrl(current.image_id);
  $("badge").textContent = current.kind === "crop" ? "Ausschnitt" : "";
  $("badge").style.display = current.kind === "crop" ? "inline" : "none";
  document.querySelectorAll(".scores button").forEach(b => b.classList.remove("sel"));
  prefetch();
}
function buildScores() {
  const box = $("scores");
  for (let s = 0; s <= 10; s++) { const b = document.createElement("button"); b.textContent = s; b.onclick = () => rate(s); box.appendChild(b); }
}
async function rate(base) {
  if (!current || busy) return;
  const score = Math.min(10, base + (half ? 0.5 : 0));
  busy = true;
  try {
    const r = await fetch(api("rate"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_id: current.image_id, score }) });
    if (!r.ok) { toast("Speichern fehlgeschlagen"); return; }
    const data = await r.json();
    lastRating = { rating_id: data.rating_id, item: current, score };
    renderProgress(data.progress);
    toast(`Note ${score.toLocaleString("de-DE")} gespeichert`);
    queue.shift(); half = false; $("half").classList.remove("on");
    current = null; showCurrent(); loadNext(false);
  } finally { busy = false; }
}
async function undo() {
  if (!lastRating) { toast("Nichts zum Zurücknehmen"); return; }
  const r = await fetch(api("undo"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rating_id: lastRating.rating_id }) });
  if (!r.ok) { toast("Zurücknehmen fehlgeschlagen"); return; }
  const data = await r.json();
  renderProgress(data.progress);
  queue.unshift(lastRating.item); lastRating = null; current = null; showCurrent();
  toast("Letzte Note zurückgenommen");
}
function skip() { if (!current) return; skipped.add(current.image_id); queue.shift(); current = null; showCurrent(); loadNext(false); }

async function loadPair() {
  const r = await fetch(api("duel"));
  const data = await r.json();
  if (!data.pair) { pair = null; $("duelHint").textContent = data.message || "Noch kein Paar verfügbar."; ["left","right"].forEach(s => $(s).querySelector("img").src = ""); return; }
  pair = data.pair;
  $("left").querySelector("img").src = imgUrl(pair[0]);
  $("right").querySelector("img").src = imgUrl(pair[1]);
  $("duelHint").textContent = `Welches gefällt dir besser? ${data.n_duels} Duelle bisher.`;
  ["left","right"].forEach(s => $(s).classList.remove("sel"));
}
async function decide(winner) {
  if (!pair) return;
  const r = await fetch(api("duel"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ left: pair[0], right: pair[1], winner }) });
  if (!r.ok) { toast("Speichern fehlgeschlagen"); return; }
  toast(winner ? "Gespeichert" : "Unentschieden gespeichert");
  loadPair();
}
function setMode(m) {
  mode = m;
  $("rate").style.display = m === "rate" ? "" : "none";
  $("duel").style.display = m === "duel" ? "" : "none";
  $("tabRate").classList.toggle("on", m === "rate");
  $("tabDuel").classList.toggle("on", m === "duel");
  if (m === "duel") loadPair();
}
buildScores();
$("half").onclick = () => { half = !half; $("half").classList.toggle("on", half); };
$("undo").onclick = undo;
$("skip").onclick = skip;
$("tabRate").onclick = () => setMode("rate");
$("tabDuel").onclick = () => setMode("duel");
$("left").onclick = () => decide(pair && pair[0]);
$("right").onclick = () => decide(pair && pair[1]);
$("tie").onclick = () => decide(null);
$("duelSkip").onclick = loadPair;
document.addEventListener("keydown", (e) => {
  if (mode !== "rate") return;
  if (e.key >= "0" && e.key <= "9") rate(Number(e.key));
  else if (e.key === "+") rate(10);
  else if (e.key === ".") { half = !half; $("half").classList.toggle("on", half); }
  else if (e.key === " ") { e.preventDefault(); skip(); }
  else if (e.key === "Backspace") { e.preventDefault(); undo(); }
});
if (!localStorage.getItem("introSeen")) { $("intro").open = true; localStorage.setItem("introSeen", "1"); }
loadNext(true);
</script>
</body>
</html>
"""
