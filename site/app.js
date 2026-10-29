// Kabir Doha Generator: page logic.
// Describe it: English text -> classifier -> theme + images. Choose themes: the same tags from menus.
// Either way the tags go to worker.js, which runs the doha model and returns one finished doha.
import { create as createClassifier } from "./lib/classifier.js";
import { fetchModel } from "./lib/load.js";

const THEMES = [            // the 10 themes the doha model knows (ids from config/tags.yaml)
  ["ahankar", "Ego and pride", "अहंकार"], ["mrityu", "Death", "मृत्यु"], ["maya", "Illusion", "माया"],
  ["guru", "The guru", "गुरु"], ["bhakti", "True devotion", "भक्ति"], ["pakhand", "Empty ritual", "पाखंड"],
  ["vani", "Speech", "वाणी"], ["prem", "Love", "प्रेम"], ["satsang", "Good company", "सत्संग"], ["dhairya", "Patience", "धैर्य"],
];
const STYLES = [
  ["sadho", "Speaking to the seeker (“Sadho…”)"], ["prashn", "A question to make you think"],
  ["kahekabir", "Signed “Kahe Kabir”"], ["ulatbansi", "A riddle-like paradox"],
];
const IDEAS = [
  "pride burns out like a lamp", "life is short, death comes for everyone", "speak sweetly like a cuckoo",
  "a false guru is like a heron waiting for fish", "love cannot be learned from books",
  "patience: the tree gives fruit in its own season", "god is inside you like musk in the deer",
  "prayer beads are useless if the mind is impure",
];
const $ = (id) => document.getElementById(id);
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pick = (a) => a[Math.floor(Math.random() * a.length)];
const themeName = Object.fromEntries(THEMES.map(([id, en]) => [id, en]));
const ROSETTE = `<svg class="corner" viewBox="-12 -12 24 24" aria-hidden="true"><g fill="currentColor"><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(0)"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(45)" opacity=".55"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(90)"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(135)" opacity=".55"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(180)"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(225)" opacity=".55"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(270)"/><ellipse cy="-6.4" rx="2.3" ry="4.4" transform="rotate(315)" opacity=".55"/><circle r="2.6" style="fill:var(--gold)"/></g></svg>`;
const LOTUS = `<svg class="crest" viewBox="0 0 46 26" aria-hidden="true"><path d="M23 2c4 5 4 12 0 20c-4-8-4-15 0-20z" style="fill:var(--vermilion)"/><path d="M23 22c-2-7-7-11-14-12c2 7 7 11 14 12zM23 22c2-7 7-11 14-12c-2 7-7 11-14 12z" style="fill:var(--vermilion)" opacity=".75"/><path d="M23 23c-5-3-11-4-20-2c6 3 13 4 20 2zM23 23c5-3 11-4 20-2c-6 3-13 4-20 2z" style="fill:var(--leaf)"/></svg>`;
let clf, images = [], busy = false, reqId = 0, lastText = "";

// ---------- tabs ----------
for (const [tab, panel] of [["tabDescribe", "panelDescribe"], ["tabPick", "panelPick"]]) {
  $(tab).onclick = () => {
    for (const t of ["tabDescribe", "tabPick"]) $(t).setAttribute("aria-selected", String(t === tab));
    $("panelDescribe").hidden = panel !== "panelDescribe";
    $("panelPick").hidden = panel !== "panelPick";
    note("");
  };
}

// ---------- menus ----------
$("theme").append(new Option("Any theme", ""), ...THEMES.map(([id, en, hi]) => new Option(`${en} · ${hi}`, id)));
$("style").append(new Option("Any style", ""), ...STYLES.map(([id, en]) => new Option(en, id)));
for (const s of IDEAS) {
  const b = document.createElement("button"); b.type = "button"; b.textContent = s;
  b.onclick = () => { $("req").value = s; describe(); };
  $("ideas").append(b);
}

// ---------- output ----------
function note(text, err = false) { $("note").textContent = text; $("note").className = "note" + (err ? " err" : ""); }

function about(tags) {
  const parts = [themeName[tags.theme]];
  for (const i of tags.images) parts.push(cap(images.find((x) => x.id === i).name));
  return parts.filter(Boolean).join(" · ");
}

function show(text, tags) {
  if (lastText) {                                    // the previous doha moves to "Earlier"
    $("past").hidden = false;
    const p = verse(lastText, "div", "p");
    $("pastList").prepend(p);
  }
  lastText = text;
  const card = document.createElement("article"); card.className = "doha";
  const t = verse(text, "p", "text");
  const a = document.createElement("div"); a.className = "about"; a.textContent = about(tags);
  const acts = document.createElement("div"); acts.className = "acts";
  const copy = button("Copy", async () => {
    try { await navigator.clipboard.writeText(text); copy.textContent = "Copied"; } catch { copy.textContent = "Couldn't copy"; }
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  });
  acts.append(button("Another one", () => run(tags.again())), copy);
  if (navigator.share) acts.append(button("Share", () => navigator.share({ text }).catch(() => {})));
  card.innerHTML = ["tl", "tr", "bl", "br"].map((c) => ROSETTE.replace('class="corner"', `class="corner ${c}"`)).join("") + LOTUS;
  card.append(t, a, acts);
  $("out").replaceChildren(card);
}
// one element per line, one span per half-line, so narrow screens wrap only at the caesura (the comma)
function verse(text, tag, cls) {
  const el = document.createElement(tag); el.className = cls; el.lang = "hi";
  text.split("\n").forEach((line, i) => {
    const l = document.createElement("span"); l.className = "line";
    const k = line.indexOf(",");
    const halves = k < 0 ? [line] : [line.slice(0, k + 1), line.slice(k + 1).trim()];
    halves.forEach((h, j) => { const sp = document.createElement("span"); sp.className = "half"; sp.textContent = h; l.append(sp, j < halves.length - 1 ? " " : ""); });
    el.append(l);
  });
  return el;
}
function button(label, onclick) { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.onclick = onclick; return b; }

function writing() {
  const w = document.createElement("article"); w.className = "doha writing";
  w.innerHTML = `${ROSETTE.replace('class="corner"', 'class="rosette"')}<p class="text">Writing your doha…</p>`;
  for (const c of $("out").children) c.hidden = true;
  $("out").append(w);
}

// ---------- the model (in a worker) ----------
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let modelReady = false;
worker.onmessage = ({ data }) => {
  if (data.type === "ready") { modelReady = true; enable(); return; }
  if (data.type === "error" && data.id === undefined) { note("Sorry, the model could not load. Please reload the page.", true); return; }
  const cb = pending.get(data.id); pending.delete(data.id);
  if (cb) cb(data);
};
worker.onerror = () => note("Sorry, this browser could not start the model. Please try a recent Chrome, Safari or Firefox.", true);

function enable() {
  if (!modelReady || !clf) return;
  $("goDescribe").disabled = false; $("goDescribe").textContent = "Write a doha";
  $("goPick").disabled = false; $("goPick").textContent = "Write a doha";
}

function run(tags) {
  if (busy) return;
  busy = true; note(""); writing();
  for (const b of ["goDescribe", "goPick"]) $(b).disabled = true;
  const id = ++reqId;
  pending.set(id, (data) => {
    busy = false;
    for (const b of ["goDescribe", "goPick"]) $(b).disabled = false;
    $("out").querySelector(".writing")?.remove();
    for (const c of $("out").children) c.hidden = false;
    if (data.type === "doha" && data.result) show(data.result.text, tags);
    else note("That one didn't come out right. Please try again.", true);
  });
  worker.postMessage({ id, tags: { theme: tags.theme, images: tags.images, device: tags.device() } });
}

// ---------- Describe it ----------
function describe() {
  const text = $("req").value.trim();
  if (!text) { note("Write a few words about what the doha should say.", true); $("req").focus(); return; }
  const r = clf.predict(text);
  if (r.nPieces === 0) { note("Please describe it in English.", true); return; }
  // theme: the most likely of the 10 the doha model knows (the classifier ranks 100)
  const theme = THEMES.map(([id]) => id).sort((a, b) => (r.allThemeProbs[b] || 0) - (r.allThemeProbs[a] || 0))[0];
  const known = new Set(images.map((i) => i.id));
  const imgs = Object.entries(r.allImageProbs).filter(([id, p]) => p >= clf.meta.thresholds.t_img && known.has(id))
    .sort((a, b) => b[1] - a[1]).slice(0, 2).map(([id]) => id);
  const tags = { theme, images: imgs, device: () => pick(STYLES)[0] };
  tags.again = () => tags;
  run(tags);
}
$("goDescribe").onclick = describe;
$("req").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("goDescribe").disabled) describe(); } });

// ---------- Choose themes ----------
$("goPick").onclick = () => {
  const fixed = { theme: $("theme").value, image: $("image").value, style: $("style").value };
  const make = () => {
    const tags = {
      theme: fixed.theme || pick(THEMES)[0],
      images: fixed.image ? [fixed.image] : [],
      device: () => fixed.style || pick(STYLES)[0],
    };
    tags.again = make;      // "Another one" re-rolls whatever was left on "Any"
    return tags;
  };
  run(make());
};

// ---------- "How it works" overlay: opens on the first visit, and from the header button ----------
const how = $("how");
const closeHow = () => { how.close(); try { localStorage.setItem("seenHow", "1"); } catch {} };
$("howBtn").onclick = () => how.showModal();
$("howClose").onclick = closeHow;
$("howStart").onclick = closeHow;
how.addEventListener("click", (e) => { if (e.target === how) closeHow(); });      // tap outside the sheet
how.addEventListener("cancel", () => { try { localStorage.setItem("seenHow", "1"); } catch {} });   // Esc
let seen = false;
try { seen = localStorage.getItem("seenHow") === "1"; } catch {}
if (!seen && how.showModal) how.showModal();

// ---------- start ----------
try {
  const [c, m] = await Promise.all([
    fetchModel("./models/", "classifier"),
    fetch("./models/doha.json").then((r) => r.json()),     // only for the image list; the worker loads the weights
  ]);
  clf = createClassifier(c.meta, c.weights.buffer);
  images = m.images;
  $("image").append(new Option("No particular image", ""),
    ...[...images].sort((a, b) => a.name.localeCompare(b.name)).map((i) => new Option(`${cap(i.name)} · ${i.words[0]}`, i.id)));
  enable();
} catch (e) {
  note("Sorry, the page could not load. Please check your connection and reload.", true);
}

if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("./sw.js").catch(() => {});
