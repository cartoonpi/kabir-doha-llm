// Matra (beat) counting and the doha meter check: a line-by-line port of ../matra.py
// (syllables, count_range, offsets, check, grade). web/steer_parity.mjs checks it agrees with Python.
//
//   import { grade, halves } from "./matra.js";
//   grade("…, …।\n…, …॥")   // "exact" | "near" | null

const CONS = new Set();
for (let c = 0x0915; c <= 0x0939; c++) CONS.add(String.fromCharCode(c));
for (const c of "क़ख़ग़ज़ड़ढ़फ़य़") CONS.add(c);
const NUKTA = "़", VIRAMA = "्", ANUSVARA = "ं", VISARGA = "ः", CHANDRA = "ँ";
const SHORT_SIGNS = new Set("िुृ"), LONG_SIGNS = new Set("ाीूेैोौ");
const SHORT_VOWELS = new Set("अइउऋ"), LONG_VOWELS = new Set("आईऊएऐओऔ");
const FLEX_WORDS = new Set(["है", "हैं", "मैं", "में", "के", "से", "जो", "सो", "ते", "को", "तो", "ने", "जे", "कै", "सों", "तें"]);
const WORDS = /[ऀ-ॣॱ-ॿ]+/g;

export function syllables(word, nasal = false) {
  const out = [];
  let i = 0;
  const n = word.length;
  while (i < n) {
    const start = i, c = word[i];
    let w;
    if (CONS.has(c)) {
      i++;
      if (i < n && word[i] === NUKTA) i++;
      while (i + 1 < n && word[i] === VIRAMA && CONS.has(word[i + 1])) {   // cluster
        if (out.length && start > 0 && !out[out.length - 1][2]) {
          out[out.length - 1][1] = 2;                                        // joined consonant lengthens the previous syllable
          out[out.length - 1][2] = true;
        }
        i += 2;
        if (i < n && word[i] === NUKTA) i++;
      }
      if (i < n && word[i] === VIRAMA) { i++; continue; }                  // word-final half consonant
      if (i < n && LONG_SIGNS.has(word[i])) { w = 2; i++; }
      else if (i < n && SHORT_SIGNS.has(word[i])) { w = 1; i++; }
      else w = 1;                                                           // inherent अ
    } else if (LONG_VOWELS.has(c)) { w = 2; i++; }
    else if (SHORT_VOWELS.has(c)) { w = 1; i++; }
    else { i++; continue; }                                                 // stray sign or punctuation
    let locked = w === 2;
    const afterShortSign = i > start ? SHORT_SIGNS.has(word[i - 1]) : false;
    while (i < n && (word[i] === ANUSVARA || word[i] === VISARGA || word[i] === CHANDRA)) {
      if (word[i] === ANUSVARA && nasal && afterShortSign && i === n - 1) { /* nasal rule */ }
      else if (word[i] === ANUSVARA || word[i] === VISARGA) { w = 2; locked = true; }
      i++;
    }
    out.push([word.slice(start, i), w, locked]);
  }
  return out.map(([s, w]) => [s, w]);
}

export function countRange(text, nasal = false, flex = false) {
  let lo = 0, hi = 0;
  for (const word of text.match(WORDS) || []) {
    const syl = syllables(word, nasal);
    syl.forEach(([s, w], k) => {
      hi += w;
      const bendable = flex && w === 2 && (FLEX_WORDS.has(word) || (k === syl.length - 1 && s.endsWith("ै") && syl.length > 1));
      lo += bendable ? 1 : w;
    });
  }
  return [lo, hi];
}

const strip = (s, chars) => {
  let a = 0, b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
};

// [l1a, l1b, l2a, l2b] distance of each half from 13 / 11 (0 = scans), or null without the 2-line comma shape
export function offsets(text, nasal = true, flex = true) {
  const lines = text.trim().split("\n");
  if (lines.length !== 2) return null;
  const out = [];
  for (let line of lines) {
    line = strip(line, " ।॥");
    const k = line.indexOf(",");
    if (k < 0) return null;
    for (const [half, t] of [[line.slice(0, k), 13], [line.slice(k + 1), 11]]) {
      const [lo, hi] = countRange(half, nasal, flex);
      out.push(lo <= t && t <= hi ? 0 : lo > t ? lo - t : hi - t);
    }
  }
  return out;
}

// matra counts per half, for display: [[lo, hi] x 4] or null
export function halves(text) {
  const lines = text.trim().split("\n");
  if (lines.length !== 2) return null;
  const out = [];
  for (let line of lines) {
    line = strip(line, " ।॥");
    const k = line.indexOf(",");
    if (k < 0) return null;
    out.push(countRange(line.slice(0, k), true, true), countRange(line.slice(k + 1), true, true));
  }
  return out;
}

export function grade(text) {
  const o = offsets(text);
  if (!o) return null;
  const bad = o.filter((x) => x !== 0);
  if (bad.length === 0) return "exact";
  return bad.length === 1 && Math.abs(bad[0]) === 1 ? "near" : null;
}
