// Tags -> one doha. Samples from the doha model and keeps the best candidate:
// it must have the two-line shape and must not copy 5 words in a row from a real Kabir doha;
// among those, scanning to the 13+11 meter and using the requested images rank higher.
// Shared by worker.js (browser) and check.mjs (Node).
import { GPT } from "./gpt.js";
import { grade, offsets } from "./matra.js";

export function makeWriter(meta, weights) {
  const W = {};
  for (const [k, { offset, shape }] of Object.entries(meta.tensors)) W[k] = weights.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
  const model = new GPT(meta.config, W, meta);
  const stoi = Object.fromEntries(meta.itos.map((s, i) => [s, i]));
  const grams = new Set();
  for (const d of meta.real_dohe) {
    const w = d.replace(/\n/g, " ").split(/\s+/);
    for (let i = 0; i + 5 <= w.length; i++) grams.add(w.slice(i, i + 5).join(" "));
  }
  const images = Object.fromEntries(meta.images.map((i) => [i.id, i.words]));

  const copies = (text) => {
    const w = text.replace(/\n/g, " ").split(/\s+/);
    for (let i = 0; i + 5 <= w.length; i++) if (grams.has(w.slice(i, i + 5).join(" "))) return true;
    return false;
  };
  const hasImage = (text, id) => {
    const words = new Set(text.match(/[ऀ-ॿ]+/g) || []);
    return images[id].some((w) => words.has(w));
  };

  // score: -1 unusable; meter exact 30 / near 20 / off by more 10 - distance; +5 if every image appears
  function score(text, imgs) {
    const o = offsets(text);
    if (!o || copies(text)) return -1;
    const g = grade(text);
    let s = g === "exact" ? 30 : g === "near" ? 20 : 10 - o.reduce((a, x) => a + Math.abs(x), 0);
    if (imgs.every((i) => hasImage(text, i))) s += 5;
    return s;
  }

  // tags: { theme, images: [], device } (ids or empty). Stops early on an exact/near doha with its images.
  function write(tags, { maxTries = 40, temperature = 0.8, rand = Math.random } = {}) {
    const toks = [tags.theme, ...tags.images, tags.device].filter(Boolean).map((t) => stoi[`<${t}>`]);
    const prompt = [stoi["\n"], stoi["\n"], ...toks, stoi["|"]];
    let best = null;
    for (let n = 1; n <= maxTries; n++) {
      const ids = model.generate(prompt, { maxNew: 160, temperature, stopId: stoi["॥"], rand });
      const text = ids.filter((i) => i < meta.first_tag).map((i) => meta.itos[i]).join("").trim();
      const s = score(text, tags.images);
      if (s >= 0 && (!best || s > best.score)) best = { text, score: s, tries: n };
      if (best && best.score >= 25) break;
    }
    return best;
  }

  return { write, model, stoi };
}
