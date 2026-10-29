// Step 8.2: browser inference for the doha tag classifier. No dependencies.
//
//   import { load } from "./classifier.js";
//   const clf = await load("../artifacts/classifier/");
//   clf.predict("something about how life runs out, use a lamp");
//   // -> { theme: "mrityu", themeProb: 0.57, images: [{id: "deepak", prob: 0.98}], allThemeProbs: {...} }
//
// Every step mirrors Python exactly: src/featurize.py, src/model.py
// (FastTextPredictor) and src/calibrate.py (decide).

// ---------------------------------------------------------------- featurize

const NOT_ALLOWED = /[^a-z0-9 ]/g;
const encoder = new TextEncoder(); // string -> UTF-8 bytes, like Python's .encode("utf-8")

export function words(text) {
  // Python: re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
  // split(" ") + filter drops the empty strings that consecutive spaces create,
  // which is what Python's argument-less split() does.
  return text.toLowerCase().replace(NOT_ALLOWED, " ").split(" ").filter((w) => w.length > 0);
}

export function pieces(text) {
  const out = [];
  for (const w of words(text)) {
    const wrapped = "<" + w + ">";
    out.push(wrapped);
    if (w.length > 1) {
      for (let i = 0; i + 3 <= wrapped.length; i++) out.push(wrapped.slice(i, i + 3));
    }
  }
  return out;
}

export function fnv1a(str) {
  // JS numbers are 64-bit floats, so h * 16777619 would lose precision.
  // Math.imul does a true 32-bit integer multiply; ">>> 0" reinterprets the
  // result as an unsigned 32-bit number (Python's "& 0xFFFFFFFF").
  let h = 2166136261;
  for (const byte of encoder.encode(str)) {
    h ^= byte;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0; // >>> 0 also covers the empty string (no loop iterations)
}

export function featurize(text, B) {
  return pieces(text).map((p) => fnv1a(p) % B);
}

// ---------------------------------------------------------------- model

// Build a classifier from meta.json (parsed) + weights.bin (ArrayBuffer).
// Separate from load() so Node's parity test can pass files read from disk.
export function create(meta, buffer) {
  const view = (name) => {
    const a = meta.arrays[name];
    const n = a.shape.reduce((x, y) => x * y, 1);
    // A typed array is a window onto the buffer: no copying, no parsing.
    return a.dtype === "int8" ? new Int8Array(buffer, a.offset, n) : new Float32Array(buffer, a.offset, n);
  };
  const { B, d } = meta;
  const T = meta.theme_ids.length;
  const I = meta.image_ids.length;
  const emb = meta.int8 ? view("embedding_q") : view("embedding");
  const scale = meta.int8 ? view("embedding_scale") : null;
  const hW = meta.hidden ? view("hidden_W") : null;
  const hb = meta.hidden ? view("hidden_b") : null;
  const tW = view("theme_W"), tb = view("theme_b");
  const iW = view("image_W"), ib = view("image_b");

  // y = W x + b, with W stored row-major as [rows x cols] (PyTorch Linear layout).
  function linear(W, b, x, rows, cols) {
    const y = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let s = b[r];
      for (let c = 0; c < cols; c++) s += W[r * cols + c] * x[c];
      y[r] = s;
    }
    return y;
  }

  function probabilities(text) {
    const ids = featurize(text, B);
    if (ids.length === 0) {
      // Same rule as Python: nothing to average -> certain "none", no images.
      return {
        ids,
        themeProbs: meta.theme_ids.map((t) => (t === "none" ? 1 : 0)),
        imageProbs: new Array(I).fill(0),
      };
    }
    // EmbeddingBag(mode="mean"): average the rows of all pieces (duplicates count).
    let h = new Float64Array(d);
    for (const id of ids) {
      const s = scale ? scale[id] : 1;
      for (let k = 0; k < d; k++) h[k] += emb[id * d + k] * s;
    }
    for (let k = 0; k < d; k++) h[k] /= ids.length;
    if (hW) h = linear(hW, hb, h, d, d).map((v) => Math.max(0, v)); // ReLU

    // Softmax: subtract the max logit first so exp() can't overflow.
    const tl = linear(tW, tb, h, T, d);
    const m = Math.max(...tl);
    const ex = tl.map((v) => Math.exp(v - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    const themeProbs = Array.from(ex, (v) => v / sum);
    const imageProbs = Array.from(linear(iW, ib, h, I, d), (v) => 1 / (1 + Math.exp(-v))); // sigmoid
    return { ids, themeProbs, imageProbs };
  }

  function predict(text) {
    const { ids, themeProbs, imageProbs } = probabilities(text);
    const { t_theme, t_img, max_images } = meta.thresholds;
    const allThemeProbs = Object.fromEntries(meta.theme_ids.map((t, i) => [t, themeProbs[i]]));

    // decide(): argmax theme, "none" if below t_theme (mirror of src/calibrate.py).
    let best = 0;
    for (let i = 1; i < T; i++) if (themeProbs[i] > themeProbs[best]) best = i; // first max wins, like Python max()
    let theme = meta.theme_ids[best];
    if (themeProbs[best] < t_theme) theme = "none";

    // Images: the (up to max_images) most likely ones with prob >= t_img,
    // then returned in tag-file order. Same as Python, where sorting back to
    // file order avoids float32-vs-float64 ties reordering two images at ~1.0.
    const images = meta.image_ids
      .map((id, i) => ({ id, prob: imageProbs[i], i }))
      .sort((a, b) => b.prob - a.prob)
      .filter((x) => x.prob >= t_img)
      .slice(0, max_images)
      .sort((a, b) => a.i - b.i)
      .map(({ id, prob }) => ({ id, prob }));

    return { theme, themeProb: allThemeProbs[theme], images, allThemeProbs,
             allImageProbs: Object.fromEntries(meta.image_ids.map((id, i) => [id, imageProbs[i]])),
             nPieces: ids.length, buckets: ids };
  }

  return { meta, predict };
}

// Browser entry point: fetch both files relative to baseUrl.
export async function load(baseUrl) {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const [meta, buffer] = await Promise.all([
    fetch(base + "meta.json").then((r) => { if (!r.ok) throw new Error("meta.json: HTTP " + r.status); return r.json(); }),
    fetch(base + "weights.bin").then((r) => { if (!r.ok) throw new Error("weights.bin: HTTP " + r.status); return r.arrayBuffer(); }),
  ]);
  return create(meta, buffer);
}
