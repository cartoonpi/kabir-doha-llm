// The doha transformer's forward pass in plain JavaScript, no libraries.
// Mirrors model.py line for line; weights come from export.py.
//
//   import { loadModel } from "./gpt.js";
//   const m = await loadModel("models/big/", 1500);
//   const s = m.newState();                 // empty context
//   const logits = m.feed(s, [12, 31, 47]);  // returns next-char scores after the last id
//
// One difference from model.py: we process ONE character at a time and remember each
// layer's keys and values ("KV cache"). Attention at a new position only needs the old
// keys/values, which never change, so we don't recompute the whole prefix for every
// new character. Same numbers, much less work.

export async function loadMeta(base) {
  const r = await fetch(base + "meta.json");
  if (!r.ok) throw new Error(`meta.json: HTTP ${r.status}`);
  return r.json();
}

export async function loadModel(base, step, meta) {
  meta = meta || (await loadMeta(base));
  const r = await fetch(`${base}step_${step}.bin`);
  if (!r.ok) throw new Error(`step_${step}.bin: HTTP ${r.status}`);
  const all = new Float32Array(await r.arrayBuffer()); // little-endian on every real machine
  const W = {};
  for (const [k, { offset, shape }] of Object.entries(meta.tensors)) {
    W[k] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
  }
  return new GPT(meta.config, W, meta);
}

// ------------------------------------------------------------ building blocks

// y = W x (+ b). PyTorch stores Linear weights as [out, in], row-major.
function linear(x, w, b, outDim) {
  const inDim = x.length, y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = b ? b[o] : 0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) s += w[row + i] * x[i];
    y[o] = s;
  }
  return y;
}

// LayerNorm: shift to mean 0, scale to variance 1, then learned scale (w) and shift (b).
function layerNorm(x, w, b) {
  const n = x.length;
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2; v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
  return y;
}

// GELU as PyTorch defines it: 0.5 x (1 + erf(x / sqrt 2)). JS has no erf, so we use
// Abramowitz & Stegun 7.1.26 (max error 1.5e-7, far below float32 noise here).
function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export function softmax(logits, temperature = 1) {
  let max = -Infinity; for (const v of logits) max = Math.max(max, v / temperature);
  const p = new Float32Array(logits.length); let sum = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] / temperature - max); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

// ------------------------------------------------------------ the model

class GPT {
  constructor(cfg, W, meta) {
    this.cfg = cfg; this.W = W; this.meta = meta;
    this.C = cfg.n_embd; this.H = cfg.n_head; this.hs = cfg.n_embd / cfg.n_head;
  }

  // A context: per layer, the keys and values of every position seen so far.
  newState() {
    return { pos: 0, ids: [], k: Array.from({ length: this.cfg.n_layer }, () => []), v: Array.from({ length: this.cfg.n_layer }, () => []) };
  }

  // Run one character through all layers at position state.pos; returns next-char logits.
  step(state, id) {
    const { C, H, hs, W } = this;
    // token embedding + position embedding (Lesson 3 D)
    let x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = W["tok_emb.weight"][id * C + i] + W["pos_emb.weight"][state.pos * C + i];

    for (let l = 0; l < this.cfg.n_layer; l++) {
      const p = `blocks.${l}.`;
      // --- attention: x = x + proj(attend(ln1(x))) ---
      const h = layerNorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"]);
      const qkv = linear(h, W[p + "attn.qkv.weight"], null, 3 * C);
      const q = qkv.subarray(0, C);
      state.k[l].push(qkv.slice(C, 2 * C));
      state.v[l].push(qkv.slice(2 * C, 3 * C));
      const keys = state.k[l], vals = state.v[l], T = keys.length;  // only the past and present: causal by construction
      const y = new Float32Array(C), scale = 1 / Math.sqrt(hs), sc = new Float32Array(T);
      for (let hd = 0; hd < H; hd++) {
        const o = hd * hs;
        for (let t = 0; t < T; t++) { let s = 0; for (let i = 0; i < hs; i++) s += q[o + i] * keys[t][o + i]; sc[t] = s * scale; }
        const a = softmax(sc);
        for (let t = 0; t < T; t++) for (let i = 0; i < hs; i++) y[o + i] += a[t] * vals[t][o + i];
      }
      const att = linear(y, W[p + "attn.proj.weight"], null, C);
      for (let i = 0; i < C; i++) x[i] += att[i];

      // --- feed-forward: x = x + W2 gelu(W1 ln2(x)) ---
      const h2 = layerNorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"]);
      const f1 = linear(h2, W[p + "ff.net.0.weight"], W[p + "ff.net.0.bias"], 4 * C);
      for (let i = 0; i < f1.length; i++) f1[i] = gelu(f1[i]);
      const f2 = linear(f1, W[p + "ff.net.2.weight"], W[p + "ff.net.2.bias"], C);
      for (let i = 0; i < C; i++) x[i] += f2[i];
    }
    state.pos++; state.ids.push(id);
    const xf = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"]);
    return linear(xf, W["head.weight"], null, this.cfg.vocab_size);
  }

  // Feed several ids; returns the logits after the last one. If the context would pass
  // block_size (128), start a fresh context from the most recent 127 characters, exactly like
  // model.generate's idx[:, -block_size:] (positions must restart at 0).
  feed(state, ids) {
    let logits;
    for (const id of ids) {
      if (state.pos >= this.cfg.block_size) {
        const keep = state.ids.slice(-(this.cfg.block_size - 1));
        Object.assign(state, this.newState());
        for (const k of keep) this.step(state, k);
      }
      logits = this.step(state, id);
    }
    return logits;
  }

  // Predict, pick, repeat. promptIds must be non-empty (the page sends "\n" for free writing).
  // rand() in [0,1) so callers can seed it.
  generate(promptIds, { maxNew = 120, temperature = 0.8, stopId = null, rand = Math.random } = {}) {
    const state = this.newState();
    let logits = this.feed(state, promptIds);
    const out = [];
    for (let n = 0; n < maxNew; n++) {
      const p = softmax(logits, temperature);
      let r = rand(), id = p.length - 1;
      for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) { id = i; break; } }
      out.push(id);
      if (id === stopId) break;
      logits = this.feed(state, [id]);
    }
    return out;
  }
}
