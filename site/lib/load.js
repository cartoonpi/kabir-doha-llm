// Fetch a model's JSON + float16 weights and widen the weights to a float32 ArrayBuffer.
// Weights ship as float16 to halve the download; the models compute in float32 as trained.

const F16 = (() => {                       // all 65,536 half-floats as float32, built once
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function widen(buf) {
  const h = new Uint16Array(buf), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = F16[h[i]];
  return out;
}

export async function fetchModel(base, name) {
  const [meta, buf] = await Promise.all([
    fetch(`${base}${name}.json`).then((r) => { if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`); return r.json(); }),
    fetch(`${base}${name}.bin`).then((r) => { if (!r.ok) throw new Error(`${name}.bin: HTTP ${r.status}`); return r.arrayBuffer(); }),
  ]);
  const w = widen(buf);
  if (w.length !== meta.floats) throw new Error(`${name}.bin: expected ${meta.floats} weights, got ${w.length}`);
  return { meta, weights: w };
}
