// Check the JS forward pass (gpt.js) against PyTorch's numbers saved by export.py.
// Run from dohamodel/:  node web/parity_test.mjs small big
import { readFile } from "node:fs/promises";
import { loadModel } from "./gpt.js";

const here = new URL(".", import.meta.url);
globalThis.fetch = async (url) => {                      // serve files from disk instead of HTTP
  const buf = await readFile(new URL(url, here));
  return { ok: true, json: async () => JSON.parse(buf.toString("utf8")), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

let worst = 0, failed = false;
for (const name of process.argv.slice(2)) {
  const meta = JSON.parse(await readFile(new URL(`models/${name}/meta.json`, here), "utf8"));
  const stoi = Object.fromEntries(meta.itos.map((c, i) => [c, i]));
  for (const step of meta.snapshots) {
    const m = await loadModel(`models/${name}/`, step, meta);
    for (const { prompt, logits } of meta.parity[step]) {
      const js = m.feed(m.newState(), [...prompt].map((c) => stoi[c]));
      const diff = Math.max(...logits.map((v, i) => Math.abs(v - js[i])));
      worst = Math.max(worst, diff);
      const ok = diff < 1e-3;
      if (!ok) failed = true;
      console.log(`${ok ? "ok  " : "FAIL"} ${name} step ${step} prompt ${JSON.stringify(prompt).slice(0, 20)}  max |diff| ${diff.toExponential(1)}`);
    }
  }
}
console.log(`worst difference: ${worst.toExponential(2)}`);
process.exit(failed ? 1 : 0);
