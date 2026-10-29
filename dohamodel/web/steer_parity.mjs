// Check the JS forward pass (gpt.js) on the tag-steered models against PyTorch (export_tags.py),
// and the JS matra checker (matra.js) against Python's matra.grade on real and synthetic dohe.
// Run from dohamodel/:  node web/steer_parity.mjs tags_5k tags_2500
import { readFile } from "node:fs/promises";
import { loadModel } from "./gpt.js";
import { grade } from "./matra.js";

const here = new URL(".", import.meta.url);
globalThis.fetch = async (url) => {
  const buf = await readFile(new URL(url, here));
  return { ok: true, json: async () => JSON.parse(buf.toString("utf8")), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

let failed = false;
for (const name of process.argv.slice(2)) {
  const meta = JSON.parse(await readFile(new URL(`models/${name}/meta.json`, here), "utf8"));
  const m = await loadModel(`models/${name}/`, "final", meta);
  for (const { ids, logits } of meta.parity) {
    const js = m.feed(m.newState(), ids);
    const diff = Math.max(...logits.map((v, i) => Math.abs(v - js[i])));
    if (diff >= 1e-3) failed = true;
    console.log(`${diff < 1e-3 ? "ok  " : "FAIL"} ${name} prompt [${ids.map((i) => meta.itos[i]).join("").replace(/\n/g, "⏎")}]  max |diff| ${diff.toExponential(1)}`);
  }
}

const cases = JSON.parse(await readFile(new URL("matra_cases.json", here), "utf8"));
const bad = cases.filter(({ text, grade: g }) => grade(text) !== g);
console.log(`matra.js vs matra.py: ${cases.length - bad.length}/${cases.length} agree`);
for (const b of bad.slice(0, 5)) console.log("  differs:", JSON.stringify(b.text), "python", b.grade, "js", grade(b.text));
if (bad.length) failed = true;
process.exit(failed ? 1 : 0);
