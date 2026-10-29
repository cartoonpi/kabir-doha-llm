// Checks the built site (../site) in Node before deploying:
//  1. doha model: float16 weights vs PyTorch's float32 logits (meta.parity)
//  2. classifier: float16 vs the original float32 weights give the same tags on real requests
//  3. end to end: the writer turns tags into dohe (prints a few, with timing)
// Run from dohellm/:  node dohamodel/site_check.mjs
import { readFile } from "node:fs/promises";
import { widen } from "../site/lib/load.js";
import { makeWriter } from "../site/lib/writer.js";
import { create } from "../site/lib/classifier.js";
import { grade } from "../site/lib/matra.js";

const rd = (p) => readFile(new URL(p, import.meta.url));
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
let failed = false;

// 1
const meta = JSON.parse(await rd("../site/models/doha.json"));
const w = makeWriter(meta, widen(ab(await rd("../site/models/doha.bin"))));
for (const { ids, logits } of meta.parity) {
  const js = w.model.feed(w.model.newState(), ids);
  const diff = Math.max(...logits.map((v, i) => Math.abs(v - js[i])));
  const top = (a) => a.indexOf(Math.max(...a));
  const ok = diff < 0.05 && top(logits) === top([...js]);
  if (!ok) failed = true;
  console.log(`${ok ? "ok  " : "FAIL"} doha fp16 logits, max |diff| ${diff.toExponential(1)}, same top token: ${top(logits) === top([...js])}`);
}

// 2
const cm = JSON.parse(await rd("../site/models/classifier.json"));
const c16 = create(cm, widen(ab(await rd("../site/models/classifier.bin"))).buffer);
const c32 = create(JSON.parse(await rd("../clasifierLM/classifier/artifacts/classifier/meta.json")), ab(await rd("../clasifierLM/classifier/artifacts/classifier/weights.bin")));
const reqs = [];
for (const f of ["requests_human.jsonl", "requests_synth.jsonl"])
  for (const l of (await rd(`../clasifierLM/classifier/data/${f}`)).toString().split("\n")) if (l.trim()) reqs.push(JSON.parse(l).text);
let same = 0;
for (const t of reqs) {
  const a = c16.predict(t), b = c32.predict(t);
  if (a.theme === b.theme && a.images.map((x) => x.id).join() === b.images.map((x) => x.id).join()) same++;
}
console.log(`${same / reqs.length > 0.995 ? "ok  " : "FAIL"} classifier fp16 vs fp32: same tags on ${same}/${reqs.length} requests`);
if (same / reqs.length <= 0.995) failed = true;

// 3
const cases = [{ theme: "ahankar", images: ["deepak"], device: "kahekabir" }, { theme: "vani", images: ["koyal"], device: "sadho" },
  { theme: "guru", images: [], device: "prashn" }, { theme: "maya", images: ["kasturi"], device: "ulatbansi" }];
let tries = 0, ms = 0, good = 0;
for (let r = 0; r < 3; r++) for (const t of cases) {
  const t0 = performance.now(), res = w.write(t);
  ms += performance.now() - t0;
  if (!res) { console.log("  (no usable doha)", JSON.stringify(t)); continue; }
  tries += res.tries;
  if (grade(res.text)) good++;
  if (r === 0) console.log(`  ${t.theme}/${t.images.join(",") || "-"}/${t.device}: ${res.text.replace("\n", " / ")}`);
}
console.log(`writer: ${good}/12 scan (exact or near), avg ${(tries / 12).toFixed(1)} tries, ${(ms / 12 / 1000).toFixed(2)} s per doha`);
process.exit(failed ? 1 : 0);
