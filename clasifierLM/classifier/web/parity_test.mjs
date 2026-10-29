// Step 8.3: check that JS gives the same answers as Python.
//
//   node web/parity_test.mjs        (run from the classifier/ folder, after `python -m src.export`)
//
// For each case in artifacts/parity.json (written by Python):
//   - bucket ids must match EXACTLY (any difference = a featurization bug)
//   - probabilities must match within 1e-5 (float32) / 1e-2 (int8)
//   - final decisions (theme + images) must match (float32 only: int8 rounding
//     may legitimately flip a decision that sits right at a threshold)
// Exits with code 1 on any mismatch, so it can gate a build.

import { readFileSync, existsSync } from "node:fs";
import { create } from "./classifier.js";

const root = new URL("../artifacts/", import.meta.url);
const parity = JSON.parse(readFileSync(new URL("parity.json", root), "utf8"));

function loadDir(dir) {
  const meta = JSON.parse(readFileSync(new URL(dir + "/meta.json", root), "utf8"));
  const buf = readFileSync(new URL(dir + "/weights.bin", root));
  // Node's Buffer may be a slice of a bigger pool; copy to a standalone ArrayBuffer
  // so byte offsets from meta.json line up.
  return create(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

let failures = 0;
function fail(msg) { failures++; console.error("  FAIL " + msg); }

function check(dir, tol, checkDecisions) {
  if (!existsSync(new URL(dir + "/meta.json", root))) { console.log(`skip ${dir} (not exported)`); return; }
  const clf = loadDir(dir);
  let maxDiff = 0, flips = 0;
  console.log(`\n== ${dir}: ${parity.cases.length} cases, tolerance ${tol} ==`);
  for (const c of parity.cases) {
    const r = clf.predict(c.text);
    const label = JSON.stringify(c.text.length > 40 ? c.text.slice(0, 40) + "…" : c.text);
    if (r.buckets.length !== c.buckets.length || r.buckets.some((b, i) => b !== c.buckets[i])) {
      fail(`${label}: bucket ids differ\n    py ${c.buckets.slice(0, 10)}\n    js ${r.buckets.slice(0, 10)}`);
      continue;
    }
    for (const [kind, py, js] of [["theme", c.theme_probs, r.allThemeProbs], ["image", c.image_probs, r.allImageProbs]]) {
      for (const k of Object.keys(py)) {
        const diff = Math.abs(py[k] - js[k]);
        maxDiff = Math.max(maxDiff, diff);
        if (!(diff <= tol)) fail(`${label}: ${kind} ${k} py ${py[k]} js ${js[k]} (diff ${diff.toExponential(2)})`);
      }
    }
    const same = r.theme === c.decision.theme &&
      JSON.stringify(r.images.map((x) => x.id)) === JSON.stringify(c.decision.images);
    if (!same) {
      if (checkDecisions) fail(`${label}: decision py ${JSON.stringify(c.decision)} js ${r.theme} ${r.images.map((x) => x.id)}`);
      else flips++;
    }
  }
  console.log(`max |py - js| probability difference: ${maxDiff.toExponential(2)}` + (checkDecisions ? "" : ` | decision flips: ${flips}`));
}

check("classifier", 1e-5, true);
check("classifier_int8", 1e-2, false);

if (failures) {
  console.error(`\nPARITY FAILED: ${failures} mismatch(es)`);
  process.exit(1);
}
console.log("\nPARITY OK");
