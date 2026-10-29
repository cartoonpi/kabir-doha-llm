// Runs the doha model off the main thread, so the page stays responsive on phones.
import { fetchModel } from "./lib/load.js";
import { makeWriter } from "./lib/writer.js";

const ready = fetchModel("./models/", "doha").then(({ meta, weights }) => makeWriter(meta, weights));
ready.then(() => postMessage({ type: "ready" }), (e) => postMessage({ type: "error", message: e.message }));

onmessage = async ({ data }) => {
  try {
    const w = await ready;
    postMessage({ type: "doha", id: data.id, result: w.write(data.tags) });
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: e.message });
  }
};
