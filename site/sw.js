// Offline support: after the first visit, the page and both models load from the device.
// VERSION is stamped by dohamodel/build_site.py; a new version replaces the old cache.
const VERSION = "83ecb2fc99";
const FILES = ["./", "index.html", "app.js", "worker.js", "lib/load.js", "lib/writer.js", "lib/gpt.js", "lib/matra.js",
  "lib/classifier.js", "models/doha.json", "models/doha.bin", "models/classifier.json", "models/classifier.bin",
  "fonts/serif-Regular.woff2", "fonts/serif-SemiBold.woff2", "fonts/display-Medium.woff2", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
});
