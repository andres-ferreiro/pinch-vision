/**
 * Offline shell + a warm start for the hand tracker.
 *
 * The app's own files use network-first, so a deploy is never masked by a stale
 * cache. The CDN payload — MediaPipe's WASM and the ~8 MB landmark model — is
 * immutable and version-pinned, so it is cache-first: after one visit the
 * tracker starts without re-downloading it, which is the difference between a
 * six second wait and an instant one on a phone.
 */

const VERSION = 'pv-2026-09-23';
const SHELL = [
  './',
  './index.html',
  './style.css',
  './app.js?v=10',
  './effects.js?v=10',
  './manifest.webmanifest',
  './icon-192.png',
  './icon-512.png',
];

const IMMUTABLE = ['cdn.jsdelivr.net', 'storage.googleapis.com'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION)
      // Individually, so one failed asset cannot abort the whole install.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  if (IMMUTABLE.some((host) => url.hostname.endsWith(host))) {
    event.respondWith(
      caches.match(request).then((hit) => hit || fetch(request).then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy));
        }
        return response;
      })),
    );
    return;
  }

  if (url.origin !== self.location.origin) return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request).then((hit) => hit || caches.match('./index.html'))),
  );
});
