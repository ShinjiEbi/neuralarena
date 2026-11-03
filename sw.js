// Incrémenter à chaque déploiement qui change la liste ASSETS
const CACHE_NAME = 'neuralarena-v26';
const ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/icon-maskable-512.png',
];
// Polices Google : mises en cache pour que l'app garde son style hors-ligne
const FONT_HOSTS = ['fonts.googleapis.com', 'fonts.gstatic.com'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Polices : cache d'abord (elles ne changent pas)
  if (FONT_HOSTS.includes(url.hostname)) {
    e.respondWith(
      caches.match(req).then(hit => hit || fetch(req).then(res => {
        if (res.ok || res.type === 'opaque') {
          const copy = res.clone();
          e.waitUntil(caches.open(CACHE_NAME).then(c => c.put(req, copy)));
        }
        return res;
      }))
    );
    return;
  }

  // Autres origines (API GitHub, fichiers de cerveaux) : jamais mises en cache
  if (url.origin !== self.location.origin) return;

  // Fichiers de l'app : réseau d'abord, cache en secours hors-ligne
  e.respondWith(
    fetch(req).then(res => {
      if (res.ok && res.type === 'basic') {
        const copy = res.clone();
        e.waitUntil(caches.open(CACHE_NAME).then(c => c.put(req, copy)).catch(() => {}));
      }
      return res;
    }).catch(async () => {
      const hit = await caches.match(req, { ignoreSearch: req.mode === 'navigate' });
      if (hit) return hit;
      if (req.mode === 'navigate') return (await caches.match('./index.html')) || Response.error();
      return Response.error();
    })
  );
});
