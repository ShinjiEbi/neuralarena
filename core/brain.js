// ================================================================
//  BRAIN — Neural Network (game-agnostic)
// ================================================================
class Brain {
  constructor(topo = [8, 12, 4]) {
    this.topo = topo; this.w = []; this.b = []; this.acts = []; this.fitness = 0;
    for (let i = 0; i < topo.length - 1; i++) {
      const R = topo[i + 1], C = topo[i], s = Math.sqrt(2 / (C + R));
      const wl = [], bl = [];
      for (let r = 0; r < R; r++) {
        const row = [];
        for (let c = 0; c < C; c++) row.push((Math.random() * 2 - 1) * s);
        wl.push(row); bl.push((Math.random() * 2 - 1) * 0.05);
      }
      this.w.push(wl); this.b.push(bl);
    }
  }

  forward(inp) {
    let cur = inp; this.acts = [cur.slice()];
    for (let l = 0; l < this.w.length; l++) {
      const nx = [], wl = this.w[l], bl = this.b[l], isOut = l === this.w.length - 1;
      for (let j = 0; j < wl.length; j++) {
        let s = bl[j];
        for (let i = 0; i < cur.length; i++) s += wl[j][i] * cur[i];
        nx.push(isOut ? 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, s)))) : (s > 0 ? s : s * 0.01));
      }
      cur = nx; this.acts.push(cur.slice());
    }
    return cur;
  }

  clone() {
    const b = new Brain(this.topo);
    for (let l = 0; l < this.w.length; l++) {
      for (let r = 0; r < this.w[l].length; r++) {
        for (let c = 0; c < this.w[l][r].length; c++) b.w[l][r][c] = this.w[l][r][c];
        b.b[l][r] = this.b[l][r];
      }
    }
    b.fitness = this.fitness; return b;
  }

  mutate(rate = 0.15, power = 0.35) {
    for (let l = 0; l < this.w.length; l++)
      for (let r = 0; r < this.w[l].length; r++) {
        for (let c = 0; c < this.w[l][r].length; c++)
          if (Math.random() < rate) {
            const u1 = Math.random(), u2 = Math.random();
            this.w[l][r][c] += Math.sqrt(-2 * Math.log(u1 || 1e-4)) * Math.cos(2 * Math.PI * u2) * power;
          }
        if (Math.random() < rate) this.b[l][r] += (Math.random() * 2 - 1) * power * 0.5;
      }
  }

  static crossover(a, b) {
    const ch = a.clone();
    for (let l = 0; l < ch.w.length; l++) {
      const cut = Math.floor(Math.random() * ch.w[l].length);
      for (let r = cut; r < ch.w[l].length; r++) {
        for (let c = 0; c < ch.w[l][r].length; c++) ch.w[l][r][c] = b.w[l][r][c];
        ch.b[l][r] = b.b[l][r];
      }
    }
    return ch;
  }

  serialize() { return { topo: this.topo, w: this.w, b: this.b }; }

  static deserialize(d) {
    const b = new Brain(d.topo);
    b.w = d.w; b.b = d.b; return b;
  }

  paramCount() {
    let n = 0;
    for (let l = 0; l < this.w.length; l++)
      n += this.w[l].length * this.w[l][0].length + this.b[l].length;
    return n;
  }
}

// ================================================================
//  EVOLUTION ENGINE (game-agnostic)
// ================================================================
class Evolver {
  constructor() {
    this.bestBrain = null;
    this.hallOfFame = [];
    this.maxHof = 15;
  }

  evolve(brain, aiWon) {
    if (aiWon) {
      brain.fitness = (brain.fitness || 0) + 10;
      if (!this.bestBrain || brain.fitness >= (this.bestBrain.fitness || 0)) {
        this.bestBrain = brain.clone();
        this.bestBrain.fitness = brain.fitness;
      }
      this.hallOfFame.push(brain.clone());
      if (this.hallOfFame.length > this.maxHof) this.hallOfFame.shift();
      brain.mutate(0.05, 0.1);
      return brain;
    } else {
      brain.fitness = Math.max(0, (brain.fitness || 0) - 3);
      let newBrain;
      if (this.bestBrain && Math.random() < 0.5) {
        newBrain = Brain.crossover(brain, this.bestBrain);
        newBrain.mutate(0.2, 0.35);
      } else if (this.hallOfFame.length > 1 && Math.random() < 0.3) {
        const a = this.hallOfFame[Math.floor(Math.random() * this.hallOfFame.length)];
        const b = this.hallOfFame[Math.floor(Math.random() * this.hallOfFame.length)];
        newBrain = Brain.crossover(a, b);
        newBrain.mutate(0.15, 0.3);
      } else {
        newBrain = brain;
        newBrain.mutate(0.25, 0.45);
      }
      return newBrain;
    }
  }

  loadBest(brainData) {
    if (brainData) {
      this.bestBrain = Brain.deserialize(brainData);
      this.bestBrain.fitness = brainData.fitness || 0;
    }
  }
}

// ================================================================
//  INDEXEDDB STORE (game-agnostic)
// ================================================================
class Store {
  constructor() { this.db = null; this.ready = this._init(); }

  _init() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('NeuralArena', 3);
      r.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('brains')) db.createObjectStore('brains', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('config')) db.createObjectStore('config', { keyPath: 'key' });
      };
      r.onsuccess = e => { this.db = e.target.result; res(); };
      r.onerror = e => rej(e);
    });
  }

  async _tx(store, mode, fn) {
    await this.ready;
    return new Promise((res, rej) => {
      const tx = this.db.transaction(store, mode);
      const s = tx.objectStore(store);
      const r = fn(s);
      r.onsuccess = () => res(r.result);
      r.onerror = e => rej(e);
    });
  }

  async put(store, data) { return this._tx(store, 'readwrite', s => s.put(data)); }
  async get(store, key) { return this._tx(store, 'readonly', s => s.get(key)); }
  async getAll(store) { return this._tx(store, 'readonly', s => s.getAll()); }
  async del(store, key) { return this._tx(store, 'readwrite', s => s.delete(key)); }
  async setConfig(k, v) { return this.put('config', { key: k, val: v }); }
  async getConfig(k, def = null) { const r = await this.get('config', k); return r ? r.val : def; }
}

// ================================================================
//  GITHUB SYNC (game-agnostic)
// ================================================================
class GitHubSync {
  constructor(store) {
    this.store = store;
    this.repo = ''; this.token = ''; this.pseudo = '';
    this.connected = false; this.brains = []; this.lastSync = null;
  }

  async loadConfig() {
    this.repo = await this.store.getConfig('github_repo', 'shinjiebi/neuralarena') || 'shinjiebi/neuralarena';
    this.token = await this.store.getConfig('github_token', '') || '';
    this.pseudo = await this.store.getConfig('pseudo', 'Anonyme') || 'Anonyme';
  }

  async saveConfig(repo, token, pseudo) {
    this.repo = repo; this.token = token; this.pseudo = pseudo;
    await this.store.setConfig('github_repo', repo);
    await this.store.setConfig('github_token', token);
    await this.store.setConfig('pseudo', pseudo);
  }

  async sync() {
    if (!this.repo) return { ok: false, msg: 'Pas de repo configuré' };
    try {
      const headers = { Accept: 'application/vnd.github.v3+json' };
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      const res = await fetch(`https://api.github.com/repos/${this.repo}/contents/brains`, { headers });
      if (!res.ok) {
        if (res.status === 404) { this.brains = []; this.connected = true; return { ok: true, msg: 'Dossier /brains/ vide', count: 0 }; }
        throw new Error(`GitHub ${res.status}`);
      }
      const files = await res.json();
      this.brains = [];
      await Promise.all(files.filter(f => f.name.endsWith('.json')).map(async f => {
        try { const r = await fetch(f.download_url); const d = await r.json(); d._filename = f.name; d._sha = f.sha; this.brains.push(d); } catch (e) { }
      }));
      this.brains.sort((a, b) => (b.fitness || 0) - (a.fitness || 0));
      this.connected = true; this.lastSync = new Date();
      return { ok: true, msg: `${this.brains.length} cerveaux`, count: this.brains.length };
    } catch (e) {
      return { ok: false, msg: e.message };
    }
  }

  async upload(brainData) {
    if (!this.repo || !this.token) return { ok: false, msg: 'Token requis' };
    try {
      const filename = `${this.pseudo}_gen${brainData.generation}_f${Math.round(brainData.fitness)}_${Date.now().toString(36)}.json`;
      const content = btoa(unescape(encodeURIComponent(JSON.stringify(brainData, null, 2))));
      const res = await fetch(`https://api.github.com/repos/${this.repo}/contents/brains/${filename}`, {
        method: 'PUT',
        headers: { 'Authorization': `Bearer ${this.token}`, 'Content-Type': 'application/json', 'Accept': 'application/vnd.github.v3+json' },
        body: JSON.stringify({ message: `🧠 ${this.pseudo} — Gen ${brainData.generation}, Fitness ${Math.round(brainData.fitness)}`, content, branch: 'main' })
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.message || res.statusText); }
      return { ok: true };
    } catch (e) {
      return { ok: false, msg: e.message };
    }
  }

  async test() {
    if (!this.repo) return { ok: false, msg: 'Aucun repo' };
    try {
      const headers = { Accept: 'application/vnd.github.v3+json' };
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      const res = await fetch(`https://api.github.com/repos/${this.repo}`, { headers });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      let canWrite = false;
      if (this.token) { const u = await fetch('https://api.github.com/user', { headers: { Authorization: `Bearer ${this.token}`, Accept: 'application/vnd.github.v3+json' } }); canWrite = u.ok; }
      return { ok: true, msg: `✓ ${data.full_name} — ${canWrite ? '✓ Écriture OK' : '⚠ Lecture seule'}`, canWrite };
    } catch (e) {
      return { ok: false, msg: '✗ ' + e.message };
    }
  }
}

// Export for use
window.NeuralArena = window.NeuralArena || {};
Object.assign(window.NeuralArena, { Brain, Evolver, Store, GitHubSync });
