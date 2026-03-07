// ================================================================
//  GAME REGISTRY — Register and switch between games
// ================================================================
const GameRegistry = {
  games: {},

  register(gameModule) {
    this.games[gameModule.id] = gameModule;
    console.log(`[NeuralArena] Game registered: ${gameModule.id} (${gameModule.name})`);
  },

  get(id) { return this.games[id] || null; },
  list() { return Object.values(this.games); }
};

// ================================================================
//  INPUT SYSTEM — Keyboard + Dual Joystick
// ================================================================
const Input = {
  keys: {},
  joy: {
    left: { active: false, x: 0, y: 0, touchId: null },
    right: { active: false, x: 0, y: 0, touchId: null },
    jump: false
  },
  DEADZONE: 0.18,

  init() {
    window.addEventListener('keydown', e => {
      this.keys[e.code] = true;
      if (['Space', 'ArrowUp', 'ArrowDown', 'KeyW'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', e => { this.keys[e.code] = false; });
  },

  setupJoystick(zoneId, knobId, stateKey) {
    const zone = document.getElementById(zoneId);
    const knob = document.getElementById(knobId);
    if (!zone || !knob) return;
    const base = knob.parentElement;
    const maxDist = 42;
    const self = this;

    function getBaseCenter() {
      const r = base.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }

    function processTouch(cx, cy) {
      const center = getBaseCenter();
      let dx = cx - center.x, dy = cy - center.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxDist) { dx = dx / dist * maxDist; dy = dy / dist * maxDist; }
      self.joy[stateKey].x = dx / maxDist;
      self.joy[stateKey].y = dy / maxDist;
      self.joy[stateKey].active = true;
      knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    }

    function reset() {
      self.joy[stateKey].x = 0; self.joy[stateKey].y = 0;
      self.joy[stateKey].active = false; self.joy[stateKey].touchId = null;
      knob.style.transform = 'translate(-50%,-50%)';
    }

    zone.addEventListener('touchstart', e => {
      e.preventDefault();
      for (const t of e.changedTouches) {
        if (e.target.id === 'jump-btn' || e.target.closest('.jump-btn')) continue;
        if (self.joy[stateKey].touchId === null) {
          self.joy[stateKey].touchId = t.identifier;
          processTouch(t.clientX, t.clientY);
        }
      }
    }, { passive: false });

    zone.addEventListener('touchmove', e => {
      e.preventDefault();
      for (const t of e.changedTouches)
        if (t.identifier === self.joy[stateKey].touchId) processTouch(t.clientX, t.clientY);
    }, { passive: false });

    const end = e => { for (const t of e.changedTouches) if (t.identifier === self.joy[stateKey].touchId) reset(); };
    zone.addEventListener('touchend', end, { passive: false });
    zone.addEventListener('touchcancel', end, { passive: false });
  },

  setupJumpButton() {
    const btn = document.getElementById('jump-btn');
    if (!btn) return;
    btn.addEventListener('touchstart', e => { e.preventDefault(); e.stopPropagation(); this.joy.jump = true; btn.classList.add('held'); }, { passive: false });
    const end = e => { e.preventDefault(); this.joy.jump = false; btn.classList.remove('held'); };
    btn.addEventListener('touchend', end, { passive: false });
    btn.addEventListener('touchcancel', end, { passive: false });
  },

  setupAllTouch() {
    this.setupJoystick('joy-left', 'knob-left', 'left');
    this.setupJoystick('joy-right', 'knob-right', 'right');
    this.setupJumpButton();
  }
};

// ================================================================
//  CAMERA
// ================================================================
const Camera = {
  x: 0, y: 0,
  targetX: 0, targetY: 0,
  smoothing: 0.08,

  follow(entity, viewW, viewH, worldW, worldH) {
    if (!entity) return;
    this.targetX = entity.x + (entity.w || 0) / 2 - viewW / 2 + (entity.vx || 0) * 18;
    this.targetY = entity.y + (entity.h || 0) / 2 - viewH / 2 + (entity.vy || 0) * 6;
    this.targetX = Math.max(0, Math.min(worldW - viewW, this.targetX));
    this.targetY = Math.max(0, Math.min(worldH - viewH, this.targetY));
    this.x += (this.targetX - this.x) * this.smoothing;
    this.y += (this.targetY - this.y) * (this.smoothing + 0.02);
  },

  reset(x, y) { this.x = x; this.y = y; this.targetX = x; this.targetY = y; }
};

// ================================================================
//  GAME LOOP MANAGER
// ================================================================
const Engine = {
  canvas: null,
  ctx: null,
  nnCanvas: null,
  nnCtx: null,
  viewW: 0, viewH: 0, dpr: 1,
  activeGame: null,
  running: false,

  init() {
    this.canvas = document.getElementById('game');
    this.ctx = this.canvas.getContext('2d');
    this.nnCanvas = document.getElementById('game-nn');
    this.nnCtx = this.nnCanvas.getContext('2d');
    this.canvas.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
    Input.init();
  },

  resize() {
    this.dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.viewW = r.width; this.viewH = r.height;
    this.canvas.width = this.viewW * this.dpr;
    this.canvas.height = this.viewH * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  },

  start(gameId) {
    const game = GameRegistry.get(gameId);
    if (!game) return console.error('Game not found:', gameId);
    this.activeGame = game;
    this.resize();
    Input.setupAllTouch();
    this.running = true;
    this._loop();
  },

  stop() {
    this.running = false;
    this.activeGame = null;
  },

  _loop() {
    if (!this.running || !this.activeGame) return;
    requestAnimationFrame(() => this._loop());
    this.activeGame.frame(this.ctx, this.viewW, this.viewH);
    this.drawNN();
  },

  drawNN() {
    const brain = window.NeuralArena?.activeBrain;
    if (!brain || !brain.acts || !brain.acts.length) return;
    const w = this.nnCanvas.width, h = this.nnCanvas.height, c = this.nnCtx;
    c.clearRect(0, 0, w, h);
    const ls = brain.acts, mx = 8;
    for (let l = 0; l < ls.length; l++) {
      const ns = ls[l], dc = Math.min(ns.length, mx);
      const lx = 12 + (l / (ls.length - 1)) * (w - 24);
      for (let n = 0; n < dc; n++) {
        const ny = 6 + (dc > 1 ? (n / (dc - 1)) * (h - 12) : h / 2 - 6);
        const v = ns[n];
        if (l < ls.length - 1) {
          const nc = Math.min(ls[l + 1].length, mx);
          const nlx = 12 + ((l + 1) / (ls.length - 1)) * (w - 24);
          for (let nn = 0; nn < nc; nn++) {
            const nny = 6 + (nc > 1 ? (nn / (nc - 1)) * (h - 12) : h / 2 - 6);
            const wt = brain.w[l]?.[nn]?.[n] || 0;
            const a = Math.min(Math.abs(wt) * 0.25, 0.3);
            c.strokeStyle = wt > 0 ? `rgba(59,240,160,${a})` : `rgba(255,68,102,${a})`;
            c.lineWidth = 0.5; c.beginPath(); c.moveTo(lx, ny); c.lineTo(nlx, nny); c.stroke();
          }
        }
        const abs = Math.min(Math.abs(v), 2) / 2;
        c.fillStyle = v > 0 ? `rgba(59,240,160,${0.2 + abs * 0.8})` : `rgba(255,68,102,${0.2 + abs * 0.8})`;
        c.beginPath(); c.arc(lx, ny, 2 + abs * 2, 0, Math.PI * 2); c.fill();
      }
    }
  }
};

// ================================================================
//  PARTICLES (shared utility)
// ================================================================
const Particles = {
  list: [],

  spawn(x, y, color, count = 30) {
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2, s = 1 + Math.random() * 6;
      this.list.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 2, life: 30 + Math.random() * 30, col: color, sz: 1 + Math.random() * 3 });
    }
  },

  update() {
    for (let i = this.list.length - 1; i >= 0; i--) {
      const p = this.list[i];
      p.x += p.vx; p.y += p.vy; p.vy += 0.1; p.life--;
      if (p.life <= 0) this.list.splice(i, 1);
    }
  },

  draw(ctx) {
    for (const p of this.list) {
      ctx.globalAlpha = p.life / 60;
      ctx.fillStyle = p.col;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.sz, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  },

  clear() { this.list = []; }
};

// roundRect polyfill
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    if (typeof r === 'number') r = [r, r, r, r];
    this.moveTo(x + r[0], y); this.lineTo(x + w - r[1], y);
    this.quadraticCurveTo(x + w, y, x + w, y + r[1]);
    this.lineTo(x + w, y + h - r[2]);
    this.quadraticCurveTo(x + w, y + h, x + w - r[2], y + h);
    this.lineTo(x + r[3], y + h);
    this.quadraticCurveTo(x, y + h, x, y + h - r[3]);
    this.lineTo(x, y + r[0]);
    this.quadraticCurveTo(x, y, x + r[0], y); this.closePath();
  };
}

// Export
window.NeuralArena = window.NeuralArena || {};
Object.assign(window.NeuralArena, { GameRegistry, Input, Camera, Engine, Particles });
