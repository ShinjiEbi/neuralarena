// ================================================================
//  ARENA ARCHER — Game Module
//  To add a new game, copy this file and change the implementation
// ================================================================
(function () {
  const { Brain, GameRegistry, Input, Camera, Particles } = window.NeuralArena;
  const NA = window.NeuralArena;

  // === CONSTANTS ===
  const WW = 1400, WH = 700;
  const GRAV = 0.38, FRIC = 0.82, JUMP = -10, MSPD = 3.5;
  const ASPD = 13, AGRAV = 0.18, ADRAG = 0.998, MAXA = 5, COOLDOWN = 26;
  const DEADZONE = 0.18;

  const PLATFORMS = [
    { x: 0, y: WH - 30, w: WW, h: 30 }, { x: 0, y: 0, w: WW, h: 10 },
    { x: 0, y: 0, w: 12, h: WH }, { x: WW - 12, y: 0, w: 12, h: WH },
    { x: 60, y: 560, w: 180, h: 16 }, { x: 30, y: 440, w: 130, h: 16 },
    { x: 180, y: 350, w: 160, h: 16 }, { x: 40, y: 260, w: 120, h: 16 },
    { x: 200, y: 180, w: 140, h: 16 },
    { x: 420, y: 520, w: 200, h: 16 }, { x: 550, y: 410, w: 180, h: 16 },
    { x: 380, y: 310, w: 160, h: 16 }, { x: 600, y: 240, w: 200, h: 16 },
    { x: 480, y: 140, w: 160, h: 16 },
    { x: 870, y: 560, w: 180, h: 16 }, { x: 1000, y: 440, w: 160, h: 16 },
    { x: 1180, y: 350, w: 150, h: 16 }, { x: 1050, y: 260, w: 140, h: 16 },
    { x: 1200, y: 180, w: 140, h: 16 },
    { x: 300, y: 480, w: 120, h: 14 }, { x: 760, y: 480, w: 120, h: 14 },
    { x: 830, y: 330, w: 100, h: 14 }, { x: 700, y: 140, w: 100, h: 14 },
    { x: 900, y: 180, w: 120, h: 14 },
  ];

  // === ENTITY ===
  class Ent {
    constructor(x, y, col, isAI) {
      this.x = x; this.y = y; this.vx = 0; this.vy = 0;
      this.w = 20; this.h = 32; this.col = col; this.ai = isAI;
      this.gnd = false; this.aim = 0; this.face = isAI ? -1 : 1;
      this.arr = []; this.cd = 0; this.ht = 0;
    }

    update() {
      this.vy += GRAV; this.vx *= this.gnd ? FRIC : 0.96;
      this.vx = Math.max(-8, Math.min(8, this.vx));
      this.vy = Math.max(-15, Math.min(15, this.vy));
      this.x += this.vx; this.y += this.vy; this.gnd = false;
      for (const p of PLATFORMS) {
        if (this.x + this.w > p.x && this.x < p.x + p.w && this.y + this.h > p.y && this.y < p.y + p.h) {
          const oL = (this.x + this.w) - p.x, oR = (p.x + p.w) - this.x;
          const oT = (this.y + this.h) - p.y, oB = (p.y + p.h) - this.y;
          const m = Math.min(oL, oR, oT, oB);
          if (m === oT && this.vy >= 0) { this.y = p.y - this.h; this.vy = 0; this.gnd = true; }
          else if (m === oB && this.vy < 0) { this.y = p.y + p.h; this.vy = 0; }
          else if (m === oL) { this.x = p.x - this.w; this.vx = 0; }
          else { this.x = p.x + p.w; this.vx = 0; }
        }
      }
      if (this.cd > 0) this.cd--;
      if (this.ht > 0) this.ht--;
      for (let i = this.arr.length - 1; i >= 0; i--) {
        const a = this.arr[i];
        if (a.st) { a.life--; if (a.life <= 0) this.arr.splice(i, 1); continue; }
        a.vy += AGRAV; a.vx *= ADRAG; a.vy *= ADRAG;
        a.x += a.vx; a.y += a.vy; a.life--;
        a.tr.push({ x: a.x, y: a.y }); if (a.tr.length > 8) a.tr.shift();
        if (a.x < -30 || a.x > WW + 30 || a.y < -60 || a.y > WH + 30 || a.life <= 0) { this.arr.splice(i, 1); continue; }
        for (const p of PLATFORMS) if (a.x > p.x && a.x < p.x + p.w && a.y > p.y && a.y < p.y + p.h) { a.st = true; a.vx = 0; a.vy = 0; a.life = Math.min(a.life, 50); break; }
      }
    }

    shoot() {
      if (this.cd > 0 || this.arr.length >= MAXA) return;
      const d = this.face;
      this.arr.push({ x: this.x + this.w / 2 + d * 14, y: this.y + 10, vx: d * Math.cos(this.aim) * ASPD + this.vx * 0.3, vy: Math.sin(this.aim) * ASPD + this.vy * 0.2, life: 220, st: false, tr: [] });
      this.cd = COOLDOWN; this.vx -= d * 0.9;
    }

    draw(ctx) {
      const cx = this.x + this.w / 2, d = this.face;
      ctx.save();
      if (this.ht > 0) ctx.globalAlpha = 0.4 + Math.sin(this.ht * 0.5) * 0.3;
      const gc = this.ai ? 'rgba(255,50,80,0.12)' : 'rgba(0,180,255,0.12)';
      const gr = ctx.createRadialGradient(cx, this.y + this.h / 2, 2, cx, this.y + this.h / 2, 35);
      gr.addColorStop(0, gc); gr.addColorStop(1, 'transparent');
      ctx.fillStyle = gr; ctx.fillRect(cx - 35, this.y - 3, 70, 70);
      const la = this.gnd ? Math.sin(Date.now() * 0.012 + (this.ai ? 99 : 0)) * (Math.abs(this.vx) > 0.5 ? 5 : 0) : 4;
      ctx.strokeStyle = this.col; ctx.lineWidth = 2.5; ctx.beginPath();
      ctx.moveTo(cx - 4, this.y + this.h - 5); ctx.lineTo(cx - 6, this.y + this.h + la);
      ctx.moveTo(cx + 4, this.y + this.h - 5); ctx.lineTo(cx + 6, this.y + this.h - la); ctx.stroke();
      ctx.fillStyle = this.col;
      ctx.beginPath(); ctx.roundRect(this.x + 3, this.y + 10, this.w - 6, this.h - 14, 4); ctx.fill();
      ctx.beginPath(); ctx.arc(cx, this.y + 7, 8, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(cx + d * 3.5, this.y + 6, 2.2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(cx + d * 4, this.y + 6, 1.1, 0, Math.PI * 2); ctx.fill();
      const bx = cx + d * 12, by = this.y + 12;
      ctx.save(); ctx.translate(bx, by); ctx.rotate(this.aim * d);
      ctx.strokeStyle = '#c8a050'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(0, 0, 12, -1.2, 1.2); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(Math.cos(-1.2) * 12, Math.sin(-1.2) * 12); ctx.lineTo(0, 0);
      ctx.lineTo(Math.cos(1.2) * 12, Math.sin(1.2) * 12); ctx.stroke(); ctx.restore();
      if (!this.ai) {
        ctx.strokeStyle = 'rgba(0,180,255,0.1)'; ctx.lineWidth = 1; ctx.setLineDash([5, 7]);
        ctx.beginPath(); ctx.moveTo(bx, by);
        ctx.lineTo(bx + d * Math.cos(this.aim) * 70, by + Math.sin(this.aim) * 70);
        ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.restore();
      for (const a of this.arr) {
        for (let t = 0; t < a.tr.length; t++) {
          ctx.fillStyle = this.ai ? `rgba(255,50,80,${(t / a.tr.length) * 0.25})` : `rgba(0,180,255,${(t / a.tr.length) * 0.25})`;
          ctx.beginPath(); ctx.arc(a.tr[t].x, a.tr[t].y, 1.5, 0, Math.PI * 2); ctx.fill();
        }
        const ang = Math.atan2(a.vy || 0.01, a.vx || 0.01);
        ctx.save(); ctx.translate(a.x, a.y); ctx.rotate(ang);
        ctx.strokeStyle = a.st ? '#665533' : (this.ai ? '#ff5566' : '#55aaff');
        ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-14, 0); ctx.lineTo(7, 0); ctx.stroke();
        ctx.fillStyle = a.st ? '#998866' : '#fff';
        ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(5, -3); ctx.lineTo(5, 3); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = this.ai ? 'rgba(255,80,100,0.5)' : 'rgba(80,150,255,0.5)';
        ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(-12, 0); ctx.lineTo(-15, -3); ctx.moveTo(-12, 0); ctx.lineTo(-15, 3); ctx.stroke();
        ctx.restore();
      }
    }
  }

  // === GAME STATE ===
  let P = null, A = null;
  let roundOver = false, roundTimer = 0;

  // === MODULE DEFINITION ===
  const ArenaArcher = {
    // --- Metadata (used by the menu & registry) ---
    id: 'arena-archer',
    name: 'Arena Archer',
    icon: '⚔',
    description: 'Arène de plateformes — Tir à l\'arc 1v1',
    topology: [16, 20, 14, 6],

    // --- Lifecycle ---
    initRound() {
      P = new Ent(200, WH - 100, '#00b4ff', false); P.face = 1;
      A = new Ent(WW - 220, WH - 100, '#ff3250', true); A.face = -1;
      roundOver = false; roundTimer = 0;
      Particles.clear();
      Camera.reset(P.x - NA.Engine.viewW / 2, P.y - NA.Engine.viewH / 2);
    },

    // --- AI interface ---
    getAIInputs(vW, vH) {
      const rX = (P.x - A.x) / vW, rY = (P.y - A.y) / vH;
      let nAX = 0, nAY = 0, nVX = 0, nVY = 0, mD = Infinity;
      for (const a of P.arr) {
        if (a.st) continue;
        const dx = a.x - A.x, dy = a.y - A.y, d = dx * dx + dy * dy;
        if (d < mD) { mD = d; nAX = dx / vW; nAY = dy / vH; nVX = a.vx / ASPD; nVY = a.vy / ASPD; }
      }
      return [rX, rY, A.vx / 8, A.vy / 15, P.vx / 8, P.vy / 15, A.aim / (Math.PI / 2), A.gnd ? 1 : 0,
        nAX, nAY, nVX, nVY, A.x / WW, (WW - A.x) / WW,
        Math.min(roundTimer / (60 * 30), 1), Math.random() * 0.15 - 0.075];
    },

    processAIOutputs(out) {
      if (out[0] > 0.5) { A.vx -= MSPD * 0.4; A.face = -1; }
      if (out[1] > 0.5) { A.vx += MSPD * 0.4; A.face = 1; }
      if (out[2] > 0.5 && A.gnd) A.vy = JUMP;
      if (out[3] > 0.5) A.aim = Math.max(A.aim - 0.05, -Math.PI / 2.2);
      if (out[4] > 0.5) A.aim = Math.min(A.aim + 0.05, Math.PI / 2.5);
      if (out[5] > 0.6) A.shoot();
      A.face = P.x < A.x ? -1 : 1;
    },

    // --- Player input ---
    handleInput() {
      if (!P) return;
      const K = Input.keys, J = Input.joy;
      // Keyboard
      if (K['KeyA'] || K['ArrowLeft']) { P.vx -= MSPD * 0.4; P.face = -1; }
      if (K['KeyD'] || K['ArrowRight']) { P.vx += MSPD * 0.4; P.face = 1; }
      if ((K['KeyW'] || K['KeyZ']) && P.gnd) P.vy = JUMP;
      if (K['ArrowUp']) P.aim = Math.max(P.aim - 0.05, -Math.PI / 2.2);
      if (K['ArrowDown']) P.aim = Math.min(P.aim + 0.05, Math.PI / 2.5);
      if (K['Space']) P.shoot();
      // Left joystick
      if (J.left.active) {
        if (Math.abs(J.left.x) > DEADZONE) { P.vx += J.left.x * MSPD * 0.5; P.face = J.left.x > 0 ? 1 : -1; }
        if (J.left.y < -0.6 && P.gnd) P.vy = JUMP;
      }
      if (J.jump && P.gnd) P.vy = JUMP;
      // Right joystick: aim + auto-fire
      if (J.right.active) {
        const rx = J.right.x, ry = J.right.y, mag = Math.sqrt(rx * rx + ry * ry);
        if (mag > DEADZONE) {
          P.aim = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.5, Math.atan2(ry, Math.abs(rx))));
          if (Math.abs(rx) > 0.15) P.face = rx > 0 ? 1 : -1;
          if (mag > 0.35) P.shoot();
        }
      }
    },

    // --- Hit detection ---
    checkHits() {
      for (let i = P.arr.length - 1; i >= 0; i--) {
        const a = P.arr[i]; if (a.st) continue;
        if (a.x > A.x && a.x < A.x + A.w && a.y > A.y && a.y < A.y + A.h) {
          Particles.spawn(a.x, a.y, A.col); P.arr.splice(i, 1); return 'player';
        }
      }
      for (let i = A.arr.length - 1; i >= 0; i--) {
        const a = A.arr[i]; if (a.st) continue;
        if (a.x > P.x && a.x < P.x + P.w && a.y > P.y && a.y < P.y + P.h) {
          Particles.spawn(a.x, a.y, P.col); A.arr.splice(i, 1); return 'ai';
        }
      }
      return null;
    },

    // === MAIN FRAME (called by engine every frame) ===
    frame(ctx, vW, vH) {
      if (!roundOver) {
        this.handleInput();
        // AI
        const brain = NA.activeBrain;
        if (brain && A) {
          const inp = this.getAIInputs(vW, vH);
          const out = brain.forward(inp);
          this.processAIOutputs(out);
          // Deep Learning : collecte d'expérience par frame (mode solo)
          if (brain.pgLearner) {
            // Récompense dense : survie + qualité de visée
            const enemyDx = P.x - A.x, enemyDy = P.y - A.y;
            const aimTarget = Math.atan2(enemyDy, Math.abs(enemyDx));
            let aimDiff = Math.abs(A.aim - aimTarget);
            if (aimDiff > Math.PI) aimDiff = Math.PI * 2 - aimDiff;
            const aimQ = Math.max(0, (0.4 - aimDiff) / 0.4);
            const pgReward = aimQ * 0.008
              + (out[5] > 0.5 && aimDiff < 0.3 ? 0.05 : 0)
              + (out[5] > 0.5 && aimDiff > 0.8 ? -0.02 : 0)
              + 0.005;
            brain.pgLearner.step(out, pgReward, brain);
          }
        }
        if (P) P.update();
        if (A) A.update();
        const hit = this.checkHits();
        if (hit) { NA.onRoundEnd(hit); roundOver = true; }
        roundTimer++;
        if (roundTimer > 60 * 35) { NA.onRoundEnd(null); roundOver = true; }
      }

      // Camera
      Camera.follow(P, vW, vH, WW, WH);
      Particles.update();

      // Draw
      this.draw(ctx, vW, vH);
    },

    // === DRAWING ===
    draw(ctx, vW, vH) {
      ctx.fillStyle = '#080a12'; ctx.fillRect(0, 0, vW, vH);
      ctx.save(); ctx.translate(-Camera.x, -Camera.y);
      // Grid
      ctx.strokeStyle = 'rgba(255,255,255,0.012)'; ctx.lineWidth = 1;
      const gs = 50, sx = Math.floor(Camera.x / gs) * gs, sy = Math.floor(Camera.y / gs) * gs;
      for (let x = sx; x < Camera.x + vW + gs; x += gs) { ctx.beginPath(); ctx.moveTo(x, Camera.y); ctx.lineTo(x, Camera.y + vH); ctx.stroke(); }
      for (let y = sy; y < Camera.y + vH + gs; y += gs) { ctx.beginPath(); ctx.moveTo(Camera.x, y); ctx.lineTo(Camera.x + vW, y); ctx.stroke(); }
      // Platforms
      for (const p of PLATFORMS) {
        if (p.x + p.w < Camera.x - 20 || p.x > Camera.x + vW + 20 || p.y + p.h < Camera.y - 20 || p.y > Camera.y + vH + 20) continue;
        if (p.w <= 12 || p.h >= WH - 40) {
          ctx.fillStyle = '#14141c'; ctx.fillRect(p.x, p.y, p.w, p.h);
          if (p.h >= WH - 40) { ctx.fillStyle = 'rgba(255,255,255,0.05)'; ctx.fillRect(p.x, p.y, p.w, 1); }
          continue;
        }
        const gr = ctx.createLinearGradient(p.x, p.y, p.x, p.y + p.h);
        gr.addColorStop(0, '#2a2a38'); gr.addColorStop(1, '#1a1a24');
        ctx.fillStyle = gr; ctx.beginPath(); ctx.roundRect(p.x, p.y, p.w, p.h, 3); ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.08)'; ctx.fillRect(p.x + 2, p.y, p.w - 4, 1);
      }
      if (P) P.draw(ctx);
      if (A) A.draw(ctx);
      Particles.draw(ctx);
      ctx.restore();
      // Offscreen indicator
      if (A && P) {
        const ex = A.x + A.w / 2 - Camera.x, ey = A.y + A.h / 2 - Camera.y;
        if (ex < -30 || ex > vW + 30 || ey < -30 || ey > vH + 30) {
          const cx2 = vW / 2, cy2 = vH / 2, ang = Math.atan2(ey - cy2, ex - cx2);
          let ix = cx2 + Math.cos(ang) * (vW / 2 - 36), iy = cy2 + Math.sin(ang) * (vH / 2 - 36);
          ix = Math.max(36, Math.min(vW - 36, ix)); iy = Math.max(36, Math.min(vH - 36, iy));
          ctx.save(); ctx.translate(ix, iy); ctx.rotate(ang);
          ctx.fillStyle = 'rgba(255,50,80,0.7)'; ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-5, -6); ctx.lineTo(-5, 6); ctx.closePath(); ctx.fill();
          ctx.rotate(-ang); ctx.font = '8px JetBrains Mono'; ctx.fillStyle = 'rgba(255,50,80,0.5)'; ctx.textAlign = 'center';
          ctx.fillText(Math.round(Math.sqrt((A.x - P.x) ** 2 + (A.y - P.y) ** 2)) + 'm', 0, -12);
          ctx.restore();
        }
      }
    },

    get isRoundOver() { return roundOver; },
    set isRoundOver(v) { roundOver = v; }
  };

  // === AUTO-REGISTER ===
  GameRegistry.register(ArenaArcher);

})();
