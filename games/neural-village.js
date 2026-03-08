// ================================================================
//  NEURAL VILLAGE — Cozy Life Simulation
//  Player controls a character, AI villagers learn to live
// ================================================================
(function () {
  const { Brain, GameRegistry, Input, Camera, Particles } = window.NeuralArena;
  const NA = window.NeuralArena;

  // === WORLD ===
  const WW = 1200, WH = 900;
  const DAY_LENGTH = 60 * 60; // ~60 seconds at 60fps
  const TILE = 40;

  // Day/night cycle
  let dayTime = 0; // 0..1 (0=sunrise, 0.25=noon, 0.5=sunset, 0.75=midnight)
  let dayCount = 0;
  let totalHappiness = 0;
  let roundActive = true;

  // Palette
  const PAL = {
    grass1: '#7ec97a', grass2: '#6ab866',
    path: '#d4c49a', water: '#7ec8d9',
    wood: '#a67c52', roof: '#e07058',
    leafDay: '#5aaa56', leafNight: '#2d6640',
    sky: (t) => {
      if (t < 0.2) return lerpColor('#1a1a3a', '#ffb87a', t / 0.2);
      if (t < 0.35) return lerpColor('#ffb87a', '#87ceeb', (t - 0.2) / 0.15);
      if (t < 0.5) return lerpColor('#87ceeb', '#ffb87a', (t - 0.35) / 0.15);
      if (t < 0.65) return lerpColor('#ffb87a', '#1a1a3a', (t - 0.5) / 0.15);
      return '#1a1a3a';
    }
  };

  function lerpColor(a, b, t) {
    const ah = parseInt(a.slice(1), 16), bh = parseInt(b.slice(1), 16);
    const ar = ah >> 16, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
    const br = bh >> 16, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
    const r = Math.round(ar + (br - ar) * t), g = Math.round(ag + (bg - ag) * t), bl = Math.round(ab + (bb - ab) * t);
    return `rgb(${r},${g},${bl})`;
  }

  function dayAlpha() { // brightness: 1=day, 0.3=night
    if (dayTime < 0.2) return 0.3 + 0.7 * (dayTime / 0.2);
    if (dayTime < 0.5) return 1;
    if (dayTime < 0.65) return 1 - 0.7 * ((dayTime - 0.5) / 0.15);
    return 0.3;
  }

  // === STATIC WORLD OBJECTS ===
  const houses = [
    { x: 200, y: 200, w: 60, h: 50, color: '#e07058' },
    { x: 500, y: 150, w: 60, h: 50, color: '#d4a058' },
    { x: 850, y: 250, w: 60, h: 50, color: '#7070c0' },
    { x: 350, y: 600, w: 60, h: 50, color: '#c07080' },
  ];

  const trees = [];
  const treePositions = [
    [80, 100], [150, 500], [400, 80], [700, 120], [950, 150],
    [100, 700], [600, 650], [900, 600], [1050, 400], [1100, 200],
    [300, 350], [750, 400], [500, 450], [850, 700], [200, 820],
    [450, 780], [700, 800], [1000, 700], [50, 400], [650, 300]
  ];
  for (const [x, y] of treePositions) trees.push({ x, y, size: 14 + Math.random() * 10, sway: Math.random() * Math.PI * 2 });

  const benches = [
    { x: 320, y: 300 }, { x: 680, y: 500 }, { x: 150, y: 600 }, { x: 900, y: 450 }
  ];

  const pond = { x: 600, y: 350, rx: 70, ry: 45 };

  let foods = [];
  let sparkles = []; // cozy particles (fireflies, petals)

  function spawnFood() {
    if (foods.length < 15) {
      const x = 40 + Math.random() * (WW - 80), y = 40 + Math.random() * (WH - 80);
      // Don't spawn in water
      const dx = x - pond.x, dy = y - pond.y;
      if ((dx * dx) / (pond.rx * pond.rx) + (dy * dy) / (pond.ry * pond.ry) > 1.3) {
        const types = ['🍎', '🍊', '🫐', '🍓', '🥕'];
        foods.push({ x, y, type: types[Math.floor(Math.random() * types.length)], energy: 20 + Math.random() * 15 });
      }
    }
  }

  // === ENTITIES ===
  class Character {
    constructor(x, y, name, color, isAI) {
      this.x = x; this.y = y;
      this.vx = 0; this.vy = 0;
      this.w = 16; this.h = 16;
      this.name = name;
      this.color = color;
      this.isAI = isAI;
      this.speed = isAI ? 1.2 : 2.0;
      this.face = 1; // 1=right, -1=left
      this.faceY = 1; // 1=down, -1=up

      // Needs (0-100)
      this.hunger = 80;
      this.energy = 90;
      this.social = 60;
      this.happiness = 70;

      // State
      this.state = 'idle'; // idle, walking, eating, sleeping, talking
      this.stateTimer = 0;
      this.emote = ''; // emoji above head
      this.emoteTimer = 0;
      this.heldItem = null;
      this.talkTarget = null;

      // Animation
      this.walkFrame = 0;
      this.bobble = Math.random() * Math.PI * 2;
    }

    showEmote(emoji, duration = 90) {
      this.emote = emoji;
      this.emoteTimer = duration;
    }

    update() {
      // Needs decay
      this.hunger = Math.max(0, this.hunger - 0.012);
      this.energy = Math.max(0, this.energy - 0.008);
      this.social = Math.max(0, this.social - 0.006);

      // Happiness = weighted average + bonuses
      this.happiness = this.hunger * 0.35 + this.energy * 0.3 + this.social * 0.35;
      if (this.hunger < 20) this.happiness -= 15;
      if (this.energy < 15) this.happiness -= 15;
      this.happiness = Math.max(0, Math.min(100, this.happiness));

      // Movement
      this.x += this.vx;
      this.y += this.vy;

      // World bounds
      this.x = Math.max(10, Math.min(WW - 10, this.x));
      this.y = Math.max(10, Math.min(WH - 10, this.y));

      // Water collision (push out of pond)
      const pdx = this.x - pond.x, pdy = this.y - pond.y;
      const inPond = (pdx * pdx) / (pond.rx * pond.rx) + (pdy * pdy) / (pond.ry * pond.ry);
      if (inPond < 0.85) {
        const ang = Math.atan2(pdy, pdx);
        this.x = pond.x + Math.cos(ang) * pond.rx * 0.92;
        this.y = pond.y + Math.sin(ang) * pond.ry * 0.92;
      }

      // Facing direction
      if (Math.abs(this.vx) > 0.1) this.face = this.vx > 0 ? 1 : -1;
      if (Math.abs(this.vy) > 0.3) this.faceY = this.vy > 0 ? 1 : -1;

      // Walk animation
      if (Math.abs(this.vx) > 0.1 || Math.abs(this.vy) > 0.1) {
        this.walkFrame += 0.15;
        this.state = 'walking';
      } else {
        this.walkFrame = 0;
        if (this.state === 'walking') this.state = 'idle';
      }

      // Friction
      this.vx *= 0.8;
      this.vy *= 0.8;

      // Emote timer
      if (this.emoteTimer > 0) this.emoteTimer--;

      // State timer
      if (this.stateTimer > 0) {
        this.stateTimer--;
        if (this.stateTimer === 0) this.state = 'idle';
      }

      // Energy restore if sleeping
      if (this.state === 'sleeping') {
        this.energy = Math.min(100, this.energy + 0.15);
        this.hunger = Math.max(0, this.hunger - 0.005);
      }

      // Social restore if talking
      if (this.state === 'talking') {
        this.social = Math.min(100, this.social + 0.3);
      }

      this.bobble += 0.03;
    }

    // Check proximity
    distTo(other) {
      const dx = this.x - other.x, dy = this.y - other.y;
      return Math.sqrt(dx * dx + dy * dy);
    }

    nearestFood() {
      let best = null, bestD = 999;
      for (const f of foods) {
        const d = Math.sqrt((this.x - f.x) ** 2 + (this.y - f.y) ** 2);
        if (d < bestD) { bestD = d; best = f; }
      }
      return { food: best, dist: bestD };
    }

    nearestHouse() {
      let best = null, bestD = 999;
      for (const h of houses) {
        const cx = h.x + h.w / 2, cy = h.y + h.h / 2;
        const d = Math.sqrt((this.x - cx) ** 2 + (this.y - cy) ** 2);
        if (d < bestD) { bestD = d; best = h; }
      }
      return { house: best, dist: bestD };
    }

    nearestCharacter(chars) {
      let best = null, bestD = 999;
      for (const c of chars) {
        if (c === this) continue;
        const d = this.distTo(c);
        if (d < bestD) { bestD = d; best = c; }
      }
      return { char: best, dist: bestD };
    }

    // Eat nearby food
    eat() {
      const { food, dist } = this.nearestFood();
      if (food && dist < 25) {
        this.hunger = Math.min(100, this.hunger + food.energy);
        foods.splice(foods.indexOf(food), 1);
        this.state = 'eating';
        this.stateTimer = 40;
        this.showEmote('😋');
        return true;
      }
      return false;
    }

    // Sleep in nearby house
    sleep() {
      const { house, dist } = this.nearestHouse();
      if (house && dist < 50) {
        this.state = 'sleeping';
        this.stateTimer = 180;
        this.showEmote('💤');
        return true;
      }
      return false;
    }

    // Talk to nearby character
    talk(chars) {
      const { char, dist } = this.nearestCharacter(chars);
      if (char && dist < 35) {
        this.state = 'talking';
        this.stateTimer = 60;
        this.talkTarget = char;
        char.social = Math.min(100, char.social + 5);
        this.showEmote('💬');
        char.showEmote('😊');
        return true;
      }
      return false;
    }

    draw(ctx) {
      const x = this.x, y = this.y;
      const bob = Math.sin(this.bobble) * 1.5;
      const walkBob = Math.sin(this.walkFrame) * 2;

      // Shadow
      ctx.fillStyle = 'rgba(0,0,0,0.1)';
      ctx.beginPath();
      ctx.ellipse(x, y + 10, 8, 3, 0, 0, Math.PI * 2);
      ctx.fill();

      // Body (round)
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(x, y - 2 + bob + walkBob * 0.3, 10, 0, Math.PI * 2);
      ctx.fill();

      // Sleeping overlay
      if (this.state === 'sleeping') {
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.beginPath();
        ctx.arc(x, y - 2 + bob, 10, 0, Math.PI * 2);
        ctx.fill();
      }

      // Face
      const faceX = x + this.face * 2;
      const faceY = y - 3 + bob + walkBob * 0.3;

      if (this.state !== 'sleeping') {
        // Eyes
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(faceX - 3, faceY, 2.5, 0, Math.PI * 2);
        ctx.arc(faceX + 3, faceY, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#222';
        ctx.beginPath();
        ctx.arc(faceX - 2.5 + this.face * 0.5, faceY, 1.2, 0, Math.PI * 2);
        ctx.arc(faceX + 3.5 + this.face * 0.5, faceY, 1.2, 0, Math.PI * 2);
        ctx.fill();

        // Mouth (happy or sad based on happiness)
        if (this.happiness > 50) {
          ctx.strokeStyle = '#222';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(faceX, faceY + 3, 2, 0.1, Math.PI - 0.1);
          ctx.stroke();
        } else {
          ctx.strokeStyle = '#222';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(faceX, faceY + 5, 2, Math.PI + 0.2, -0.2);
          ctx.stroke();
        }
      } else {
        // Closed eyes
        ctx.strokeStyle = '#222'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(faceX - 5, faceY); ctx.lineTo(faceX - 1, faceY);
        ctx.moveTo(faceX + 1, faceY); ctx.lineTo(faceX + 5, faceY);
        ctx.stroke();
      }

      // Hair / hat (little tuft on top)
      ctx.fillStyle = this.isAI ? darkenColor(this.color, 0.3) : '#5588cc';
      ctx.beginPath();
      ctx.arc(x, y - 12 + bob, 5, Math.PI, 0);
      ctx.fill();

      // Feet (when walking)
      if (Math.abs(this.vx) > 0.1 || Math.abs(this.vy) > 0.1) {
        const legOff = Math.sin(this.walkFrame) * 3;
        ctx.fillStyle = darkenColor(this.color, 0.2);
        ctx.beginPath();
        ctx.arc(x - 4, y + 8 + legOff, 3, 0, Math.PI * 2);
        ctx.arc(x + 4, y + 8 - legOff, 3, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = darkenColor(this.color, 0.2);
        ctx.beginPath();
        ctx.arc(x - 4, y + 8, 3, 0, Math.PI * 2);
        ctx.arc(x + 4, y + 8, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Name tag
      ctx.font = '7px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillStyle = this.isAI ? 'rgba(255,255,255,0.5)' : 'rgba(100,200,255,0.8)';
      ctx.fillText(this.name, x, y + 20);

      // Emote
      if (this.emoteTimer > 0) {
        const emoteY = y - 22 + bob - Math.sin(this.emoteTimer * 0.1) * 3;
        ctx.font = '14px serif';
        ctx.globalAlpha = Math.min(1, this.emoteTimer / 20);
        ctx.fillText(this.emote, x, emoteY);
        ctx.globalAlpha = 1;
      }

      // Need indicator (small bar below name if critical)
      if (this.hunger < 30) {
        ctx.fillStyle = 'rgba(255,100,50,0.6)';
        ctx.fillRect(x - 10, y + 23, 20 * (this.hunger / 100), 2);
        ctx.strokeStyle = 'rgba(255,100,50,0.3)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(x - 10, y + 23, 20, 2);
      }
      if (this.energy < 25) {
        ctx.fillStyle = 'rgba(100,100,255,0.6)';
        ctx.fillRect(x - 10, y + 26, 20 * (this.energy / 100), 2);
        ctx.strokeStyle = 'rgba(100,100,255,0.3)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(x - 10, y + 26, 20, 2);
      }
    }
  }

  function darkenColor(hex, amount) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgb(${Math.round(r * (1 - amount))},${Math.round(g * (1 - amount))},${Math.round(b * (1 - amount))})`;
  }

  // === GAME STATE ===
  let player = null;
  let villagers = [];
  let allChars = [];

  const villagerData = [
    { name: 'Mochi', color: '#f0a0b0' },
    { name: 'Nori', color: '#a0d8a0' },
    { name: 'Yuzu', color: '#f0d080' },
    { name: 'Taro', color: '#b0a0d8' },
  ];

  // === AI PERCEPTION ===
  function getVillagerInputs(v, vW, vH) {
    // Needs (4)
    const hunger = v.hunger / 100;
    const energy = v.energy / 100;
    const social = v.social / 100;
    const happy = v.happiness / 100;

    // Nearest food direction + dist (3)
    const nf = v.nearestFood();
    const foodDist = nf.food ? Math.min(nf.dist / 300, 1) : 1;
    const foodAng = nf.food ? Math.atan2(nf.food.y - v.y, nf.food.x - v.x) : 0;

    // Nearest house direction + dist (3)
    const nh = v.nearestHouse();
    const houseDist = nh.house ? Math.min(nh.dist / 300, 1) : 1;
    const houseAng = nh.house ? Math.atan2((nh.house.y + nh.house.h / 2) - v.y, (nh.house.x + nh.house.w / 2) - v.x) : 0;

    // Nearest other character (includes player!) (2)
    const nc = v.nearestCharacter(allChars);
    const charDist = nc.char ? Math.min(nc.dist / 200, 1) : 1;
    const charAng = nc.char ? Math.atan2(nc.char.y - v.y, nc.char.x - v.x) : 0;

    // Time of day (2) — sin/cos cycle
    const timeX = Math.sin(dayTime * Math.PI * 2);
    const timeY = Math.cos(dayTime * Math.PI * 2);

    return [
      hunger, energy, social, happy,
      1 - foodDist, Math.sin(foodAng), Math.cos(foodAng),
      1 - houseDist, Math.sin(houseAng), Math.cos(houseAng),
      1 - charDist, Math.sin(charAng), Math.cos(charAng),
      timeX
    ];
  }

  function processVillagerOutputs(v, out) {
    // out: [moveX, moveY, eat, sleep, socialize]
    const mx = (out[0] - 0.5) * 2;
    const my = (out[1] - 0.5) * 2;

    if (v.state === 'sleeping' || v.state === 'eating') return; // Busy

    // Move
    if (Math.abs(mx) > 0.15 || Math.abs(my) > 0.15) {
      v.vx += mx * v.speed * 0.4;
      v.vy += my * v.speed * 0.4;
    }

    // Eat
    if (out[2] > 0.6) v.eat();

    // Sleep (prefer at night)
    if (out[3] > 0.6) v.sleep();

    // Socialize
    if (out[4] > 0.6) v.talk(allChars);
  }

  // === COZY PARTICLES ===
  function updateSparkles() {
    // Spawn
    const isNight = dayTime > 0.55 || dayTime < 0.15;
    if (isNight && sparkles.length < 40 && Math.random() < 0.08) {
      sparkles.push({
        x: Camera.x + Math.random() * NA.Engine.viewW,
        y: Camera.y + Math.random() * NA.Engine.viewH,
        vx: (Math.random() - 0.5) * 0.3,
        vy: -0.1 - Math.random() * 0.2,
        life: 120 + Math.random() * 120,
        size: 1 + Math.random() * 2,
        type: 'firefly'
      });
    }
    if (!isNight && sparkles.length < 20 && Math.random() < 0.03) {
      sparkles.push({
        x: Camera.x + Math.random() * NA.Engine.viewW,
        y: Camera.y - 10,
        vx: 0.3 + Math.random() * 0.5,
        vy: 0.5 + Math.random() * 0.3,
        life: 200 + Math.random() * 100,
        size: 2 + Math.random() * 2,
        type: 'petal'
      });
    }

    for (let i = sparkles.length - 1; i >= 0; i--) {
      const s = sparkles[i];
      s.x += s.vx + Math.sin(Date.now() * 0.002 + i) * 0.15;
      s.y += s.vy;
      s.life--;
      if (s.life <= 0) sparkles.splice(i, 1);
    }
  }

  // === PLAYER INPUT ===
  function handlePlayerInput() {
    if (!player) return;
    const K = Input.keys, J = Input.joy;

    // Keyboard
    if (K['KeyA'] || K['ArrowLeft']) { player.vx -= player.speed * 0.5; }
    if (K['KeyD'] || K['ArrowRight']) { player.vx += player.speed * 0.5; }
    if (K['KeyW'] || K['ArrowUp'] || K['KeyZ']) { player.vy -= player.speed * 0.5; }
    if (K['KeyS'] || K['ArrowDown']) { player.vy += player.speed * 0.5; }

    // Action key
    if (K['Space'] || K['KeyE']) {
      K['Space'] = false; K['KeyE'] = false; // Single press
      playerAction();
    }

    // Left joystick: move
    if (J.left.active) {
      if (Math.abs(J.left.x) > 0.15) player.vx += J.left.x * player.speed * 0.6;
      if (Math.abs(J.left.y) > 0.15) player.vy += J.left.y * player.speed * 0.6;
    }

    // Jump/Action button
    if (J.jump) {
      J.jump = false;
      playerAction();
    }

    // Right joystick: interact in direction (auto-action when pushed)
    if (J.right.active) {
      const mag = Math.sqrt(J.right.x ** 2 + J.right.y ** 2);
      if (mag > 0.5) {
        playerAction();
      }
    }
  }

  let actionCooldown = 0;

  function playerAction() {
    if (actionCooldown > 0) return;
    actionCooldown = 20;

    // Try talk to nearest villager
    const nc = player.nearestCharacter(allChars);
    if (nc.char && nc.dist < 35) {
      player.state = 'talking';
      player.stateTimer = 30;
      nc.char.social = Math.min(100, nc.char.social + 12);
      nc.char.happiness = Math.min(100, nc.char.happiness + 5);
      player.showEmote('👋');
      nc.char.showEmote(nc.char.happiness > 60 ? '❤️' : '😊');
      player.social = Math.min(100, player.social + 8);
      return;
    }

    // Try eat food
    if (player.eat()) return;

    // Try sleep
    const nh = player.nearestHouse();
    if (nh.house && nh.dist < 50) {
      player.sleep();
      return;
    }

    // Wave emote
    player.showEmote('✋');
  }

  // === FOOD SPAWN ===
  let foodSpawnTimer = 0;

  // === MODULE DEFINITION ===
  const NeuralVillage = {
    id: 'neural-village',
    name: 'Neural Village',
    icon: '🏡',
    description: 'Vie cozy — Les villageois IA apprennent à vivre',
    topology: [14, 16, 10, 5],

    initRound() {
      dayTime = 0.2; // Start at morning
      dayCount = 0;
      totalHappiness = 0;
      roundActive = true;
      foods = [];
      sparkles = [];
      actionCooldown = 0;
      foodSpawnTimer = 0;

      // Player
      player = new Character(WW / 2, WH / 2, 'Joueur', '#70b0e0', false);
      player.hunger = 90; player.energy = 95; player.social = 80;

      // Villagers
      villagers = villagerData.map((d, i) => {
        const angle = (i / villagerData.length) * Math.PI * 2;
        const v = new Character(
          WW / 2 + Math.cos(angle) * 120,
          WH / 2 + Math.sin(angle) * 120,
          d.name, d.color, true
        );
        return v;
      });

      allChars = [player, ...villagers];

      // Initial food
      for (let i = 0; i < 10; i++) spawnFood();

      Camera.reset(player.x - NA.Engine.viewW / 2, player.y - NA.Engine.viewH / 2);
    },

    frame(ctx, vW, vH) {
      if (!roundActive) return this.draw(ctx, vW, vH);

      // Day cycle
      dayTime += 1 / DAY_LENGTH;
      if (dayTime >= 1) {
        dayTime = 0;
        dayCount++;
        // Day ended = round end — evaluate fitness
        let avgHappy = 0;
        for (const v of villagers) avgHappy += v.happiness;
        avgHappy /= villagers.length;
        NA.onRoundEnd(avgHappy > 55 ? 'ai' : 'player'); // AI "wins" if villagers are happy
        roundActive = false;
        return this.draw(ctx, vW, vH);
      }

      // Player input
      handlePlayerInput();
      if (actionCooldown > 0) actionCooldown--;

      // Food spawn
      foodSpawnTimer++;
      if (foodSpawnTimer > 120) { spawnFood(); foodSpawnTimer = 0; }

      // Update player
      player.update();

      // AI villagers
      const brain = NA.activeBrain;
      for (const v of villagers) {
        if (brain) {
          const inputs = getVillagerInputs(v, vW, vH);
          const outputs = brain.forward(inputs);
          processVillagerOutputs(v, outputs);
        }
        v.update();
      }

      // Camera follow player
      Camera.follow(player, vW, vH, WW, WH);

      // Cozy particles
      updateSparkles();

      // Track total happiness for fitness
      for (const v of villagers) totalHappiness += v.happiness;

      this.draw(ctx, vW, vH);
    },

    draw(ctx, vW, vH) {
      const alpha = dayAlpha();
      const skyCol = PAL.sky(dayTime);

      // Sky
      ctx.fillStyle = skyCol;
      ctx.fillRect(0, 0, vW, vH);

      ctx.save();
      ctx.translate(-Camera.x, -Camera.y);

      // Grass
      const grd = ctx.createLinearGradient(0, 0, 0, WH);
      grd.addColorStop(0, PAL.grass1);
      grd.addColorStop(1, PAL.grass2);
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, WW, WH);

      // Night overlay
      if (alpha < 1) {
        ctx.fillStyle = `rgba(10,10,30,${(1 - alpha) * 0.5})`;
        ctx.fillRect(0, 0, WW, WH);
      }

      // Grass tufts
      ctx.fillStyle = `rgba(90,170,86,${0.3 * alpha})`;
      for (let x = 20; x < WW; x += 35) {
        for (let y = 20; y < WH; y += 40) {
          if ((x * 7 + y * 13) % 5 === 0) {
            ctx.beginPath();
            ctx.arc(x + (y % 7), y + (x % 5), 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Pond
      ctx.fillStyle = '#7ec8d9';
      ctx.beginPath();
      ctx.ellipse(pond.x, pond.y, pond.rx, pond.ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.beginPath();
      ctx.ellipse(pond.x - 15, pond.y - 10, pond.rx * 0.4, pond.ry * 0.3, -0.3, 0, Math.PI * 2);
      ctx.fill();

      // Benches
      for (const b of benches) {
        ctx.fillStyle = '#a67c52';
        ctx.fillRect(b.x - 12, b.y - 3, 24, 6);
        ctx.fillStyle = '#8a6840';
        ctx.fillRect(b.x - 10, b.y + 3, 3, 6);
        ctx.fillRect(b.x + 7, b.y + 3, 3, 6);
      }

      // Food
      ctx.font = '14px serif';
      ctx.textAlign = 'center';
      for (const f of foods) {
        // Soft glow
        ctx.fillStyle = 'rgba(255,220,100,0.1)';
        ctx.beginPath(); ctx.arc(f.x, f.y, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillText(f.type, f.x, f.y + 5);
      }

      // Houses (sorted by Y for depth)
      for (const h of houses) {
        // Wall
        ctx.fillStyle = '#f5e6d0';
        ctx.fillRect(h.x, h.y, h.w, h.h);
        // Roof
        ctx.fillStyle = h.color;
        ctx.beginPath();
        ctx.moveTo(h.x - 6, h.y);
        ctx.lineTo(h.x + h.w / 2, h.y - 22);
        ctx.lineTo(h.x + h.w + 6, h.y);
        ctx.closePath();
        ctx.fill();
        // Door
        ctx.fillStyle = '#8a6840';
        ctx.fillRect(h.x + h.w / 2 - 6, h.y + h.h - 16, 12, 16);
        // Window
        ctx.fillStyle = alpha > 0.6 ? 'rgba(180,220,255,0.5)' : 'rgba(255,200,100,0.6)';
        ctx.fillRect(h.x + 8, h.y + 10, 10, 10);
        ctx.fillRect(h.x + h.w - 18, h.y + 10, 10, 10);
      }

      // Trees (sorted by Y for depth)
      const sortedTrees = [...trees].sort((a, b) => a.y - b.y);
      for (const t of sortedTrees) {
        const sway = Math.sin(Date.now() * 0.001 + t.sway) * 1.5;
        // Trunk
        ctx.fillStyle = '#8a6840';
        ctx.fillRect(t.x - 3, t.y - 5, 6, 18);
        // Canopy
        const leafCol = alpha > 0.6 ? PAL.leafDay : PAL.leafNight;
        ctx.fillStyle = leafCol;
        ctx.beginPath();
        ctx.arc(t.x + sway, t.y - 10, t.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.arc(t.x + sway - 3, t.y - 14, t.size * 0.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Characters (sorted by Y for depth)
      const sortedChars = [...allChars].sort((a, b) => a.y - b.y);
      for (const ch of sortedChars) {
        ch.draw(ctx);
      }

      // Sparkles / particles
      for (const s of sparkles) {
        if (s.type === 'firefly') {
          const pulse = 0.4 + Math.sin(Date.now() * 0.005 + s.x) * 0.4;
          ctx.fillStyle = `rgba(255,255,150,${pulse * (s.life / 200)})`;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
          ctx.fill();
          // Glow
          ctx.fillStyle = `rgba(255,255,100,${pulse * 0.15 * (s.life / 200)})`;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.size * 4, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = `rgba(255,180,200,${s.life / 300})`;
          ctx.beginPath();
          ctx.ellipse(s.x, s.y, s.size, s.size * 0.6, Date.now() * 0.002, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.restore();

      // === HUD overlay (needs bars + time) ===
      this.drawHUD(ctx, vW, vH);
    },

    drawHUD(ctx, vW, vH) {
      // Day/night indicator
      const timeLabel = dayTime < 0.2 ? '🌅 Aube' : dayTime < 0.4 ? '☀️ Jour' : dayTime < 0.55 ? '🌇 Crépuscule' : '🌙 Nuit';
      const dayProgress = dayTime;

      // Bottom left: player needs
      const bx = 12, by = vH - 100;
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.beginPath();
      ctx.roundRect(bx, by, 100, 88, 6);
      ctx.fill();

      ctx.font = '8px JetBrains Mono';
      ctx.textAlign = 'left';
      const needs = [
        { label: '🍎 Faim', val: player ? player.hunger : 0, col: '#f0a060' },
        { label: '⚡ Énergie', val: player ? player.energy : 0, col: '#70a0f0' },
        { label: '💬 Social', val: player ? player.social : 0, col: '#a0d870' },
        { label: '😊 Bonheur', val: player ? player.happiness : 0, col: '#f0d060' },
      ];

      needs.forEach((n, i) => {
        const ny = by + 10 + i * 20;
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText(n.label, bx + 6, ny);
        // Bar
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fillRect(bx + 6, ny + 3, 82, 5);
        ctx.fillStyle = n.col;
        ctx.fillRect(bx + 6, ny + 3, 82 * (n.val / 100), 5);
      });

      // Top center: day info
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.beginPath();
      ctx.roundRect(vW / 2 - 60, 44, 120, 22, 4);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${timeLabel} · Jour ${dayCount + 1}`, vW / 2, 58);

      // Day progress bar
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      ctx.fillRect(vW / 2 - 50, 67, 100, 3);
      ctx.fillStyle = 'rgba(255,220,100,0.4)';
      ctx.fillRect(vW / 2 - 50, 67, 100 * dayProgress, 3);
    },

    get isRoundOver() { return !roundActive; },
    set isRoundOver(v) { roundActive = !v; }
  };

  // === AUTO-REGISTER ===
  GameRegistry.register(NeuralVillage);
})();
