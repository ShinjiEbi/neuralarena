// ================================================================
//  NEURAL ARENA — CŒUR DE SIMULATION
//  Partagé par la page (index.html) et le worker d'entraînement
//  (train-worker.js) : aucun accès au DOM au chargement.
//  Contenu : réseaux (GRU, dense, multi-experts), gradient de politique,
//  arènes, entités, IA, entraîneurs scriptés, pièges, notation,
//  stockage IndexedDB, session d'entraînement.
// ================================================================

// Message à l'utilisateur : toast dans la page, transmis à la page depuis le worker
function notify(msg){
  if(typeof toast==='function') toast(msg);
  else if(typeof self!=='undefined' && typeof self.postMessage==='function' && typeof document==='undefined') self.postMessage({type:'toast', msg});
}

// ================================================================
//  GRU EXPERT — réseau récurrent léger avec mémoire interne
// ================================================================
// inputs : sous-ensemble des 26 inputs globaux (indices configurables)
// H      : taille de la mémoire (hidden state), défaut 8
// outSize: taille de la sortie projetée après le GRU
//
// Architecture GRU minimale :
//   z = sigmoid(Wz · [h, x])   porte update
//   r = sigmoid(Wr · [h, x])   porte reset
//   c = tanh(Wc · [r⊙h, x])   candidat
//   h = (1-z)⊙h + z⊙c         nouvel état
//   out = Wo · h + bo           projection linéaire → outSize
//
class GRUExpert {
  constructor(inSize, H=8, outSize=4, xavierScale=1.0){
    this.inSize=inSize; this.H=H; this.outSize=outSize;
    const s=Math.sqrt(2/(inSize+H))*xavierScale;
    // Initialisation Xavier pour Wz, Wr, Wc : shape [H][H+inSize]
    const mat=(rows,cols,sc)=>Array.from({length:rows},()=>Array.from({length:cols},()=>(Math.random()*2-1)*sc));
    const vec=(n,sc=0.05)=>Array.from({length:n},()=>(Math.random()*2-1)*sc);
    this.p={
      Wz:mat(H,H+inSize,s), bz:vec(H),
      Wr:mat(H,H+inSize,s), br:vec(H),
      Wc:mat(H,H+inSize,s), bc:vec(H),
      Wo:mat(outSize,H,Math.sqrt(2/H)), bo:vec(outSize),
    };
    this.h=new Array(H).fill(0);  // état caché (mémoire)
    this.fitness=0; this.elo=1200;
  }

  // Instance sans tirage de poids aléatoires (les poids sont fournis ensuite)
  static _bare(inSize, H, outSize, p){
    const e=Object.create(GRUExpert.prototype);
    e.inSize=inSize; e.H=H; e.outSize=outSize; e.p=p;
    e.h=new Array(H).fill(0); e.fitness=0; e.elo=1200;
    return e;
  }

  // Fonctions d'activation
  _sig(x){return 1/(1+Math.exp(-Math.max(-10,Math.min(10,x))))}
  _tanh(x){const e=Math.exp(-2*Math.max(-10,Math.min(10,x)));return(1-e)/(1+e)}
  _matvec(W,b,v){return W.map((row,i)=>b[i]+row.reduce((s,w,j)=>s+w*v[j],0))}

  // Même calcul que la version map/reduce d'origine (mêmes sommes dans le même ordre),
  // en boucles sans tableaux intermédiaires : ~10× plus rapide.
  forward(x){
    const {Wz,bz,Wr,br,Wc,bc,Wo,bo}=this.p;
    const H=this.H, h=this.h, n=x.length;
    const z=this._z||(this._z=new Array(H)), r=this._r||(this._r=new Array(H));
    for(let i=0;i<H;i++){
      const wz=Wz[i], wr=Wr[i];
      let sz=0, sr=0;
      for(let j=0;j<H;j++){ sz+=wz[j]*h[j]; sr+=wr[j]*h[j]; }
      for(let j=0;j<n;j++){ sz+=wz[H+j]*x[j]; sr+=wr[H+j]*x[j]; }
      z[i]=this._sig(bz[i]+sz); r[i]=this._sig(br[i]+sr);
    }
    const hNew=new Array(H);
    for(let i=0;i<H;i++){
      const wc=Wc[i];
      let sc=0;
      for(let j=0;j<H;j++) sc+=wc[j]*(r[j]*h[j]);
      for(let j=0;j<n;j++) sc+=wc[H+j]*x[j];
      const c=this._tanh(bc[i]+sc);
      hNew[i]=(1-z[i])*h[i]+z[i]*c;
    }
    this.h=hNew;
    // Projection vers outSize
    const out=new Array(this.outSize);
    for(let o=0;o<this.outSize;o++){
      const wo=Wo[o]; let so=0;
      for(let i=0;i<H;i++) so+=wo[i]*hNew[i];
      out[o]=this._tanh(bo[o]+so);
    }
    return out;
  }

  resetMemory(){this.h=new Array(this.H).fill(0);}

  // Mutation gaussienne d'un seul expert
  mutate(rate=0.15, power=0.35){
    const gauss=()=>{const u1=Math.random(),u2=Math.random();return Math.sqrt(-2*Math.log(u1||1e-4))*Math.cos(2*Math.PI*u2)};
    for(const key of['Wz','Wr','Wc','Wo']){
      for(let r=0;r<this.p[key].length;r++)
        for(let c=0;c<this.p[key][r].length;c++)
          if(Math.random()<rate) this.p[key][r][c]+=gauss()*power;
    }
    for(const key of['bz','br','bc','bo']){
      for(let i=0;i<this.p[key].length;i++)
        if(Math.random()<rate) this.p[key][i]+=(Math.random()*2-1)*power*0.5;
    }
  }

  clone(){
    const cp=this.p, p={};
    for(const k in cp) p[k]=Array.isArray(cp[k][0]) ? cp[k].map(r=>r.slice()) : cp[k].slice();
    const e=GRUExpert._bare(this.inSize,this.H,this.outSize,p);
    e.fitness=this.fitness; e.elo=this.elo;
    return e;
  }

  // Crossover : mélange ligne par ligne dans Wz,Wr,Wc,Wo
  static crossover(a,b){
    const ch=a.clone();
    for(const key of['Wz','Wr','Wc','Wo']){
      const cut=Math.floor(Math.random()*ch.p[key].length);
      for(let r=cut;r<ch.p[key].length;r++)
        ch.p[key][r]=b.p[key][r].slice();
    }
    for(const key of['bz','br','bc','bo']){
      const cut=Math.floor(Math.random()*ch.p[key].length);
      for(let i=cut;i<ch.p[key].length;i++)
        ch.p[key][i]=b.p[key][i];
    }
    return ch;
  }

  serialize(){return{inSize:this.inSize,H:this.H,outSize:this.outSize,p:this.p}}
  static deserialize(d){return GRUExpert._bare(d.inSize,d.H,d.outSize,d.p)}
  paramCount(){
    const {Wz,Wr,Wc,Wo,bz,br,bc,bo}=this.p;
    return[Wz,Wr,Wc,Wo].reduce((s,m)=>s+m.length*m[0].length,0)
          +[bz,br,bc,bo].reduce((s,v)=>s+v.length,0);
  }
}

// ================================================================
//  DENSE LAYER — couche dense simple pour l'arbitre
class DenseLayer {
  constructor(inSize,outSize,activation='sigmoid',xavierScale=1.0){
    const s=Math.sqrt(2/(inSize+outSize))*xavierScale;
    this.W=Array.from({length:outSize},()=>Array.from({length:inSize},()=>(Math.random()*2-1)*s));
    this.b=Array.from({length:outSize},()=>(Math.random()*2-1)*0.05);
    this.act=activation;
  }
  // Garde l'entrée et la pré-activation du dernier passage (_lastX/_lastZ) :
  // le PolicyGradientLearner en a besoin pour la rétropropagation.
  forward(x){
    const W=this.W, b=this.b, n=W.length;
    const z=new Array(n), y=new Array(n);
    for(let i=0;i<n;i++){
      const row=W[i]; let s=0;
      for(let j=0;j<row.length;j++) s+=row[j]*x[j];
      z[i]=b[i]+s;
    }
    this._lastX = x;
    this._lastZ = z;
    if(this.act==='sigmoid'){ for(let i=0;i<n;i++) y[i]=1/(1+Math.exp(-Math.max(-10,Math.min(10,z[i])))); return y; }
    if(this.act==='relu'){    for(let i=0;i<n;i++) y[i]=z[i]>0?z[i]:z[i]*0.01; return y; } // leaky relu
    return z;
  }
  mutate(rate,power){
    const gauss=()=>{const u1=Math.random(),u2=Math.random();return Math.sqrt(-2*Math.log(u1||1e-4))*Math.cos(2*Math.PI*u2)};
    for(let r=0;r<this.W.length;r++)for(let c=0;c<this.W[r].length;c++)if(Math.random()<rate)this.W[r][c]+=gauss()*power;
    for(let i=0;i<this.b.length;i++)if(Math.random()<rate)this.b[i]+=(Math.random()*2-1)*power*0.5;
  }
  static _bare(W,b,act){ const l=Object.create(DenseLayer.prototype); l.W=W; l.b=b; l.act=act; return l; }
  clone(){ return DenseLayer._bare(this.W.map(r=>r.slice()), this.b.slice(), this.act); }
  static crossover(a,b){const ch=a.clone();const cut=Math.floor(Math.random()*ch.W.length);for(let r=cut;r<ch.W.length;r++){ch.W[r]=b.W[r].slice();ch.b[r]=b.b[r]}return ch}
  serialize(){return{W:this.W,b:this.b,act:this.act}}
  static deserialize(d){return DenseLayer._bare(d.W, d.b, d.act||'sigmoid')}
  paramCount(){return this.W.length*this.W[0].length+this.b.length}
}

// ================================================================
//  MULTI-EXPERT BRAIN  — format sérialisé 'multi-expert-v4'
//
//  26 inputs → 4 experts GRU spécialisés → arbitre 3 couches → 7 outputs
//
//  EXPERT 0 — DANGER   : esquive flèches        [12,13,14,15,16,3,4]
//  EXPERT 1 — VISÉE    : viser et tirer          [0,1,2,7,8,9,11,5,6]
//  EXPERT 2 — TERRAIN  : positionnement          [10,17,18,19,3,4,20..25]
//  EXPERT 3 — TACTIQUE : lecture globale         [0,1,2,5,6,7,8,16,19]
//
//  ARBITRE : concat(4 experts) → dense(relu) → dense(relu) → dense(sigmoid,7)
//
class MultiExpertBrain {
  // Indices des inputs lus par chaque expert
  static EXPERT_INPUTS=[
    [12,13,14,15,16,3,4],               // DANGER   : flèche proche + mes vélocités (7)
    [0,1,2,7,8,9,11,5,6],               // VISÉE    : position ennemi + erreur visée + cooldown + vélo ennemi (9)
    [10,17,18,19,3,4,20,21,22,23,24,25],// TERRAIN  : sol + position + timer + vélocités + capteurs spatiaux (12)
    [0,1,2,5,6,7,8,16,19],              // TACTIQUE : vue globale distance+vélo+visée+danger+timer (9)
  ];
  static H=12;
  static OUT_PER_EXPERT=4;
  static N_INPUTS=26;
  static N_OUTPUTS=7;

  constructor(H=null, outPerExpert=null, arbSize=null, xavierScale=1.0){
    const h   = H            || MultiExpertBrain.H;
    const out = outPerExpert || MultiExpertBrain.OUT_PER_EXPERT;
    const arb = arbSize      || 16;
    this._H=h; this._out=out; this._arb=arb;

    this.experts=MultiExpertBrain.EXPERT_INPUTS.map(
      idx=>new GRUExpert(idx.length, h, out, xavierScale)
    );
    const arbIn=this.experts.length*out; // 4×4=16
    // Arbitre plus profond : 3 couches
    this.arb1=new DenseLayer(arbIn, arb,    'relu',    xavierScale);
    this.arb2=new DenseLayer(arb,   arb,    'relu',    xavierScale);
    this.arb3=new DenseLayer(arb,   MultiExpertBrain.N_OUTPUTS, 'sigmoid', xavierScale);
    this.fitness=0; this.elo=1200;
    this.topo=[MultiExpertBrain.N_INPUTS,h,h,h,h,arbIn,arb,arb,MultiExpertBrain.N_OUTPUTS];
  }

  forward(inp){
    const EI=MultiExpertBrain.EXPERT_INPUTS;
    const arbIn=new Array(this.experts.length*this._out);
    let k=0;
    for(let i=0;i<this.experts.length;i++){
      const idx=EI[i], x=new Array(idx.length);
      for(let j=0;j<idx.length;j++) x[j]=inp[idx[j]]||0;
      const o=this.experts[i].forward(x);
      for(let j=0;j<o.length;j++) arbIn[k++]=o[j];
    }
    const h1=this.arb1.forward(arbIn);
    const h2=this.arb2.forward(h1);
    return this.arb3.forward(h2);
  }

  resetMemory(){
    this.experts.forEach(e=>e.resetMemory());
  }

  // Mutation différenciée par expert selon leur rôle
  mutate(rate=0.15, power=0.35, expertIdx=null){
    if(expertIdx!==null && expertIdx<this.experts.length){
      this.experts[expertIdx].mutate(rate, power);
    } else {
      // Expert visée muté plus fort (plus difficile à apprendre)
      this.experts[0].mutate(rate,        power);        // danger
      this.experts[1].mutate(rate*1.2,    power*1.2);    // visée : + fort
      this.experts[2].mutate(rate,        power);        // terrain
      this.experts[3].mutate(rate*0.9,    power*0.9);    // tactique : + stable
      this.arb1.mutate(rate*0.6, power*0.6);
      this.arb2.mutate(rate*0.6, power*0.6);
      this.arb3.mutate(rate*0.5, power*0.5); // sortie : très conservatrice
    }
  }

  // Instance vide (sans tirer de poids aléatoires qui seraient aussitôt remplacés)
  static _bare(H,out,arb){
    const b=Object.create(MultiExpertBrain.prototype);
    b._H=H; b._out=out; b._arb=arb; b.fitness=0; b.elo=1200;
    b.topo=[MultiExpertBrain.N_INPUTS,H,H,H,H,MultiExpertBrain.EXPERT_INPUTS.length*out,arb,arb,MultiExpertBrain.N_OUTPUTS];
    return b;
  }

  clone(){
    const b=MultiExpertBrain._bare(this._H, this._out, this._arb);
    b.experts=this.experts.map(e=>e.clone());
    b.arb1=this.arb1.clone();
    b.arb2=this.arb2.clone();
    b.arb3=this.arb3.clone();
    b.fitness=this.fitness; b.elo=this.elo;
    return b;
  }

  static crossover(a,b){
    const ch=a.clone();
    // Crossover expert par expert
    for(let i=0;i<ch.experts.length;i++){
      if(Math.random()<0.5) ch.experts[i]=GRUExpert.crossover(a.experts[i],b.experts[i]);
    }
    // Crossover arbitre couche par couche
    if(Math.random()<0.5) ch.arb1=DenseLayer.crossover(a.arb1,b.arb1);
    if(Math.random()<0.5) ch.arb2=DenseLayer.crossover(a.arb2,b.arb2);
    if(Math.random()<0.3) ch.arb3=DenseLayer.crossover(a.arb3,b.arb3); // sortie : rarement
    ch.fitness=(a.fitness+b.fitness)/2; ch.elo=Math.round((a.elo+b.elo)/2);
    return ch;
  }

  topologyLabel(){ return `GRU(H=${this._H})×${this.experts.length}+ARB(${this._arb})`; }

  paramCount(){
    return this.experts.reduce((s,e)=>s+e.paramCount(),0)
          +(this.arb1?.paramCount()||0)+(this.arb2?.paramCount()||0)+(this.arb3?.paramCount()||0);
  }

  serialize(){
    return{
      type:'multi-expert-v4',
      experts:this.experts.map(e=>e.serialize()),
      arb1:this.arb1.serialize(),
      arb2:this.arb2.serialize(),
      arb3:this.arb3.serialize(),
      _H:this._H, _out:this._out, _arb:this._arb,
    };
  }

  static deserialize(d){
    if(!d) return null;
    // v4 : topologie actuelle (Expert 2 inSize=12, N_INPUTS=26)
    if(d.type==='multi-expert-v4'){
      const b=MultiExpertBrain._bare(d._H||12, d._out||4, d._arb||16);
      b.experts=d.experts.map(e=>GRUExpert.deserialize(e));
      b.arb1=DenseLayer.deserialize(d.arb1);
      b.arb2=DenseLayer.deserialize(d.arb2);
      b.arb3=DenseLayer.deserialize(d.arb3);
      return b;
    }
    // v3/v2 → migration : Expert 2 avait inSize=6, maintenant 12 (poids incompatibles)
    // On préserve experts 0,1,3 + arbitrateurs, Expert 2 repart de zéro
    if(d.type==='multi-expert-v3'||d.type==='multi-expert-v2'){
      const b=new MultiExpertBrain(d._H||12, d._out||4, d._arb||16);
      b.experts[0]=GRUExpert.deserialize(d.experts[0]);
      b.experts[1]=GRUExpert.deserialize(d.experts[1]);
      // experts[2] : fresh — inSize 6→12 incompatible
      b.experts[3]=GRUExpert.deserialize(d.experts[3]);
      b.arb1=DenseLayer.deserialize(d.arb1);
      b.arb2=DenseLayer.deserialize(d.arb2);
      b.arb3=DenseLayer.deserialize(d.arb3);
      return b;
    }
    // v1 et très anciens formats denses : topologie incompatible
    return null;
  }
}

// Désérialisation universelle : un format inconnu ou obsolète donne un cerveau neuf
function deserializeBrain(data){
  return MultiExpertBrain.deserialize(data) || new MultiExpertBrain();
}

// ================================================================
//  INDEXEDDB STORE
class Store {
  constructor(){this.db=null;this.ready=this._init()}
  _init(){
    return new Promise((res,rej)=>{
      if(!self.indexedDB) return rej(new Error('IndexedDB indisponible (navigation privée ?)'));
      const r=indexedDB.open('NeuralArena',3);
      r.onupgradeneeded=e=>{
        const db=e.target.result;
        if(!db.objectStoreNames.contains('brains'))db.createObjectStore('brains',{keyPath:'id'});
        if(!db.objectStoreNames.contains('config'))db.createObjectStore('config',{keyPath:'key'});
      };
      r.onsuccess=e=>{
        this.db=e.target.result;
        // Une autre version de l'app (autre onglet) veut mettre à jour la base
        this.db.onversionchange=()=>{ this.db.close(); notify('Mise à jour : recharge la page'); };
        res();
      };
      r.onblocked=()=>notify('Ferme les autres onglets Neural Arena pour continuer');
      r.onerror=()=>rej(r.error||new Error('Ouverture IndexedDB impossible'));
    });
  }
  // Résout au commit de la transaction (pas avant) : une écriture annulée est signalée.
  async _tx(store,mode,fn){
    await this.ready;
    return new Promise((res,rej)=>{
      const tx=this.db.transaction(store,mode);
      const r=fn(tx.objectStore(store));
      tx.oncomplete=()=>res(r.result!==undefined?r.result:null);
      tx.onerror=()=>rej(tx.error||r.error||new Error('Erreur IndexedDB'));
      tx.onabort=()=>rej(tx.error||new Error('Transaction IndexedDB annulée'));
    });
  }
  async put(store,data){return this._tx(store,'readwrite',s=>s.put(data))}
  async get(store,key){return this._tx(store,'readonly',s=>s.get(key))}
  async getAll(store){return this._tx(store,'readonly',s=>s.getAll())}
  async del(store,key){return this._tx(store,'readwrite',s=>s.delete(key))}
  async setConfig(key,val){return this.put('config',{key,val})}
  async getConfig(key,def=null){const r=await this.get('config',key);return r?r.val:def}
}

// Couleurs par slot : Bleu, Rouge, Vert, Jaune
const SLOT_COLORS=['#60c8ff','#ff4444','#60ff80','#f0c040'];
const SLOT_NAMES=['P1','P2','P3','P4'];

const GRAV=0.38,FRIC=0.82,JUMP=-10,MSPD=3.5;
const ASPD=13,AGRAV=0.18,ADRAG=0.998,MAXA=5,COOLDOWN=26;

// ================================================================
//  ARÈNES — cartes en tuiles sur un seul écran, bords qui se rebouclent
//  Sortir par la gauche fait réapparaître à droite ; tomber dans un
//  puits fait réapparaître en haut (et inversement).
// ================================================================
const TILE=20, MAP_COLS=40, MAP_ROWS=24;
const WW=MAP_COLS*TILE, WH=MAP_ROWS*TILE;   // 800×480
const T_EMPTY=0, T_SOLID=1, T_ONEWAY=2, T_CRUMBLE=3;

const wrapX  = x  => ((x%WW)+WW)%WW;
const wrapY  = y  => ((y%WH)+WH)%WH;
const wrapDX = dx => dx - WW*Math.round(dx/WW);   // plus court écart horizontal
const wrapDY = dy => dy - WH*Math.round(dy/WH);

// Une flèche (point) touche-t-elle l'entité ? (tient compte du rebouclage)
function arrowHitsEnt(a, e){
  return Math.abs(wrapDX(a.x-(e.x+e.w/2))) < e.w/2 && Math.abs(wrapDY(a.y-(e.y+e.h/2))) < e.h/2;
}

function hashStr(s){ let h=2166136261; for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); } return h>>>0; }
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

// '#' pierre · '.' vide · '-' plateforme traversable (par en dessous)
// 'x' bloc friable (pièges « fragiles » activés) · 'S' point d'apparition
const ARENA_DEFS = [
  { id:'crypte', name:'Crypte', rows:[
      '########....################....########',
      '##....................................##',
      '#......................................#',
      '#......................................#',
      '#...S..............................S...#',
      '#..######.......########.......######..#',
      '#..##.............####.............##..#',
      '#......................................#',
      '#..........S................S..........#',
      '#.........-------......-------.........#',
      '#......................................#',
      '........................................',
      '........................................',
      '..S..................................S..',
      '#####.......###..........###.......#####',
      '####........##............##........####',
      '#......................................#',
      '#.......S......................S.......#',
      '#......####..................####......#',
      '#......................................#',
      '#.................xxxx.................#',
      '#......................................#',
      '#.S..................................S.#',
      '########....################....########',
    ] },
  { id:'forge', name:'Forge', rows:[
      '########################################',
      '#......................................#',
      '#......................................#',
      '#..S................................S..#',
      '#.#####........----------........#####.#',
      '#.##................................##.#',
      '#..................##..................#',
      '#.........S........##........S.........#',
      '#........#####.....##.....#####........#',
      '#.........###......##......###.........#',
      '...................##...................',
      '...................##...................',
      '..S................##................S..',
      '######.............##.............######',
      '#####.......S......##......S.......#####',
      '#..........#####..####..#####..........#',
      '#......................................#',
      '#......................................#',
      '#....xxxx......................xxxx....#',
      '#......................................#',
      '#......................................#',
      '#......S........................S......#',
      '#....#####....................#####....#',
      '########################################',
    ] },
  { id:'jardins', name:'Jardins suspendus', rows:[
      '#####......##############......#########',
      '#.......................##.............#',
      '#.......................##.............#',
      '#..S.....................#.....S.......#',
      '#.######..........S.......----#######..#',
      '#.##.............####..........##......#',
      '#.................##...................#',
      '#......................................#',
      '#..........S...........................#',
      '#.........######.............S.........#',
      '#.........#.................#####......#',
      '........................................',
      '........................................',
      '.S..................................S...',
      '####.........------............#########',
      '###............................#########',
      '#......................S...............#',
      '#.....................####.............#',
      '#......S...............................#',
      '#.....####.....................xxxx....#',
      '#......................................#',
      '#..............S.......................#',
      '#..S.........#####................S....#',
      '#####......##############......#########',
    ] },
  { id:'beffroi', name:'Beffroi', rows:[
      '##########..################..##########',
      '#......................................#',
      '#......................................#',
      '#.......S......................S.......#',
      '#.....#####..................#####.....#',
      '#.....##........................##.....#',
      '#..................##..................#',
      '#.S................##................S.#',
      '####...............##...............####',
      '###........S.......##.......S........###',
      '#.........----.....##.....----.........#',
      '#..................##..................#',
      '#..................##..................#',
      '#....S.............##.............S....#',
      '#...######.........##.........######...#',
      '........................................',
      '........................................',
      '..............S..........S..............',
      '##...........xxxxxxxxxxxxxx...........##',
      '#......................................#',
      '#......................................#',
      '#...S..............................S...#',
      '#.#####..........................#####.#',
      '##########..################..##########',
    ] }
];

// Carte du parcours de sélection (gauntlet) : sol plat fermé, pas de rebouclage
const GAUNTLET_DEF = { id:'parcours', name:'Parcours', theme:'crypte', rows:[
  '#'.repeat(MAP_COLS),
  ...Array.from({length:MAP_ROWS-3}, ()=>'#'+'.'.repeat(MAP_COLS-2)+'#'),
  '#'.repeat(MAP_COLS),
  '#'.repeat(MAP_COLS),
]};
const GL_FLOOR_Y  = (MAP_ROWS-2)*TILE;     // haut du sol du parcours
const GL_FINISH_X = WW - 100;              // ligne d'arrivée étape 1
const GL_MAX_S1 = 900;  // 15 s étape 1
const GL_MAX_S2 = 900;  // 15 s étape 2
const GL_MAX_S3 = 900;  // 15 s étape 3

const ARENA_THEMES = {
  crypte:  { bg:'#0d0816', brick:'#150e24', brick2:'#1a122c', mortar:'#09050f', stone:'#3a2d58', hi:'#6a58a8', lo:'#1d1533', speck:'#2b2145', plank:'#7a5a3a', plankHi:'#a87e4c', deco:'arches', glow:'rgba(128,96,255,0.10)' },
  forge:   { bg:'#120605', brick:'#1f0d08', brick2:'#26110a', mortar:'#0a0302', stone:'#4a2a20', hi:'#b0602c', lo:'#22110c', speck:'#38201a', plank:'#5a3e28', plankHi:'#8a6038', deco:'embers', glow:'rgba(255,96,24,0.14)' },
  jardins: { bg:'#06100b', brick:'#0c1a13', brick2:'#102219', mortar:'#030805', stone:'#2e4434', hi:'#5f9a46', lo:'#152418', speck:'#22362a', plank:'#6a5434', plankHi:'#92744a', deco:'vines', glow:'rgba(96,200,96,0.08)' },
  beffroi: { bg:'#080d1a', brick:'#0f172b', brick2:'#131d35', mortar:'#050812', stone:'#33405c', hi:'#7d96c4', lo:'#18203a', speck:'#27324e', plank:'#5a4a3a', plankHi:'#8a7050', deco:'moon', glow:'rgba(160,190,255,0.10)' },
};

const _arenaCacheStore = new Map(); // rendu statique partagé : clé = id + fragiles actifs

class Arena {
  // opts.crumble : les blocs 'x' sont friables (sinon pierre ordinaire)
  constructor(def, opts={}){
    this.def=def; this.id=def.id; this.name=def.name;
    this.theme=ARENA_THEMES[def.theme||def.id]||ARENA_THEMES.crypte;
    this.crumbleOn=!!opts.crumble;
    this.base=new Uint8Array(MAP_COLS*MAP_ROWS);
    this.spawnCells=[];
    def.rows.forEach((row,ty)=>{
      for(let tx=0;tx<MAP_COLS;tx++){
        const ch=row[tx]; let t=T_EMPTY;
        if(ch==='#') t=T_SOLID;
        else if(ch==='-') t=T_ONEWAY;
        else if(ch==='x') t=this.crumbleOn?T_CRUMBLE:T_SOLID;
        else if(ch==='S') this.spawnCells.push({tx,ty});
        this.base[ty*MAP_COLS+tx]=t;
      }
    });
    this.grid=this.base.slice();
    this.surfaces=this._findSurfaces();
    this.torches=this._pickTorches();
    this._initCrumbles();
  }

  // Nouveau round : blocs friables restaurés
  reset(){ this.grid.set(this.base); this._initCrumbles(); }

  _initCrumbles(){
    this.crumbles=[];
    for(let i=0;i<this.base.length;i++) if(this.base[i]===T_CRUMBLE)
      this.crumbles.push({tx:i%MAP_COLS, ty:Math.floor(i/MAP_COLS), state:0, t:0, dy:0, vy:0});
  }

  tileAt(tx,ty){
    tx=((tx%MAP_COLS)+MAP_COLS)%MAP_COLS; ty=((ty%MAP_ROWS)+MAP_ROWS)%MAP_ROWS;
    return this.grid[ty*MAP_COLS+tx];
  }
  solidTile(tx,ty){ const t=this.tileAt(tx,ty); return t===T_SOLID||t===T_CRUMBLE; }
  solidAt(px,py){ return this.solidTile(Math.floor(px/TILE), Math.floor(py/TILE)); }
  boxSolid(x,y,w,h){
    const x0=Math.floor(x/TILE), x1=Math.floor((x+w-0.001)/TILE);
    const y0=Math.floor(y/TILE), y1=Math.floor((y+h-0.001)/TILE);
    for(let ty=y0;ty<=y1;ty++) for(let tx=x0;tx<=x1;tx++) if(this.solidTile(tx,ty)) return true;
    return false;
  }
  // Plateforme traversable franchie par les pieds en descendant ? Renvoie le haut de la tuile ou -1
  oneWayLanding(x,w,oldBottom,newBottom){
    const ty=Math.floor(newBottom/TILE), top=ty*TILE;
    if(oldBottom>top+0.01 || newBottom<=top) return -1;
    const x0=Math.floor(x/TILE), x1=Math.floor((x+w-0.001)/TILE);
    for(let tx=x0;tx<=x1;tx++) if(this.tileAt(tx,ty)===T_ONEWAY) return top;
    return -1;
  }
  // Distance normalisée [0,1] jusqu'au premier obstacle (1 = rien sur maxDist)
  raycast(ox,oy,dx,dy,maxDist){
    const len=Math.hypot(dx,dy), sx=dx/len*5, sy=dy/len*5;
    let x=ox, y=oy;
    for(let d=5; d<=maxDist; d+=5){
      x+=sx; y+=sy;
      const t=this.tileAt(Math.floor(x/TILE), Math.floor(y/TILE));
      if(t===T_SOLID||t===T_CRUMBLE||(t===T_ONEWAY&&dy>0)) return d/maxDist;
    }
    return 1;
  }
  // Un obstacle coupe-t-il le segment ? (pour la ligne de tir)
  lineBlocked(x0,y0,x1,y1){
    const n=Math.ceil(Math.hypot(x1-x0,y1-y0)/10);
    for(let i=1;i<n;i++){ const t=i/n; if(this.solidAt(x0+(x1-x0)*t, y0+(y1-y0)*t)) return true; }
    return false;
  }

  // Suites de tuiles où l'on peut se tenir debout : {tx,ty,n,x,y,w} (y = haut du sol)
  _findSurfaces(){
    const out=[];
    const isFloor=(tx,ty)=>{ const t=this.base[ty*MAP_COLS+tx]; return t===T_SOLID||t===T_ONEWAY||t===T_CRUMBLE; };
    const isFree =(tx,ty)=>{ const t=this.base[((ty+MAP_ROWS)%MAP_ROWS)*MAP_COLS+tx]; return t===T_EMPTY||t===T_ONEWAY; };
    for(let ty=1;ty<MAP_ROWS;ty++){
      let start=-1;
      for(let tx=0;tx<=MAP_COLS;tx++){
        const ok = tx<MAP_COLS && isFloor(tx,ty) && isFree(tx,ty-1) && isFree(tx,ty-2) && this.base[(ty-1)*MAP_COLS+tx]===T_EMPTY;
        if(ok && start<0) start=tx;
        if(!ok && start>=0){ const n=tx-start; out.push({tx:start,ty,n,x:start*TILE,y:ty*TILE,w:n*TILE}); start=-1; }
      }
    }
    return out;
  }

  _pickTorches(){
    const rnd=mulberry32(hashStr(this.id+':torches'));
    const cands=this.surfaces.filter(s=>s.n>=3 && s.ty>4);
    const out=[];
    for(let k=0;k<8 && out.length<5;k++){
      const s=cands[Math.floor(rnd()*cands.length)]; if(!s) break;
      const x=s.x+TILE/2+Math.floor(rnd()*s.n)*TILE, y=s.y-34;
      if(out.every(o=>Math.abs(o.x-x)>120||Math.abs(o.y-y)>60)) out.push({x,y});
    }
    return out;
  }

  // N positions d'apparition éloignées les unes des autres (distance rebouclée)
  randomSpawns(n, minDist=200){
    const cells=this.spawnCells.slice();
    for(let i=cells.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [cells[i],cells[j]]=[cells[j],cells[i]]; }
    const pos=[];
    for(let md=minDist; pos.length<n && md>=0; md-=40){
      for(const c of cells){
        if(pos.length>=n) break;
        const x=c.tx*TILE, y=(c.ty+1)*TILE-32;
        if(pos.some(p=>p.x===x&&p.y===y)) continue;
        if(pos.every(p=>Math.hypot(wrapDX(p.x-x),wrapDY(p.y-y))>=md))
          pos.push({x, y, face: x<WW/2?1:-1});
      }
    }
    while(pos.length<n){ const c=cells[pos.length%cells.length]; pos.push({x:c.tx*TILE, y:(c.ty+1)*TILE-32, face:1}); }
    return pos;
  }

  // Blocs friables : tremblent quand on se tient dessus, puis tombent pour le reste du round
  updateCrumbles(ents){
    for(const c of this.crumbles){
      if(c.state===0){
        const cx=c.tx*TILE+TILE/2, top=c.ty*TILE;
        for(const e of ents){
          if(!e || !e.gnd) continue;
          if(Math.abs(wrapDY(e.y+e.h-top))<2 && Math.abs(wrapDX(e.x+e.w/2-cx))<(e.w+TILE)/2-1){ c.state=1; c.t=0; break; }
        }
      } else if(c.state===1){
        if(++c.t>=36){ c.state=2; this.grid[c.ty*MAP_COLS+c.tx]=T_EMPTY; }
      } else if(c.dy<WH){
        c.vy+=0.4; c.dy+=c.vy;
      }
    }
  }

  // Pièges placés d'après la géométrie de la carte (toujours au même endroit pour une carte donnée)
  buildTraps(config){
    const rnd=mulberry32(hashStr(this.id+':traps'));
    const traps=[];
    const spawnPts=this.spawnCells.map(c=>({x:c.tx*TILE+10, y:(c.ty+1)*TILE}));
    const clearOf=(x,y,pts,d)=>pts.every(p=>Math.hypot(wrapDX(p.x-x),wrapDY(p.y-y))>d);
    const solidFloors=this.surfaces.filter(s=>this.base[s.ty*MAP_COLS+s.tx]===T_SOLID);
    const floorTiles=[];
    for(const s of solidFloors) for(let i=0;i<s.n;i++) floorTiles.push({x:(s.tx+i)*TILE+TILE/2, y:s.y});
    const pickTiles=(count,minGap)=>{
      const used=[];
      for(let k=0;k<200 && used.length<count;k++){
        const t=floorTiles[Math.floor(rnd()*floorTiles.length)]; if(!t) break;
        if(clearOf(t.x,t.y,spawnPts,70) && clearOf(t.x,t.y,used,minGap)) used.push(t);
      }
      return used;
    };
    if(config.mine) for(const t of pickTiles(3,140)) traps.push(new Trap('mine', t.x, t.y-6));
    if(config.electric){
      const long=solidFloors.filter(s=>s.n>=4);
      for(let k=0;k<2 && long.length;k++){
        const s=long.splice(Math.floor(rnd()*long.length),1)[0];
        traps.push(new Trap('electric', s.x+Math.floor(rnd()*(s.n-3))*TILE, s.y-6));
      }
    }
    if(config.teleport){
      const [a]=pickTiles(1,0);
      const b=a && floorTiles.filter(t=>Math.hypot(wrapDX(t.x-a.x),wrapDY(t.y-a.y))>320 && clearOf(t.x,t.y,spawnPts,70))
                         .sort(()=>rnd()-0.5)[0];
      if(a&&b) traps.push(new Trap('teleport', a.x, a.y-16, {partnerX:b.x, partnerY:b.y-16}));
    }
    if(config.launcher){
      // Tourelle dans un mur, face à un couloir libre ; dalle de déclenchement au sol devant elle
      const cands=[];
      for(let ty=2;ty<MAP_ROWS-2;ty++) for(let tx=0;tx<MAP_COLS;tx++){
        if(this.base[ty*MAP_COLS+tx]!==T_SOLID) continue;
        for(const dir of [1,-1]){
          let free=true;
          for(let k=1;k<=8&&free;k++) for(const dy of [-1,0,1]) if(this.tileAt(tx+dir*k,ty+dy)!==T_EMPTY) free=false;
          if(free) cands.push({x:tx*TILE+(dir>0?TILE:0), y:ty*TILE+TILE/2, dir});
        }
      }
      for(const dir of [1,-1]){
        const list=cands.filter(c=>c.dir===dir); if(!list.length) continue;
        const L=list[Math.floor(rnd()*list.length)];
        const btn=floorTiles.filter(t=>Math.sign(wrapDX(t.x-L.x))===dir && Math.abs(wrapDX(t.x-L.x))<260 && clearOf(t.x,t.y,spawnPts,50))
                            .sort((p,q)=>Math.abs(p.y-L.y)-Math.abs(q.y-L.y))[0];
        if(btn) traps.push(new Trap('launcher', L.x, L.y, {dir, btnX:btn.x, btnY:btn.y}));
      }
    }
    if(config.wind){
      const col=6+Math.floor(rnd()*(MAP_COLS-18));
      traps.push(new Trap('wind', col*TILE, 0, {w:6*TILE, h:WH, dir:rnd()<0.5?1:-1, strength:2}));
    }
    return traps;
  }

  // ── Rendu ──────────────────────────────────────────────────────
  draw(ctx, t){
    ctx.drawImage(this._cache(), 0, 0);
    for(const tc of this.torches) drawTorch(ctx, tc.x, tc.y, t);
    for(const c of this.crumbles) this._drawCrumble(ctx, c);
  }

  _drawCrumble(ctx, c){
    if(c.state===2 && c.dy>=WH) return;
    const th=this.theme;
    const shake = c.state===1 ? Math.round(Math.sin(c.t*2.1)*1.5) : 0;
    const x=c.tx*TILE+shake, y=c.ty*TILE+c.dy;
    ctx.globalAlpha = c.state===2 ? Math.max(0,1-c.dy/160) : 1;
    ctx.fillStyle=th.stone; ctx.fillRect(x,y,TILE,TILE);
    ctx.fillStyle=th.hi; ctx.fillRect(x,y,TILE,2);
    ctx.fillStyle=th.lo; ctx.fillRect(x,y+TILE-3,TILE,3);
    // fissures
    ctx.fillStyle='#120a08';
    ctx.fillRect(x+4,y+4,1,6); ctx.fillRect(x+5,y+9,4,1); ctx.fillRect(x+12,y+6,1,8); ctx.fillRect(x+13,y+13,3,1);
    ctx.fillStyle = c.state===1 ? 'rgba(255,170,60,0.35)' : 'rgba(255,170,60,0.12)';
    ctx.fillRect(x,y,TILE,TILE);
    ctx.globalAlpha=1;
  }

  _cache(){
    const key=this.id+(this.crumbleOn?':c':'');
    let cv=_arenaCacheStore.get(key);
    if(!cv){ cv=this._render(); _arenaCacheStore.set(key,cv); }
    return cv;
  }

  _render(){
    const cv=document.createElement('canvas'); cv.width=WW; cv.height=WH;
    const g=cv.getContext('2d'); const th=this.theme;
    const rnd=mulberry32(hashStr(this.id+':art'));
    const solid=(tx,ty)=>{ tx=((tx%MAP_COLS)+MAP_COLS)%MAP_COLS; ty=((ty%MAP_ROWS)+MAP_ROWS)%MAP_ROWS; const t=this.base[ty*MAP_COLS+tx]; return t===T_SOLID||(t===T_CRUMBLE&&false); };

    // Mur du fond : briques décalées avec variations
    g.fillStyle=th.mortar; g.fillRect(0,0,WW,WH);
    for(let y=0,row=0;y<WH;y+=10,row++){
      for(let x=-(row%2)*10;x<WW;x+=20){
        g.fillStyle = rnd()<0.18 ? th.brick2 : th.brick;
        g.fillRect(x+1,y+1,18,8);
        if(rnd()<0.08){ g.fillStyle=th.mortar; g.fillRect(x+3+Math.floor(rnd()*12),y+3,2,2); }
      }
    }
    this._renderDeco(g, rnd);

    // Tuiles pleines : bord clair en haut si exposé, ombre en bas, joints
    for(let ty=0;ty<MAP_ROWS;ty++) for(let tx=0;tx<MAP_COLS;tx++){
      const tt=this.base[ty*MAP_COLS+tx];
      const x=tx*TILE, y=ty*TILE;
      if(tt===T_SOLID){
        g.fillStyle=th.stone; g.fillRect(x,y,TILE,TILE);
        // pavés internes (2 par tuile, décalés une rangée sur deux)
        g.fillStyle=th.lo;
        g.fillRect(x, y+9, TILE, 1);
        g.fillRect(x+(ty%2?5:14), y, 1, 9);
        g.fillRect(x+(ty%2?14:5), y+10, 1, 10);
        for(let k=0;k<3;k++){ if(rnd()<0.5){ g.fillStyle=th.speck; g.fillRect(x+1+Math.floor(rnd()*17), y+1+Math.floor(rnd()*17), 2, 1); } }
        if(!solid(tx,ty-1)){ g.fillStyle=th.hi; g.fillRect(x,y,TILE,3); g.fillStyle=th.stone; g.fillRect(x,y+3,TILE,1); }
        if(!solid(tx,ty+1)){ g.fillStyle=th.lo; g.fillRect(x,y+TILE-3,TILE,3); }
        if(!solid(tx-1,ty)){ g.fillStyle=th.hi; g.globalAlpha=0.5; g.fillRect(x,y,2,TILE); g.globalAlpha=1; }
        if(!solid(tx+1,ty)){ g.fillStyle=th.lo; g.fillRect(x+TILE-2,y,2,TILE); }
        if(th.deco==='vines' && !solid(tx,ty-1)){
          g.fillStyle='#4f8f3a'; g.fillRect(x,y,TILE,2);
          if(rnd()<0.45){ g.fillStyle='#3d7a2e'; g.fillRect(x+Math.floor(rnd()*16),y-2,3,2); }
        }
        if(th.deco==='embers' && !solid(tx,ty-1) && rnd()<0.3){ g.fillStyle='#ff7a20'; g.fillRect(x+Math.floor(rnd()*17),y+1,2,1); }
      } else if(tt===T_ONEWAY){
        g.fillStyle=th.plank; g.fillRect(x,y,TILE,5);
        g.fillStyle=th.plankHi; g.fillRect(x,y,TILE,1);
        g.fillStyle='rgba(0,0,0,0.35)'; g.fillRect(x+TILE-1,y,1,5);
        if(tx%3===0){ g.fillStyle=th.plank; g.fillRect(x+8,y+5,3,4); }
      }
    }

    // Vignette
    const vg=g.createRadialGradient(WW/2,WH/2,WH*0.35,WW/2,WH/2,WW*0.7);
    vg.addColorStop(0,'rgba(0,0,0,0)'); vg.addColorStop(1,'rgba(0,0,0,0.45)');
    g.fillStyle=vg; g.fillRect(0,0,WW,WH);
    return cv;
  }

  // Décor de fond propre à chaque thème (dessiné avant les tuiles)
  _renderDeco(g, rnd){
    const th=this.theme;
    if(th.deco==='arches'){
      for(const ax of [130, 330, 470, 670]){
        g.fillStyle='rgba(0,0,0,0.35)';
        g.fillRect(ax-24,120,48,140); g.beginPath(); g.arc(ax,120,24,Math.PI,0); g.fill();
        g.fillStyle=th.glow; g.fillRect(ax-16,128,32,124); g.beginPath(); g.arc(ax,128,16,Math.PI,0); g.fill();
      }
    } else if(th.deco==='embers'){
      const lg=g.createLinearGradient(0,WH*0.55,0,WH);
      lg.addColorStop(0,'rgba(255,80,10,0)'); lg.addColorStop(1,'rgba(255,80,10,0.22)');
      g.fillStyle=lg; g.fillRect(0,0,WW,WH);
      g.fillStyle='rgba(0,0,0,0.4)';
      for(const cx of [90, 250, 550, 710]){ for(let y=0;y<150;y+=8){ g.fillRect(cx,y,3,5); } g.fillRect(cx-4,150,11,6); }
    } else if(th.deco==='vines'){
      for(let k=0;k<26;k++){
        const x=Math.floor(rnd()*WW), len=40+Math.floor(rnd()*160);
        g.fillStyle='rgba(60,120,50,0.35)';
        for(let y=0;y<len;y+=4){ g.fillRect(x+Math.round(Math.sin(y*0.12+k)*2),y,2,4); if(rnd()<0.12) g.fillRect(x-2,y,6,3); }
      }
    } else if(th.deco==='moon'){
      g.fillStyle='rgba(0,0,0,0.35)'; g.beginPath(); g.arc(WW/2,96,58,0,Math.PI*2); g.fill();
      g.fillStyle='rgba(170,200,255,0.16)'; g.beginPath(); g.arc(WW/2,96,50,0,Math.PI*2); g.fill();
      g.fillStyle='rgba(0,0,0,0.45)'; g.fillRect(WW/2-2,46,4,100); g.fillRect(WW/2-50,94,100,4);
      const lg=g.createLinearGradient(WW/2,96,WW/2,WH);
      lg.addColorStop(0,'rgba(170,200,255,0.10)'); lg.addColorStop(1,'rgba(170,200,255,0)');
      g.fillStyle=lg; g.beginPath(); g.moveTo(WW/2-40,110); g.lineTo(WW/2+40,110); g.lineTo(WW/2+150,WH); g.lineTo(WW/2-150,WH); g.fill();
    }
  }
}

// Arènes disponibles (cartes de combat, sans le parcours)
function arenaById(id){ return ARENA_DEFS.find(d=>d.id===id) || ARENA_DEFS[0]; }
// 'random' → carte tirée au hasard
function makeArena(mapId, traps={}){
  const def = (!mapId || mapId==='random') ? ARENA_DEFS[Math.floor(Math.random()*ARENA_DEFS.length)] : arenaById(mapId);
  return new Arena(def, {crumble:!!traps.collapse});
}

// Transformation monde → écran : l'arène entière (fit) ou l'écran rempli (cover).
// top : hauteur réservée en haut (bandeau du jeu)
function arenaView(W,H,cover=false,top=0){
  const h=H-top;
  const s = cover ? Math.max(W/WW,h/WH) : Math.min(W/WW,h/WH);
  return { s, ox:(W-WW*s)/2, oy:top+(h-WH*s)/2 };
}

// Dessine e (et sa copie de l'autre côté quand son corps dépasse un bord rebouclé)
function drawWrapped(ctx, e, fn){
  const xs=[0]; if(e.x>WW-e.w-4) xs.push(-WW);
  const ys=[0]; if(e.y>WH-e.h-4) ys.push(-WH);
  for(const ox of xs) for(const oy of ys){
    if(!ox && !oy){ fn(); continue; }
    ctx.save(); ctx.translate(ox,oy); fn(); ctx.restore();
  }
}

// Scène complète en coordonnées monde (le contexte est déjà transformé et découpé)
function drawArenaScene(ctx, arena, sc){
  ctx.imageSmoothingEnabled=false;
  arena.draw(ctx, sc.t||0);
  for(const tr of sc.traps||[]) tr.draw(ctx);
  for(const p of sc.trapProj||[]){ ctx.fillStyle='#ff8020'; ctx.fillRect(Math.round(p.x-3),Math.round(p.y-1),6,2); }
  for(const e of sc.ents||[]) if(e) drawWrapped(ctx, e, ()=>e.draw(ctx));
  for(const p of sc.particles||[]){
    const alpha=Math.max(0,Math.min(1,p.life/45));
    ctx.fillStyle=p.col+(Math.round(alpha*255).toString(16).padStart(2,'0'));
    ctx.fillRect(Math.round(p.x),Math.round(p.y),Math.ceil(p.sz||2),Math.ceil(p.sz||2));
  }
}

// Projectiles des pièges : vol, rebouclage, plantés dans le décor, touches
function updateTrapProjectiles(projs, arena, ents, onHit){
  for(let i=projs.length-1;i>=0;i--){
    const p=projs[i];
    if(p.st){ if(--p.life<=0) projs.splice(i,1); continue; }
    p.vy+=AGRAV; p.vx*=ADRAG; p.vy*=ADRAG;
    p.x=wrapX(p.x+p.vx); p.y=wrapY(p.y+p.vy);
    if(--p.life<=0){ projs.splice(i,1); continue; }
    if(arena.solidAt(p.x,p.y)){ p.st=true; p.vx=0; p.vy=0; p.life=Math.min(p.life,40); continue; }
    for(let j=0;j<ents.length;j++){
      const e=ents[j]; if(!e) continue;
      if(arrowHitsEnt(p,e)){ projs.splice(i,1); onHit(j,p); break; }
    }
  }
}


let particles=[];

// ================================================================
//  ENTITIES
class Ent {
  // arena : décor de collision (tuiles) dans lequel l'entité évolue
  constructor(x,y,col,isAI,arena){this.x=x;this.y=y;this.vx=0;this.vy=0;this.w=20;this.h=32;this.col=col;this.ai=isAI;this.gnd=false;this.aim=0;this.face=isAI?-1:1;this.arr=[];this.cd=0;this.ht=0;this.hp=3;this.hitFlash=0;this.arena=arena;this.dropT=0;}
  update(){
    const A=this.arena;
    this.vy+=GRAV; this.vx*=this.gnd?FRIC:0.96;
    this.vx=Math.max(-8,Math.min(8,this.vx)); this.vy=Math.max(-15,Math.min(15,this.vy));
    // Déplacement par axe contre la grille de tuiles (vitesses < 1 tuile/frame : pas de traversée)
    this.x+=this.vx;
    if(this.vx!==0 && A.boxSolid(this.x,this.y,this.w,this.h)){
      this.x = this.vx>0 ? Math.floor((this.x+this.w)/TILE)*TILE-this.w-0.001 : (Math.floor(this.x/TILE)+1)*TILE+0.001;
      this.vx=0;
    }
    const oldBottom=this.y+this.h;
    this.y+=this.vy; this.gnd=false;
    if(this.vy>0){
      if(A.boxSolid(this.x,this.y,this.w,this.h)){
        this.y=Math.floor((this.y+this.h)/TILE)*TILE-this.h; this.vy=0; this.gnd=true;
      } else if(this.dropT>0){
        this.dropT--;            // traverse volontairement les plateformes '-'
      } else {
        const top=A.oneWayLanding(this.x,this.w,oldBottom,this.y+this.h);
        if(top>=0){ this.y=top-this.h; this.vy=0; this.gnd=true; }
      }
    } else if(this.vy<0 && A.boxSolid(this.x,this.y,this.w,this.h)){
      this.y=(Math.floor(this.y/TILE)+1)*TILE; this.vy=0;
    }
    // Rebouclage : sortir d'un côté fait revenir de l'autre
    this.x=wrapX(this.x); this.y=wrapY(this.y);
    if(this.cd>0)this.cd--; if(this.ht>0)this.ht--; if(this.hitFlash>0)this.hitFlash--;
    for(let i=this.arr.length-1;i>=0;i--){
      const a=this.arr[i];
      if(a.st){ if(--a.life<=0) this.arr.splice(i,1); continue; }
      a.vy+=AGRAV; a.vx*=ADRAG; a.vy*=ADRAG;
      const nx=a.x+a.vx, ny=a.y+a.vy;
      a.x=wrapX(nx); a.y=wrapY(ny);
      if(a.x!==nx||a.y!==ny) a.tr.length=0;            // pas de traînée à travers le bord
      else { a.tr.push({x:a.x,y:a.y}); if(a.tr.length>8) a.tr.shift(); }
      if(--a.life<=0){ this.arr.splice(i,1); continue; }
      if(A.solidAt(a.x,a.y)){ a.st=true; a.ang=Math.atan2(a.vy,a.vx); a.vx=0; a.vy=0; a.life=Math.min(a.life,90); }
    }
  }
  // Se laisser tomber à travers une plateforme traversable sur laquelle on se tient
  dropThrough(){
    if(!this.gnd) return;
    const ty=Math.floor((this.y+this.h+1)/TILE);
    const x0=Math.floor(this.x/TILE), x1=Math.floor((this.x+this.w-0.001)/TILE);
    for(let tx=x0;tx<=x1;tx++) if(this.arena.solidTile(tx,ty)) return; // sol plein dessous
    this.dropT=8; this.gnd=false; this.vy=Math.max(this.vy,1);
  }
  shoot(){
    if(this.cd>0||this.arr.length>=MAXA)return;
    // 360°: direction entièrement pilotée par aim (angle absolu)
    const ax=Math.cos(this.aim),ay=Math.sin(this.aim);
    const cx=this.x+this.w/2,cy=this.y+this.h/2;
    this.arr.push({x:cx+ax*16,y:cy+ay*16,vx:ax*ASPD+this.vx*0.3,vy:ay*ASPD+this.vy*0.2,life:220,st:false,tr:[]});
    this.cd=COOLDOWN;
    // Recul
    this.vx-=ax*0.9;
  }
  draw(ctx){
    const cx=Math.round(this.x+this.w/2), cy=Math.round(this.y+this.h/2);
    const px=Math.round(this.x), py=Math.round(this.y);
    ctx.save();
    if(this.ht>0) ctx.globalAlpha=0.3+Math.sin(this.ht*0.8)*0.4;

    // Flash rouge dégât (humain seulement)
    const flashAlpha = !this.ai && this.hitFlash>0 ? (this.hitFlash/20)*0.55 : 0;
    const C = this.col;          // couleur principale ex: '#00b4ff'
    const Cd = this._darken(C);  // ombre

    // ── CAPE (derrière le corps) ──────────────────────────────────
    const capeFlap = this.gnd ? Math.sin(Date.now()*0.008)*2*Math.sign(this.vx||this.face) : this.vy*0.3;
    ctx.fillStyle = Cd;
    // Cape : rectangle avec coin bas qui flotte
    ctx.beginPath();
    ctx.moveTo(cx - this.face*2, py+8);
    ctx.lineTo(cx - this.face*10, py+8);
    ctx.lineTo(cx - this.face*12 + capeFlap, py+this.h-2);
    ctx.lineTo(cx - this.face*4,  py+this.h-2);
    ctx.closePath();
    ctx.fill();

    // ── CORPS — pixel art carré ───────────────────────────────────
    // Torse
    ctx.fillStyle = C;
    ctx.fillRect(px+2, py+8, this.w-4, this.h-16);
    // Épaule highlight
    ctx.fillStyle = this._lighten(C);
    ctx.fillRect(px+2, py+8, this.w-4, 3);

    // ── JAMBES ───────────────────────────────────────────────────
    const legAnim = this.gnd && Math.abs(this.vx)>0.5 ? Math.sin(Date.now()*0.018)*3 : 0;
    ctx.fillStyle = Cd;
    ctx.fillRect(px+3,  py+this.h-10, 6, 8+legAnim);   // jambe gauche
    ctx.fillRect(px+this.w-9, py+this.h-10, 6, 8-legAnim); // jambe droite
    // Pieds (carrés)
    ctx.fillStyle = '#1a1020';
    ctx.fillRect(px+2,  py+this.h-3, 7, 3);
    ctx.fillRect(px+this.w-9, py+this.h-3, 7, 3);

    // ── TÊTE + CAPUCHE ───────────────────────────────────────────
    // Tête
    ctx.fillStyle = '#e8c090';  // peau
    ctx.fillRect(cx-5, py, 10, 10);
    // Capuche
    ctx.fillStyle = C;
    ctx.fillRect(cx-6, py-2, 12, 8);  // haut capuche
    ctx.fillRect(cx-5, py+4, 4, 5);   // côté gauche
    ctx.fillRect(cx+1, py+4, 4, 5);   // côté droit
    // Ombre capuche
    ctx.fillStyle = Cd;
    ctx.fillRect(cx-6, py-2, 12, 2);

    // ── YEUX sous la capuche ─────────────────────────────────────
    const aimDx = Math.cos(this.aim), aimDy = Math.sin(this.aim);
    const eyeX = cx + Math.round(aimDx*2);
    const eyeY = py+5 + Math.round(aimDy*1);
    ctx.fillStyle = '#fff';
    ctx.fillRect(eyeX-1, eyeY-1, 2, 2);
    // Pupille
    ctx.fillStyle = '#000';
    ctx.fillRect(eyeX + Math.round(aimDx), eyeY + Math.round(aimDy), 1, 1);

    // ── ARC (pixel art) ──────────────────────────────────────────
    const bowDist = 14;
    const bx = Math.round(cx + aimDx*bowDist);
    const by = Math.round(cy + aimDy*10 - 4);
    ctx.save();
    ctx.translate(bx, by);
    ctx.rotate(this.aim);
    // Corps de l'arc (bois)
    ctx.fillStyle = '#c8782a';
    ctx.fillRect(-1, -10, 2, 20);
    // Extrémités courbées (pixel)
    ctx.fillStyle = '#a05820';
    ctx.fillRect(-2, -12, 4, 3);
    ctx.fillRect(-2, 9, 4, 3);
    // Corde
    ctx.fillStyle = '#e8e0c0';
    ctx.fillRect(0, -10, 1, 20);
    // Flèche sur l'arc si en train de viser
    if(this.cd===0){
      ctx.fillStyle = '#d4a030';
      ctx.fillRect(1, -2, 18, 1);
      ctx.fillStyle = '#60a030';
      ctx.fillRect(17, -3, 3, 3);
    }
    ctx.restore();

    // ── Ligne de visée joueur (pointillés pixel) ──────────────────
    if(!this.ai && this.cd===0){
      ctx.fillStyle = `${C}44`;
      for(let d=20;d<90;d+=6){
        const tx=Math.round(bx+aimDx*d), ty=Math.round(by+aimDy*d);
        ctx.fillRect(tx, ty, 2, 2);
      }
    }

    ctx.restore();

    // ── FLÈCHES EN VOL + PLANTÉES ────────────────────────────────
    for(const a of this.arr){
      const ang = Math.atan2(a.vy||0.01, a.vx||0.01);
      // Trail pixel
      for(let t=0;t<a.tr.length-1;t++){
        const alpha = (t/a.tr.length)*0.35;
        ctx.fillStyle = a.st ? `rgba(160,100,40,${alpha})` : `${C}${Math.round(alpha*255).toString(16).padStart(2,'0')}`;
        ctx.fillRect(Math.round(a.tr[t].x), Math.round(a.tr[t].y), 2, 2);
      }
      // Corps de la flèche (plantée : garde l'angle d'impact)
      ctx.save();
      ctx.translate(Math.round(a.x), Math.round(a.y));
      ctx.rotate(a.st ? (a.ang||0) : ang);
      // Hampe
      ctx.fillStyle = a.st ? '#8a6030' : '#d4a030';
      ctx.fillRect(-12, -1, 16, 2);
      // Pointe
      ctx.fillStyle = a.st ? '#666' : '#e8e8e8';
      ctx.fillRect(4, -2, 5, 4);
      ctx.fillRect(7, -1, 3, 2);
      // Plumes
      ctx.fillStyle = a.st ? '#5a4020' : (this.ai ? '#cc3344' : '#3388cc');
      ctx.fillRect(-14, -3, 4, 2);
      ctx.fillRect(-14, 1, 4, 2);
      ctx.restore();
    }

    // ── FLASH ROUGE DÉGÂT ────────────────────────────────────────
    if(flashAlpha > 0){
      ctx.save();
      ctx.globalAlpha = flashAlpha;
      ctx.fillStyle = '#ff2020';
      ctx.fillRect(px-2, py-2, this.w+4, this.h+4);
      ctx.restore();
    }

    // ── CŒURS HP ─────────────────────────────────────────────────
    const heartSize = 6;
    const totalW = this.hp * (heartSize+2) - 2;
    const hx = cx - Math.floor(totalW/2);
    const hy = py - 14;
    for(let h=0;h<3;h++){
      const full = h < this.hp;
      ctx.fillStyle = full ? '#ff3344' : 'rgba(80,40,40,0.6)';
      const hbx = hx + h*(heartSize+2);
      // Cœur pixel art simple (2 carrés + triangle)
      ctx.fillRect(hbx,   hy+1, 2, 2);
      ctx.fillRect(hbx+3, hy+1, 2, 2);
      ctx.fillRect(hbx,   hy+3, 5, 2);
      ctx.fillRect(hbx+1, hy+5, 3, 1);
      if(full){
        ctx.fillStyle='rgba(255,180,180,0.6)';
        ctx.fillRect(hbx+1,hy+1,1,1);
      }
    }
  }

  // Helpers couleur pixel art
  _darken(hex){
    const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
    return `rgb(${Math.max(0,r-60)},${Math.max(0,g-60)},${Math.max(0,b-60)})`;
  }
  _lighten(hex){
    const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
    return `rgb(${Math.min(255,r+50)},${Math.min(255,g+50)},${Math.min(255,b+50)})`;
  }
}

// ================================================================
//  INPUTS / OUTPUTS IA  — 26 inputs, 7 outputs (MultiExpertBrain v4)
//
//  INPUTS (26) :
//   0  rX          : position relative ennemi X (normalisée)
//   1  rY          : position relative ennemi Y
//   2  dist        : distance normalisée (0=près, 1=loin)
//   3  mVx         : vélocité cerveau X
//   4  mVy         : vélocité cerveau Y
//   5  eVx         : vélocité ennemi X
//   6  eVy         : vélocité ennemi Y
//   7  aim         : angle visée actuel (-1..1)
//   8  aimErr      : erreur visée vs optimal (-1..1)
//   9  aimErrPred  : erreur visée vs position prédite ennemi
//  10  onGround    : au sol (0/1)
//  11  cdNorm      : cooldown tir normalisé
//  12  nAX         : direction flèche ennemie proche X
//  13  nAY         : direction flèche ennemie proche Y
//  14  nVX         : vélocité flèche ennemie X
//  15  nVY         : vélocité flèche ennemie Y
//  16  arrowDanger : danger flèche (1=imminent, 0=aucune)
//  17  posX        : position X cerveau normalisée
//  18  posY        : position Y cerveau normalisée
//  19  t           : progression du round
//  20  sL          : distance mur gauche   (raycast, 1 = rien)
//  21  sR          : distance mur droit
//  22  sU          : distance plafond
//  23  sD          : distance sol
//  24  sDL         : plateforme en bas à gauche
//  25  sDR         : plateforme en bas à droite
//
//  OUTPUTS (7) :
//   0  moveLeft    : aller à gauche
//   1  moveRight   : aller à droite
//   2  jump        : sauter
//   3  aimLeft     : rotation visée gauche (vitesse proportionnelle)
//   4  aimRight    : rotation visée droite
//   5  shoot       : tirer
//   6  aimSnap     : snap de visée directe vers l'ennemi prédit
//

// Les écarts (cible, flèches) sont les plus courts à travers les bords rebouclés :
// l'IA voit l'adversaire « de l'autre côté » s'il est plus proche par là.
function aiInputsFor(me, target, allArrows, timer){
  const cx=me.x+me.w/2, cy=me.y+me.h/2;
  const dxT=wrapDX(target.x+target.w/2-cx), dyT=wrapDY(target.y+target.h/2-cy);

  const rX=dxT/WW;
  const rY=dyT/WH;
  const dist=Math.min(Math.sqrt(rX*rX+rY*rY)*2, 1);

  const mVx=me.vx/MSPD, mVy=me.vy/15;
  const eVx=target.vx/MSPD, eVy=target.vy/15;

  const aim=me.aim/Math.PI;

  // Angle optimal vers la position actuelle de l'ennemi
  const optAim=Math.atan2(dyT, dxT);
  let aimErr=me.aim-optAim;
  aimErr=Math.atan2(Math.sin(aimErr),Math.cos(aimErr))/Math.PI;

  // Angle optimal vers la position PRÉDITE de l'ennemi
  const tof=Math.hypot(dxT,dyT)/ASPD; // temps de vol estimé
  const k=Math.min(tof,20);
  const predAim=Math.atan2(dyT+target.vy*k, dxT+target.vx*k);
  let aimErrPred=me.aim-predAim;
  aimErrPred=Math.atan2(Math.sin(aimErrPred),Math.cos(aimErrPred))/Math.PI;

  const onG=me.gnd?1:0;
  const cdNorm=Math.min((me.cd||0)/COOLDOWN,1);

  // Flèche ennemie la plus proche + niveau de danger
  let nAX=0,nAY=0,nVX=0,nVY=0,arrowDanger=0;
  let mD=Infinity;
  for(const a of allArrows){
    if(a.st) continue;
    const dx=wrapDX(a.x-cx), dy=wrapDY(a.y-cy);
    const d=dx*dx+dy*dy;
    if(d<mD){
      mD=d;
      nAX=dx/WW; nAY=dy/WH;
      nVX=a.vx/ASPD; nVY=a.vy/ASPD;
      // Danger : flèche qui se rapproche rapidement
      const sd=Math.sqrt(d);
      const approaching=-(dx*a.vx+dy*a.vy)/Math.max(sd,1);
      arrowDanger=Math.max(0, Math.min(1, approaching/5 * (1-sd/400)));
    }
  }

  const t=Math.min((timer||0)/(60*35),1);

  // Capteurs spatiaux directionnels (indices 20–25), à travers les bords
  const A=me.arena;
  return[
    rX, rY, dist,          // 0,1,2
    mVx, mVy,              // 3,4
    eVx, eVy,              // 5,6
    aim,                   // 7
    aimErr,                // 8
    aimErrPred,            // 9
    onG,                   // 10
    cdNorm,                // 11
    nAX, nAY,              // 12,13
    nVX, nVY,              // 14,15
    arrowDanger,           // 16
    me.x/WW,               // 17
    me.y/WH,               // 18
    t,                     // 19
    A.raycast(cx, cy, -1,  0, 200),   // 20: distance mur gauche
    A.raycast(cx, cy,  1,  0, 200),   // 21: distance mur droit
    A.raycast(cx, cy,  0, -1, 300),   // 22: distance plafond
    A.raycast(cx, cy,  0,  1, 400),   // 23: distance sol
    A.raycast(cx, cy, -1,  1, 300),   // 24: plateforme bas-gauche
    A.raycast(cx, cy,  1,  1, 300),   // 25: plateforme bas-droit
  ];
}

const AIM_ASSIST_RATE = 0.06; // rad/frame (la visée manuelle, sorties 3/4, va jusqu'à 0.18)

// Politique stochastique : chaque sortie est la probabilité d'appuyer sur le bouton.
// L'IA joue partout (partie, entraînement, évaluation) en tirant ses actions selon ces
// probabilités : c'est exactement la politique que le gradient optimise. (Seuiller à 0.5
// en partie faisait par ex. qu'une IA qui tire « 30 % du temps quand c'est bien visé »
// ne tirait jamais.) Des sorties déjà binaires (0/1) restent inchangées.
function aiSample(out){
  const a=new Array(out.length);
  for(let i=0;i<out.length;i++) a[i]=Math.random()<out[i]?1:0;
  return a;
}

// Applique les sorties du réseau à l'entité. Renvoie les actions jouées (0/1).
function aiProcessFor(out, ent, target=null){
  const a=aiSample(out);
  if(a[0]){ ent.vx-=MSPD*0.4; ent.face=-1; }
  if(a[1]){ ent.vx+=MSPD*0.4; ent.face=1; }
  if(a[2] && ent.gnd) ent.vy=JUMP;
  ent.aim += (a[4]-a[3]) * 0.18;
  // Garder l'angle dans [-π, π] : sinon l'entrée aim/π sature le réseau et les
  // calculs d'écart de visée (récompenses) partent en valeurs absurdes.
  ent.aim = Math.atan2(Math.sin(ent.aim), Math.cos(ent.aim));
  if(a[5]) ent.shoot();
  // Assistance de visée (sortie 6) : tourne l'arc vers l'ennemi à vitesse limitée.
  // Ce n'est plus un placement instantané : viser juste (anticiper le mouvement et la
  // chute de la flèche avec les sorties 3/4) reste une compétence à apprendre.
  if(a[6] && target){
    const want=Math.atan2(wrapDY((target.y+target.h/2)-(ent.y+ent.h/2)), wrapDX((target.x+target.w/2)-(ent.x+ent.w/2)));
    const d=Math.atan2(Math.sin(want-ent.aim), Math.cos(want-ent.aim));
    const na=ent.aim+Math.max(-AIM_ASSIST_RATE,Math.min(AIM_ASSIST_RATE,d));
    ent.aim=Math.atan2(Math.sin(na), Math.cos(na));
  }
  return a;
}

function boom(x,y,col){
  // L'entraînement en arrière-plan appelle boom() sans boucle de rendu pour vider les particules
  if(particles.length>400) particles.splice(0, particles.length-380);
  for(let i=0;i<20;i++){
    const a=Math.random()*Math.PI*2, s=1+Math.random()*5;
    particles.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s-2,life:25+Math.random()*20,col,sz:2+Math.random()*2});
  }
  for(let i=0;i<10;i++){
    const a=Math.random()*Math.PI*2, s=2+Math.random()*7;
    particles.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s-3,life:15+Math.random()*15,col:'#f0c040',sz:1});
  }
}
function updateParticles(){for(let i=particles.length-1;i>=0;i--){const p=particles[i];p.x+=p.vx;p.y+=p.vy;p.vy+=0.1;p.life--;if(p.life<=0)particles.splice(i,1)}}

// ================================================================
//  SCORING — ELO + FITNESS
//
//  ELO  : mesure la force relative contre des adversaires.
//         K-factor dynamique selon écart ELO + nombre de parties.
//         Plus l'adversaire est fort, plus la victoire rapporte.
//
//  FITNESS (0-100) : moyenne exponentielle glissante du comportement.
//         Composantes : précision visée, tirs réussis, esquive, survie.
//         N'explose pas, converge vers la vraie performance récente.

const ELO_BASE = 1200;

// ── SKILLS : 6 dimensions de compétence ─────────────────────────────────────
function updateSkills(skills, rs, totalFrames){
  const α=0.15, maxT=60*35;
  const rawDodge    = Math.min((rs.arrowsDodged||0)/20*100, 100);
  const rawAim      = Math.min((rs.framesAiming||0)/Math.max(totalFrames,1)*250, 100);
  const rawPrec     = (rs.shotsFired||0)>0
    ? Math.min((rs.preciseShots||0)/(rs.shotsFired||1)*100, 100)
    : (skills.precision||0);
  const rawSurvival = Math.min(totalFrames/maxT*100, 100);
  const rawAggro    = (rs.shootOpps||0)>0
    ? Math.min((rs.shotsFired||0)/(rs.shootOpps||1)*100, 100) : 0;
  const rawMobility = Math.min(
    (1-(rs.stationaryFrames||0)/Math.max(totalFrames,1))*100, 100);
  const up=(old,raw)=>(old||0)*(1-α)+raw*α;
  return {
    dodge   : up(skills.dodge,    rawDodge),
    aim     : up(skills.aim,      rawAim),
    precision:up(skills.precision,rawPrec),
    survival: up(skills.survival, rawSurvival),
    aggro   : up(skills.aggro,    rawAggro),
    mobility: up(skills.mobility, rawMobility),
  };
}

function computeGlobalFitness(skills){
  if(!skills) return 0;
  return Math.min(100, Math.max(0,
    (skills.dodge||0)*0.20 + (skills.aim||0)*0.20 +
    (skills.precision||0)*0.20 + (skills.survival||0)*0.15 +
    (skills.aggro||0)*0.15 + (skills.mobility||0)*0.10));
}

function drawRadarChart(ctx, cx, cy, r, vals, labels, fillCol, strokeCol){
  const N=vals.length, step=Math.PI*2/N, off=-Math.PI/2;
  [0.25,0.5,0.75,1].forEach(p=>{
    ctx.beginPath();
    for(let i=0;i<N;i++){
      const a=off+i*step;
      const x=cx+Math.cos(a)*r*p, y=cy+Math.sin(a)*r*p;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.strokeStyle=`rgba(255,255,255,${p===1?0.15:0.07})`;
    ctx.lineWidth=0.5; ctx.stroke();
  });
  for(let i=0;i<N;i++){
    const a=off+i*step;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+Math.cos(a)*r, cy+Math.sin(a)*r);
    ctx.strokeStyle='rgba(255,255,255,0.12)'; ctx.lineWidth=0.5; ctx.stroke();
  }
  ctx.beginPath();
  for(let i=0;i<N;i++){
    const a=off+i*step, v=Math.max(0,Math.min(1,(vals[i]||0)/100));
    const x=cx+Math.cos(a)*r*v, y=cy+Math.sin(a)*r*v;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.closePath(); ctx.fillStyle=fillCol; ctx.fill();
  ctx.strokeStyle=strokeCol; ctx.lineWidth=1.5; ctx.stroke();
  ctx.font='7px monospace'; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillStyle=strokeCol;
  for(let i=0;i<N;i++){
    const a=off+i*step;
    ctx.fillText(labels[i], cx+Math.cos(a)*(r+16), cy+Math.sin(a)*(r+16));
  }
}
const SKILL_KEYS=['dodge','aim','precision','survival','aggro','mobility'];
const SKILL_LABELS=['ESQUIVE','VISÉE','PRÉCIS','SURVIE','AGGRO','MOBIL'];
const SKILL_FILL='rgba(96,200,255,0.15)';
const SKILL_STROKE='#60c8ff';

function eloExpected(ra, rb){ return 1/(1+Math.pow(10,(rb-ra)/400)); }

// K-factor dynamique : dépend du nombre de parties ET de l'écart ELO
function eloUpdate(rating, expected, score, played){
  // K de base selon maturité
  const Kbase = played < 10 ? 48 : played < 30 ? 32 : played < 100 ? 20 : 14;
  // Bonus si adversaire beaucoup plus fort (récompense les upsets)
  const surprise = Math.max(1, 1 + (1-expected)*1.5);
  const K = Kbase * surprise;
  return Math.max(400, Math.round(rating + K * (score - expected)));
}

// Gain comportemental d'un round — partagé solo/multi/entraînement
// stats : {arrowsDodged, shotsNearMiss, preciseShots, framesAiming, timer}
// Multiplicateurs de récompense par objectif d'entraînement
const OBJECTIVE_WEIGHTS = {
  balanced : { aimPrec:1.0, shootWell:1.0, arrowNear:1.0, arrowDodge:1.0, wall:1.0, stationary:-1.0, survival:1.0 },
  dodge    : { aimPrec:0.4, shootWell:1.0, arrowNear:1.0, arrowDodge:3.0, wall:2.0, stationary:-2.0, survival:1.0 },
  aim      : { aimPrec:3.0, shootWell:2.0, arrowNear:1.0, arrowDodge:1.0, wall:1.0, stationary:-1.0, survival:1.0 },
  mobility : { aimPrec:1.0, shootWell:1.0, arrowNear:1.0, arrowDodge:1.0, wall:2.0, stationary:-3.0, survival:1.0 },
  aggro    : { aimPrec:1.0, shootWell:3.0, arrowNear:2.0, arrowDodge:1.0, wall:0.5, stationary:-1.0, survival:1.0 },
};

function computeFitnessGain(stats, won, timer){
  const dodge        = Math.min((stats.arrowsDodged  ||0)*1.0,  10);
  const nearMiss     = Math.min((stats.shotsNearMiss ||0)*0.8,   8);
  const preciseShots = Math.min((stats.preciseShots  ||0)*3.0,  12); // tirs bien visés
  const aiming       = Math.min((stats.framesAiming  ||0)/(60*35)*10, 10); // % frames bien visé
  const survival     = Math.min((timer||0)/(60*35)*6, 6);
  const winBonus     = won ? evoParams.winBonus : 0;
  return dodge + nearMiss + preciseShots + aiming + survival + winBonus;
}

// Fitness bornée 0-100 : EMA plus réactive (alpha=0.12 au lieu de 0.05)
function updateFitness(currentFitness, gain, won){
  // Max théorique : 10+8+12+10+6+8 = 54 → on normalise sur 54
  const maxGain = 54;
  const scaledTarget = Math.min(gain/maxGain*100, 100);
  // EMA rapide : converge en ~20 rounds au lieu de 60
  const alpha = won ? 0.12 : 0.08;
  const next = currentFitness*(1-alpha) + scaledTarget*alpha;
  return Math.max(0, Math.min(100, next));
}

// Copie les poids d'un cerveau dans un autre (tous les experts + arbitre)
function copyBrainInto(target, src){
  target.experts = src.experts.map(e=>e.clone());
  target.arb1    = src.arb1.clone();
  target.arb2    = src.arb2.clone();
  if(src.arb3) target.arb3 = src.arb3.clone();
}

// Nouveau candidat : 1 ou 2 experts GRU mutés (la visée plus souvent), plus fort en stagnation.
// L'arbitre n'est pas muté : il apprend par gradient.
function mutateExperts(brain, stag){
  const expl  = Math.min(1 + stag*0.15, 2.5);
  const rate  = Math.min(evoParams.mutRateLoss*expl, 0.4);
  const power = Math.min(evoParams.mutPower*expl, 0.6);
  const pickW = [0.2, 0.4, 0.2, 0.2];
  const k = Math.random()<0.5 ? 1 : 2;
  for(let n=0;n<k;n++){
    let r=Math.random(), idx=0;
    while(idx<pickW.length-1 && (r-=pickW[idx])>0) idx++;
    brain.mutate(rate, power, Math.min(idx, brain.experts.length-1));
  }
  brain.resetMemory();
}

// ── Dessine une torche animée ─────────────────────────────────────
function drawTorch(ctx, x, y, t){
  const flicker = Math.sin(t*0.08)*2 + Math.sin(t*0.13)*1.5;
  // Support métal
  ctx.fillStyle='#6a5030';
  ctx.fillRect(x-2, y, 4, 10);
  ctx.fillStyle='#8a7040';
  ctx.fillRect(x-3, y-2, 6, 4);
  // Flamme (plusieurs rectangles empilés)
  const flames=[
    {w:6,h:4,col:'#ff8820',dy:0},
    {w:4,h:4,col:'#ffaa30',dy:-4+flicker},
    {w:3,h:3,col:'#ffcc44',dy:-8+flicker*0.5},
    {w:2,h:2,col:'#ffee88',dy:-11},
  ];
  for(const f of flames){
    ctx.fillStyle=f.col;
    ctx.fillRect(Math.round(x-f.w/2), Math.round(y-4+f.dy), f.w, f.h);
  }
  // Halo lumineux (disque doux)
  ctx.fillStyle=`rgba(255,140,30,${0.05+Math.abs(flicker)*0.012})`;
  ctx.beginPath(); ctx.arc(x, y-6, 22, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(x, y-6, 12, 0, Math.PI*2); ctx.fill();
}

const EVO_DEFAULTS = {
  mutRateWin  : 0.04,
  mutRateLoss : 0.16,
  mutPower    : 0.26,
  hofSize     : 8,
  winBonus    : 8,
};
let evoParams = {...EVO_DEFAULTS};

// ================================================================
//  SCRIPTED AGENT — IA codée à la main (4 niveaux)
class ScriptedAgent {
  constructor(level=1){
    this.level = level; // 1=KURO 2=SERA 3=VENN 4=RYUN
    this.reset();
  }

  // Réinitialise la mémoire entre deux rounds (toujours passer par ici :
  // un champ manquant donne NaN dans act() et l'agent ne tire plus).
  reset(){
    this.mem = {
      jumpCooldown:0, retreatDir:0, retreatTimer:0,
      patternBuf:[], losSearchDir:0, losSearchTimer:0, platSearchTimer:0,
      shootCooldown:0,  // discipline de tir (L2+)
      frozenTimer:0,    // pause confusion KURO (L1)
      missStreak:0,     // frames sans impact proche (L3+)
      prevEnemyVy:0,    // suivi vy ennemi pour détection apex (L4)
    };
  }

  act(me, enemy, enemyArrows, traps, timer){
    const m = this.mem;
    m.jumpCooldown   = Math.max(0, m.jumpCooldown-1);
    m.retreatTimer   = Math.max(0, m.retreatTimer-1);
    m.losSearchTimer = Math.max(0, m.losSearchTimer-1);
    m.platSearchTimer= Math.max(0, (m.platSearchTimer||0)-1);
    m.shootCooldown  = Math.max(0, m.shootCooldown-1);
    m.frozenTimer    = Math.max(0, m.frozenTimer-1);

    // L1 — KURO : confusion aléatoire (pause totale)
    if(this.level===1 && m.frozenTimer>0) return;
    if(this.level===1 && Math.random()<0.0015) m.frozenTimer=40+Math.floor(Math.random()*40);

    // Écart le plus court (bords rebouclés) ; ex/ey = position de l'ennemi « vue » depuis moi
    const dx   = wrapDX(enemy.x - me.x);
    const dy   = wrapDY(enemy.y - me.y);
    const ex   = me.x + dx, ey = me.y + dy;
    const dist = Math.sqrt(dx*dx+dy*dy);
    const blocked = this._shotBlocked(me, dx, dy);

    // Avantage de hauteur : suis-je au-dessus de l'ennemi ?
    const heightAdv = me.y - enemy.y; // négatif = je suis plus haut
    const hasHighGround = heightAdv < -40;
    const enemyAbove = heightAdv > 60;

    // ── TROUVER LA MEILLEURE PLATEFORME ────────────────────────
    // Chercher une plateforme au-dessus de l'ennemi pour avoir l'avantage
    let targetPlat = null;
    if(!hasHighGround && this.level >= 2){
      let bestScore = -Infinity;
      for(const p of me.arena.surfaces){
        if(p.w<60) continue; // ignorer les petites corniches
        // La plateforme doit être au-dessus de l'ennemi
        if(p.y >= enemy.y - 20) continue;
        // Score : plus haut que l'ennemi = mieux, mais pas trop loin horizontalement
        const platCX = p.x + p.w/2;
        const platCY = p.y;
        const hScore = (enemy.y - platCY) * 2; // bonus hauteur
        const dScore = -Math.abs(platCX - enemy.x) * 0.5; // pénalité distance horizontale
        const myDist = -Math.abs(platCX - me.x) * 0.3; // préférer les proches de moi
        const score = hScore + dScore + myDist;
        if(score > bestScore){ bestScore=score; targetPlat=p; }
      }
    }

    // ── MOUVEMENT ──────────────────────────────────────────────
    let moveX = 0;
    let wantJumpToPlat = false;

    if(blocked){
      // Chercher une ligne de vue libre en se déplaçant latéralement
      if(m.losSearchTimer === 0){
        m.losSearchDir = Math.random() < 0.5 ? 1 : -1;
        m.losSearchTimer = 60;
      }
      moveX = m.losSearchDir;
      if(me.x < 60)  { moveX = 1;  m.losSearchDir = 1; }
      if(me.x > WW-80){ moveX = -1; m.losSearchDir = -1; }
      // Si bloqué, sauter pour changer de plateforme
      if(m.losSearchTimer > 40 && me.gnd && m.jumpCooldown===0){
        wantJumpToPlat = true;
      }
    } else if(targetPlat && !hasHighGround && this.level >= 2){
      // ── NAVIGUER VERS LA PLATEFORME CIBLE (high ground) ──────
      const platCX = targetPlat.x + targetPlat.w/2;
      const platCY = targetPlat.y;
      const dxPlat = platCX - me.x;
      const dyPlat = platCY - me.y;

      // Se déplacer horizontalement vers la plateforme
      if(Math.abs(dxPlat) > 40){
        moveX = dxPlat > 0 ? 1 : -1;
      }
      // Sauter si on est en dessous de la plateforme cible et à peu près aligné
      if(dyPlat < -20 && Math.abs(dxPlat) < targetPlat.w && me.gnd && m.jumpCooldown===0){
        wantJumpToPlat = true;
      }
    } else if(hasHighGround){
      // ── POSITION DOMINANTE — rester au-dessus, se centrer sur l'ennemi ──
      const optOffsetX = this.level >= 3 ? 0 : 30; // MASTER: pile au dessus
      if(Math.abs(dx) > 60 + optOffsetX) moveX = dx > 0 ? 1 : -1;
      // Si l'ennemi est loin en dessous, on peut descendre pour être plus agressif
      if(this.level >= 3 && Math.abs(dy) > 250 && me.gnd && m.jumpCooldown===0){
        // Ne pas sauter, juste se rapprocher du bord pour tirer vers le bas
      }
      // Retraite après esquive
      if(m.retreatTimer > 0) moveX = m.retreatDir;
    } else {
      // Ligne de vue libre, pas de high ground target → positionnement normal
      m.losSearchTimer = 0;

      if(this.level === 1){
        // KURO : mouvement erratique — inverse parfois brusquement
        if(timer % 90 < 45) moveX = dx > 0 ? 1 : -1;
        else moveX = Math.random() < 0.5 ? 1 : -1;
        if(timer % 70 === 0 && Math.random()<0.35) moveX *= -1; // inversion surprise
      } else {
        const optDist = this.level >= 3 ? 300 : 280;
        if(dist > optDist + 80) moveX = dx > 0 ? 1 : -1;
        else if(dist < optDist - 80) moveX = dx > 0 ? -1 : 1;
        if(m.retreatTimer > 0) moveX = m.retreatDir;
      }
    }

    // ── ESQUIVE ────────────────────────────────────────────────
    let doJump = false;
    if(this.level >= 2){
      for(const a of enemyArrows){
        if(a.st) continue;
        const ax = wrapDX(a.x - me.x), ay = wrapDY(a.y - me.y);
        const adist = Math.sqrt(ax*ax+ay*ay);
        const thresh = this.level >= 3 ? 130 : 80;
        if(adist < thresh && Math.abs(ay) < me.h && a.vx * ax > 0){
          if(me.gnd && m.jumpCooldown === 0){ doJump = true; m.jumpCooldown = 35; }
          if(this.level >= 3){
            moveX = -Math.sign(a.vx);
            m.retreatDir = -Math.sign(a.vx);
            m.retreatTimer = 20;
          }
        }
        // Esquive flèches venant du dessous (plongeantes vers le haut)
        if(adist < 100 && a.vy < -2 && ay < 0 && Math.abs(ax) < 40){
          // L4 RYUN : esquive perpendiculaire au vecteur flèche
          if(this.level >= 4){
            const perpX = a.vy !== 0 ? Math.sign(a.vx * a.vy > 0 ? -1 : 1) : (Math.random()<0.5?1:-1);
            moveX = perpX; m.retreatDir=perpX; m.retreatTimer=20;
          } else {
            moveX = Math.random() < 0.5 ? 1 : -1;
            m.retreatDir = moveX; m.retreatTimer = 15;
          }
        }
      }
    }

    // ── CONSCIENCE DES PIÈGES (L3+) ────────────────────────────
    if(this.level >= 3 && moveX !== 0){
      const mineAhead = traps.filter(t=>t.type==='mine'&&t.active)
        .some(t=>Math.hypot(t.x+8-(me.x+me.w/2), t.y+8-(me.y+me.h/2))<80
                && Math.sign((t.x+8)-(me.x+me.w/2))===moveX);
      if(mineAhead){ moveX*=-1; m.retreatDir=moveX; m.retreatTimer=30; }
    }

    // ── SAUT ──────────────────────────────────────────────────
    if(!doJump && me.gnd && m.jumpCooldown === 0){
      if(wantJumpToPlat){
        doJump = true; m.jumpCooldown = 30;
      } else if(this.level === 1){
        if(Math.random() < 0.003) { doJump = true; m.jumpCooldown = 60; }
      } else {
        // Monter vers l'ennemi s'il est au-dessus
        if(enemyAbove && Math.random() < 0.04){
          doJump = true; m.jumpCooldown = 35;
        }
        // Chercher le high ground activement (même sans targetPlat)
        if(!hasHighGround && Math.random() < (this.level >= 3 ? 0.025 : 0.012)){
          doJump = true; m.jumpCooldown = 40;
        }
        if(blocked && Math.random() < 0.03){
          doJump = true; m.jumpCooldown = 40;
        }
        if(this.level >= 3 && Math.random() < 0.008){
          doJump = true; m.jumpCooldown = 55;
        }
      }
    }

    // ── VISÉE ─────────────────────────────────────────────────
    let targetAim = Math.atan2(dy, dx);

    // Bonus: si on a le high ground, viser vers le bas (angle plongeant = plus facile)
    if(hasHighGround && this.level >= 2){
      // Viser légèrement en avance pour les tirs plongeants (la gravité aide)
      const tof = dist / ASPD;
      const gravDrop = 0.5 * AGRAV * tof * tof; // la flèche tombe
      const predX = ex + enemy.vx * tof * (this.level >= 3 ? 1.0 : 0.5);
      const predY = ey + enemy.vy * tof * 0.4 - gravDrop * 0.3; // compenser la gravité
      targetAim = Math.atan2(predY - me.y, predX - me.x);
    } else if(this.level >= 2){
      const tof = dist / ASPD;
      const predX = ex + enemy.vx * tof * (this.level >= 3 ? 1.0 : 0.4);
      const predY = ey + enemy.vy * tof * (this.level >= 3 ? 0.8 : 0.2);
      targetAim = Math.atan2(predY - me.y, predX - me.x);

      // L4 RYUN : prédiction complète avec gravité, pas de bruit intentionnel
      if(this.level >= 4){
        const tof4 = dist / ASPD;
        const predX4 = ex + enemy.vx * tof4;
        const predY4 = ey + enemy.vy * tof4 - 0.5 * AGRAV * tof4 * tof4 * 0.25;
        targetAim = Math.atan2(predY4 - me.y, predX4 - me.x);
        m.patternBuf.push({x:enemy.x, y:enemy.y, vy:enemy.vy});
        if(m.patternBuf.length > 60) m.patternBuf.shift();
      }
    }

    // L1 KURO : bruit d'aim ±12°
    if(this.level === 1) targetAim += (Math.random()-0.5) * 0.42;

    // Appliquer rotation de visée (commun à tous les niveaux, une seule fois par frame)
    {
      const aimSpeed = this.level === 1 ? 0.04 : this.level === 2 ? 0.09 : this.level === 3 ? 0.14 : 0.18;
      // High ground: visée plus rapide (angle plus stable)
      const finalSpeed = hasHighGround ? aimSpeed * 1.3 : aimSpeed;
      let diff = targetAim - me.aim;
      while(diff > Math.PI) diff -= Math.PI*2;
      while(diff < -Math.PI) diff += Math.PI*2;
      me.aim += diff * finalSpeed;
    }

    // ── TIR ───────────────────────────────────────────────────
    let doShoot = false;
    let diff2 = targetAim - me.aim;
    while(diff2 > Math.PI) diff2 -= Math.PI*2;
    while(diff2 < -Math.PI) diff2 += Math.PI*2;
    const aimErr = Math.abs(diff2);

    const highGroundBonus = hasHighGround ? 1.3 : 1.0;
    const shootThresh = (this.level===1?0.50:this.level===2?0.32:this.level===3?0.15:0.12)*highGroundBonus;
    const shootChance = this.level === 1 ? 0.18 : 1.0;

    if(aimErr < shootThresh && dist < 900 && me.cd === 0 && !blocked && m.shootCooldown === 0){
      doShoot = Math.random() < shootChance;
      if(doShoot) m.shootCooldown = this.level===1?0:this.level===2?18:this.level===3?10:6;
    }

    // L4 RYUN : forcer tir au passage à l'apex du saut ennemi
    if(this.level >= 4 && me.cd === 0 && !blocked){
      const atApex = m.prevEnemyVy < 0 && enemy.vy >= 0 && !enemy.gnd;
      if(atApex && aimErr < 0.28) doShoot = true;
    }
    m.prevEnemyVy = enemy.vy;

    // L4 RYUN : mode berserk (>60% du round écoulé → distance cible 100px)
    if(this.level >= 4 && timer > 60*22 && !hasHighGround && dist > 120){
      moveX = dx > 0 ? 1 : -1; // foncer sur l'ennemi
    }

    // ── ZONE D'ESQUIVE : ne pas coller l'adversaire ──────────────
    // Au contact, l'entraîneur recule (et saute parfois) au lieu de rester collé.
    const personalSpace = [0, 60, 80, 100, 110][this.level];
    if(dist < personalSpace){
      moveX = dx > 0 ? -1 : 1;
      m.retreatDir = moveX; m.retreatTimer = 12;
      if(me.gnd && m.jumpCooldown === 0 && Math.random() < 0.08*this.level){ doJump = true; m.jumpCooldown = 30; }
    }

    // ── APPLIQUER ─────────────────────────────────────────────
    if(moveX < 0){ me.vx -= MSPD*0.4; me.face=-1; }
    if(moveX > 0){ me.vx += MSPD*0.4; me.face=1; }
    if(doJump && me.gnd) me.vy = JUMP;
    if(doShoot) me.shoot();
    me.face = dx > 0 ? 1 : -1;
  }

  // Un mur coupe-t-il la ligne de tir vers l'ennemi (écart dx,dy déjà rebouclé) ?
  _shotBlocked(me, dx, dy){
    const x0=me.x+me.w/2, y0=me.y+me.h/2;
    return me.arena.lineBlocked(x0, y0, x0+dx, y0+dy);
  }

  get label(){ return ['ROOKIE','FIGHTER','VETERAN','MASTER'][this.level-1]; }
  get color(){ return ['#888888','#60c8ff','#f0c040','#ff4444'][this.level-1]; }
}

// ELO fictif des agents scriptés par niveau
const AGENT_ELO = [1100, 1350, 1600, 1900];

// ── DONNÉES DES ENTRAÎNEURS ──────────────────────────────────────
const TRAINER_DATA = [
  { name:'KURO',  title:'le Fantôme',      color:'#aaaaaa',
    quotes:{ start:  ["Euh... on commence ?","J'essaie de mon mieux...","Tu vas sûrement gagner."],
             win:    ["... j'ai gagné ?","C'était peut-être un accident.","Hmm."],
             lose:   ["Bah voilà.","Ça m'arrive tout le temps.","Je l'avais senti."],
             streak: ["Arrête. Tu me humilies.","D'accord, t'as compris le truc."] }},
  { name:'SERA',  title:"l'Arbalétrier",   color:'#60c8ff',
    quotes:{ start:  ["Concentre-toi. Chaque tir compte.","La technique avant tout.","Viser, c'est observer."],
             win:    ["La précision prime toujours.","Position dominante, victoire assurée.","Sois patient."],
             lose:   ["Bien visé.","Tu comprends la géométrie.","Intéressant choix d'angle."],
             streak: ["Tu commences à viser correctement.","Retiens cet angle."] }},
  { name:'VENN',  title:'le Chasseur',     color:'#f0c040',
    quotes:{ start:  ["Je t'ai dans le viseur. Bonne chance.","La chasse commence.","Cours. Ça ne changera rien."],
             win:    ["On ne fuit pas le chasseur.","Trop prévisible.","Je t'avais cerné dès le début."],
             lose:   ["Tu commences à chasser toi aussi.","Plus intéressant.","Méfie-toi. J'adapte."],
             streak: ["Impressionnant. Vraiment.","Le chasseur devient la proie ?"] }},
  { name:'RYUN',  title:'le Maître',       color:'#ff6060',
    quotes:{ start:  ["Montre-moi ce que ton cerveau a appris.","Tes patterns sont transparents.","Intéressant spécimen."],
             win:    ["Pas encore. Continue.","Prévisible.","Je vois chaque mouvement avant toi."],
             lose:   ["Inattendu. Bien.","Tu évolues.","Conserve cet instinct."],
             streak: ["Tu m'obliges à changer de stratégie.","Voilà une IA qui apprend vraiment."] }},
];

function trainerQuote(level, bank){
  const t = TRAINER_DATA[(level||1)-1];
  if(!t) return '';
  const arr = t.quotes[bank] || [];
  if(!arr.length) return '';
  return `[${t.name}] « ${arr[Math.floor(Math.random()*arr.length)]} »`;
}

// ── PERSONNALITÉS DES CERVEAUX ───────────────────────────────────
const PERSONALITIES = {
  ghost: {
    name:'Fantôme', color:'#a0a0c0', icon:'👻',
    rewardBias:{ arrowDodge:2.5, stationary:-3.0, aimPrec:0.5, wall:2.5 },
    quotes:{
      roundStart:['...','Tu ne me verras pas venir.','Je suis déjà là.'],
      win:       ['(silence)','Insaisissable.','Tu n\'étais pas prêt.'],
      lose:      ['Je reviendrai.','...','Hmph.'],
      rage:      ['Tu as été chanceux.','Cette fois, tu ne m\'échapperas pas.','Je ne rate pas deux fois.'],
      taunt:     ['Je n\'étais pas concentré.','Essaie encore.'],
    }
  },
  berserk: {
    name:'Berserker', color:'#ff4040', icon:'🔥',
    rewardBias:{ aggro:2.5, arrowNear:2.0, survival:0.4, arrowDodge:0.5 },
    quotes:{
      roundStart:['VIENS !','Je vais tout casser.','MONTRE-MOI TOUT !'],
      win:       ['ENCORE !','Trop facile. Prochain !','C\'est ça ta meilleure attaque ?!'],
      lose:      ['JE VAIS TE DÉCHIRER !','RAGE !','Ça ne se passera pas comme ça !'],
      rage:      ['BERSERK MODE ACTIVÉ !','PLUS RIEN NE M\'ARRÊTE !!!','JE SENS LA CHALEUR !!!'],
      taunt:     ['C\'est tout ce que t\'as ?','ALLEZ, PLUS FORT !'],
    }
  },
  sniper: {
    name:'Tireur d\'Élite', color:'#60c8ff', icon:'🎯',
    rewardBias:{ aimPrec:3.0, shootWell:2.5, arrowNear:-0.5, mobility:0.5 },
    quotes:{
      roundStart:['Cible acquise.','Un seul tir suffit.','Je t\'ai déjà dans le viseur.'],
      win:       ['Prévisible.','Comme calculé.','Distance optimale maintenue.'],
      lose:      ['Je t\'analyserai mieux.','Intéressant schéma.','J\'ajuste mes calculs.'],
      rage:      ['Ne bouge plus.','J\'ai ta trajectoire.','Aucune esquive ne t\'aidera.'],
      taunt:     ['Tu croyais m\'échapper ?','J\'attendais ça depuis le début.'],
    }
  },
  taunter: {
    name:'Provocateur', color:'#f0c040', icon:'😏',
    rewardBias:{ aggro:1.5, mobility:1.5 },
    quotes:{
      roundStart:['Essaie de me toucher.','Amuse-moi.','J\'espère que tu es bon...'],
      win:       ['T\'appelles ça jouer ?','Même pas fatiguant.','Trop facile 😏'],
      lose:      ['Ok ok, je suis sérieux maintenant.','Bon, t\'as eu de la chance.','Tu viens de commettre une erreur.'],
      rage:      ['Tu viens de commettre UNE ERREUR.','Je vais te montrer ce que je vaux vraiment.','Fini de jouer.'],
      taunt:     ['Trop lent 😏','Raté encore ?','Tu veux des conseils ?'],
    }
  },
  tactician: {
    name:'Calculateur', color:'#40c080', icon:'🧮',
    rewardBias:{ mobility:2.0, wall:2.0, aimPrec:1.5 },
    platformBonus:true,
    quotes:{
      roundStart:['Variables analysées.','Position optimale calculée.','Phase d\'acquisition de données.'],
      win:       ['Résultat conforme aux projections.','Efficacité : 94%.','Séquence exécutée.'],
      lose:      ['Données insuffisantes. Recalcul.','Erreur de variable. Correction.','Hypothèse incorrecte.'],
      rage:      ['Recalibrage d\'urgence.','Adaptation en cours.','Nouvelle variable détectée.'],
      taunt:     ['Ça n\'aurait pas dû fonctionner.','Ta stratégie est transparente.'],
    }
  },
  loner: {
    name:'Loup Solitaire', color:'#c060ff', icon:'🐺',
    rewardBias:{ aggro:1.5, survival:1.5, arrowDodge:1.5 },
    quotes:{
      roundStart:['Je n\'ai besoin de personne.','...','On verra bien.'],
      win:       ['C\'est dans l\'ordre des choses.','Je m\'en doutais.','Seul comme toujours.'],
      lose:      ['La prochaine fois.','Ce n\'est pas fini.','Je n\'oublie pas.'],
      rage:      ['Tu m\'as forcé.','Je n\'oublie pas.','Un loup acculé est dangereux.'],
      taunt:     ['Satisfait ?','Tu n\'as rien compris à mon style.'],
    }
  },
};
const PERSONALITY_KEYS = Object.keys(PERSONALITIES);

// ================================================================
//  PIÈGES (TRAPS)
class Trap {
  constructor(type, x, y, config={}){
    this.type = type; this.x=x; this.y=y;
    this.active = true; this.timer = 0;
    this.cfg = config;
    // État propre selon type
    switch(type){
      case 'mine':     this.triggered=false; this.triggerTimer=0; break;
      case 'launcher': this.cooldown=0; this.buttonX=config.btnX||x+50; this.buttonY=config.btnY||y-20; this.pressed=false; break;
      case 'electric': this.phase=Math.random()*Math.PI*2; break;
      case 'teleport': this.partnerX=config.partnerX||x+400; this.partnerY=config.partnerY||y; this.cooldown=0; break;
      case 'wind':     this.dir=config.dir||1; this.strength=config.strength||3; break;
    }
  }

  update(entities, projectiles){
    this.timer++;
    switch(this.type){
      case 'mine': this._updateMine(entities, projectiles); break;
      case 'launcher': this._updateLauncher(entities, projectiles); break;
      case 'electric': this._updateElectric(entities); break;
      case 'teleport': this._updateTeleport(entities); break;
      case 'wind': this._updateWind(entities, projectiles); break;
    }
  }

  _updateMine(entities, proj){
    if(this.triggered){ this.triggerTimer++; if(this.triggerTimer>8){this.active=false;} return; }
    for(const e of entities){
      const dx=e.x+e.w/2-this.x, dy=e.y+e.h-this.y;
      if(Math.abs(dx)<16&&Math.abs(dy)<14){
        this.triggered=true;
        // Explosion : salve de 8 flèches
        for(let a=0;a<8;a++){
          const ang=a*Math.PI/4;
          proj.push({x:this.x,y:this.y,vx:Math.cos(ang)*ASPD*0.7,vy:Math.sin(ang)*ASPD*0.7,
            life:120,st:false,tr:[],owner:'trap',col:'#ff8020'});
        }
        boom(this.x,this.y,'#ff8020');
      }
    }
  }

  _updateLauncher(entities, proj){
    this.cooldown=Math.max(0,this.cooldown-1);
    // Dalle : quelqu'un se tient dessus ? (un autre joueur ailleurs ne la relâche pas)
    const onPlate=entities.some(e=>Math.abs(wrapDX(e.x+e.w/2-this.buttonX))<18 && Math.abs(e.y+e.h-this.buttonY)<12 && e.gnd);
    if(onPlate && !this.pressed && this.cooldown===0){
      this.cooldown=120;
      // Salve en éventail depuis le mur
      const dir=this.cfg.dir||1;
      for(let i=0;i<3;i++){
        proj.push({x:this.x+dir*4,y:this.y,vx:dir*ASPD,vy:(i-1)*1.6,life:200,st:false,tr:[],owner:'trap',col:'#ff4444'});
      }
    }
    this.pressed=onPlate;
  }

  _updateElectric(entities){
    const active = Math.sin(this.timer*0.05+this.phase) > 0.3;
    if(!active) return;
    for(const e of entities){
      if(e.x+e.w>this.x && e.x<this.x+80 && e.y+e.h>this.y && e.y<this.y+8){
        // Choc : repousse vers le haut
        e.vy = Math.min(e.vy, JUMP*0.8);
        e.vx += (Math.random()-0.5)*6;
      }
    }
  }

  // Deux portails reliés dans les deux sens
  _updateTeleport(entities){
    if(this.cooldown>0){this.cooldown--;return;}
    const ends=[[this.x,this.y,this.partnerX,this.partnerY],[this.partnerX,this.partnerY,this.x,this.y]];
    for(const e of entities){
      for(const [ax,ay,bx,by] of ends){
        if(Math.abs(wrapDX(e.x+e.w/2-ax))<16 && Math.abs(wrapDY(e.y+e.h/2-ay))<20){
          e.x=bx-e.w/2; e.y=by+16-e.h;
          e.vx*=0.3; e.vy*=0.3;
          this.cooldown=60;
          boom(ax,ay,'#a060ff'); boom(bx,by,'#a060ff');
          return;
        }
      }
    }
  }

  _updateWind(entities, proj){
    for(const e of entities){
      if(e.x+e.w>this.x&&e.x<this.x+this.cfg.w&&e.y+e.h>this.y&&e.y<this.y+(this.cfg.h||WH)){
        e.vx += this.dir*this.strength*0.04;
      }
    }
    for(const a of proj){
      if(!a.st&&a.x>this.x&&a.x<this.x+this.cfg.w){
        a.vx += this.dir*this.strength*0.08;
      }
    }
  }

  draw(ctx){
    if(!this.active) return;
    switch(this.type){
      case 'mine':     this._drawMine(ctx); break;
      case 'launcher': this._drawLauncher(ctx); break;
      case 'electric': this._drawElectric(ctx); break;
      case 'teleport': this._drawTeleport(ctx); break;
      case 'wind':     this._drawWind(ctx); break;
    }
  }

  _drawMine(ctx){
    if(this.triggered){
      ctx.fillStyle=`rgba(255,128,32,${1-this.triggerTimer/8})`;
      ctx.beginPath();ctx.arc(this.x,this.y,30+this.triggerTimer*4,0,Math.PI*2);ctx.fill();
      return;
    }
    ctx.fillStyle='#cc4400';ctx.fillRect(this.x-8,this.y-8,16,16);
    ctx.fillStyle='#ff6600';ctx.fillRect(this.x-4,this.y-12,8,6);
    ctx.fillStyle='#ffaa00';ctx.fillRect(this.x-2,this.y-6,4,4);
    // Indicateur danger
    if(Math.floor(this.timer/15)%2===0){
      ctx.fillStyle='rgba(255,80,0,0.4)';ctx.beginPath();ctx.arc(this.x,this.y,24,0,Math.PI*2);ctx.fill();
    }
  }

  _drawLauncher(ctx){
    // Tourelle encastrée dans le mur
    const d=this.cfg.dir||1, ready=this.cooldown===0;
    ctx.fillStyle='#2a1408'; ctx.fillRect(this.x-(d>0?4:8),this.y-8,12,16);
    ctx.fillStyle=ready?'#ff5a10':'#6a2a10'; ctx.fillRect(this.x+(d>0?4:-8),this.y-3,4,6);
    ctx.fillStyle='#120804'; ctx.fillRect(this.x+(d>0?7:-8),this.y-1,1,2);
    // Dalle de déclenchement
    ctx.fillStyle=this.pressed?'#ff2200':'#7a2a0c';
    ctx.fillRect(this.buttonX-8,this.buttonY-3,16,3);
    ctx.fillStyle='#c84a14'; ctx.fillRect(this.buttonX-6,this.buttonY-4,12,1);
  }

  _drawElectric(ctx){
    const active=Math.sin(this.timer*0.05+this.phase)>0.3;
    ctx.fillStyle=active?'#ffff44':'#444420';
    ctx.fillRect(this.x,this.y,80,8);
    if(active){
      ctx.strokeStyle='rgba(255,255,0,0.6)';ctx.lineWidth=1;
      for(let i=0;i<6;i++){
        ctx.beginPath();ctx.moveTo(this.x+i*14,this.y);
        ctx.lineTo(this.x+i*14+(Math.random()-0.5)*8,this.y-8-Math.random()*8);
        ctx.stroke();
      }
    }
  }

  _drawTeleport(ctx){
    const pulse=Math.sin(this.timer*0.08)*0.3+0.7;
    // Portail A
    ctx.fillStyle=`rgba(160,96,255,${pulse*0.3})`;ctx.beginPath();ctx.arc(this.x,this.y,20,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=`rgba(160,96,255,${pulse})`;ctx.lineWidth=2;ctx.beginPath();ctx.arc(this.x,this.y,20,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='rgba(160,96,255,0.8)';ctx.font='6px monospace';ctx.textAlign='center';ctx.fillText('A',this.x,this.y+3);
    // Portail B
    ctx.fillStyle=`rgba(96,255,160,${pulse*0.3})`;ctx.beginPath();ctx.arc(this.partnerX,this.partnerY,20,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=`rgba(96,255,160,${pulse})`;ctx.lineWidth=2;ctx.beginPath();ctx.arc(this.partnerX,this.partnerY,20,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='rgba(96,255,160,0.8)';ctx.fillText('B',this.partnerX,this.partnerY+3);
  }

  _drawWind(ctx){
    const w=this.cfg.w||100, h=this.cfg.h||WH;
    const alpha=0.06+Math.sin(this.timer*0.04)*0.03;
    ctx.fillStyle=`rgba(200,220,255,${alpha})`;
    ctx.fillRect(this.x,this.y,w,h);
    // Filets d'air qui défilent dans le sens du courant
    ctx.fillStyle='rgba(170,210,255,0.28)';
    for(let i=0;i<12;i++){
      const off=((this.timer*this.dir*2+i*37)%w+w)%w;
      ctx.fillRect(Math.round(this.x+off), Math.round(this.y+(i*41)%h), 14, 1);
    }
  }
}

// ================================================================
//  DEEP LEARNING — Policy Gradient (REINFORCE + Adam) sur l'arbitre
//  Entraîne arb1..arb3 par rétropropagation ; les experts GRU
//  évoluent par mutation/sélection.
// ================================================================
class PolicyGradientLearner {
  constructor(lr=0.001, gamma=0.97){
    this.lr      = lr;
    this.gamma   = gamma;
    this.buffer  = [];
    this.baseline= 0;
    this.adamT   = 0;
    this.adamS   = {arb1:{}, arb2:{}, arb3:{}};
    this.lossHist= [];
    this.gradNorm= 0;
    this.roundsLearned = 0;
    this.entropyBeta = 0.01; // pousse à garder un peu d'exploration
  }

  // actions : actions réellement jouées (échantillonnées à l'entraînement) ;
  // par défaut l'action déterministe p>0.5 (jeu PvP).
  step(logits, reward, brain, actions=null){
    if(this.buffer.length >= 2500) return;
    if(!brain.arb1?._lastX) return;
    this.buffer.push({
      logits : logits.slice(),
      actions: actions ? actions.slice() : logits.map(p => p > 0.5 ? 1 : 0),
      reward,
      a1x: brain.arb1._lastX.slice(), a1z: brain.arb1._lastZ.slice(),
      a2x: brain.arb2._lastX.slice(), a2z: brain.arb2._lastZ.slice(),
      a3x: brain.arb3._lastX.slice(), a3z: brain.arb3._lastZ.slice(),
    });
  }

  _computeAdvantage(){
    const n = this.buffer.length;
    const G = new Array(n);
    let run = 0;
    for(let t = n-1; t >= 0; t--){
      run = this.buffer[t].reward + this.gamma * run;
      G[t] = run;
    }
    return this._advantageFromReturns(G);
  }

  // Avantage = (retour − baseline glissante) / écart-type glissant.
  // Ne PAS normaliser par la moyenne de l'épisode : sinon une victoire et une
  // défaite donnent des avantages symétriques et le bonus ±3 de fin de round
  // ne dit plus rien sur le résultat.
  _advantageFromReturns(G){
    const n = G.length;
    const mean = G.reduce((a,b)=>a+b,0)/n;
    const b0 = this.baseline;
    const varNow = G.reduce((a,g)=>a+(g-b0)**2,0)/n;
    this.retVar   = 0.95*(this.retVar ?? varNow) + 0.05*varNow;
    this.baseline = 0.95*b0 + 0.05*mean;
    const std = Math.sqrt(this.retVar) + 1e-6;
    return G.map(g => Math.max(-5, Math.min(5, (g - b0)/std)));
  }

  // Refuse une mise à jour non finie (NaN/Infinity) : elle corromprait le cerveau pour de bon
  _finiteGrads(...mats){
    for(const M of mats) for(const row of M) for(const v of row) if(!Number.isFinite(v)) return false;
    return true;
  }

  _adam(state, W, dW, b, db){
    const b1=0.9, b2=0.999, eps=1e-8;
    const t = this.adamT;
    if(!state.mW){
      state.mW = W.map(r => r.map(()=>0));
      state.vW = W.map(r => r.map(()=>0));
      state.mb = b.map(()=>0);
      state.vb = b.map(()=>0);
    }
    const lrT = this.lr * Math.sqrt(1 - b2**t) / (1 - b1**t);
    const clip = v => Math.max(-2, Math.min(2, v));
    for(let i=0; i<W.length; i++){
      for(let j=0; j<W[i].length; j++){
        const g = clip(dW[i][j]);
        state.mW[i][j] = b1*state.mW[i][j] + (1-b1)*g;
        state.vW[i][j] = b2*state.vW[i][j] + (1-b2)*g*g;
        W[i][j] -= lrT * state.mW[i][j] / (Math.sqrt(state.vW[i][j]) + eps);
      }
    }
    for(let i=0; i<b.length; i++){
      const g = clip(db[i]);
      state.mb[i] = b1*state.mb[i] + (1-b1)*g;
      state.vb[i] = b2*state.vb[i] + (1-b2)*g*g;
      b[i] -= lrT * state.mb[i] / (Math.sqrt(state.vb[i]) + eps);
    }
  }

  // Gradients moyens de -A·log π(a) (+ bonus d'entropie) sur arb3 → arb2 → arb1.
  // Tampons typés accumulés en place : aucune matrice allouée par pas de temps.
  _gradients(brain, batch, adv){
    const layers=[brain.arb1, brain.arb2, brain.arb3];
    const gW=layers.map(L=>L.W.map(r=>new Float64Array(r.length)));
    const gb=layers.map(L=>new Float64Array(L.b.length));
    const dz=layers.map(L=>new Float64Array(L.b.length));
    const dx=layers.map(L=>new Float64Array(L.W[0].length));
    const ENT=this.entropyBeta;
    let loss=0;
    for(let t=0;t<batch.length;t++){
      const e=batch[t], A=adv[t], d3=dz[2];
      for(let i=0;i<d3.length;i++){
        const p=Math.min(1-1e-6,Math.max(1e-6,e.logits[i])), a=e.actions[i];
        // sortie sigmoïde : d(-A·log π)/dz = -A·(a-p) ; entropie : dH/dz = p(1-p)·ln((1-p)/p)
        d3[i] = -A*(a-p) - (ENT ? ENT*p*(1-p)*Math.log((1-p)/p) : 0);
        loss -= (a*Math.log(p)+(1-a)*Math.log(1-p))*Math.abs(A);
      }
      PolicyGradientLearner._back(layers[2].W, e.a3x, d3, gW[2], gb[2], dx[2]);
      PolicyGradientLearner._actGrad(layers[1].act, e.a2z, dx[2], dz[1]);
      PolicyGradientLearner._back(layers[1].W, e.a2x, dz[1], gW[1], gb[1], dx[1]);
      PolicyGradientLearner._actGrad(layers[0].act, e.a1z, dx[1], dz[0]);
      PolicyGradientLearner._back(layers[0].W, e.a1x, dz[0], gW[0], gb[0], dx[0]);
    }
    const inv=1/batch.length;
    for(let k=0;k<3;k++){ for(const r of gW[k]) for(let j=0;j<r.length;j++) r[j]*=inv; for(let i=0;i<gb[k].length;i++) gb[k][i]*=inv; }
    return {gW, gb, loss:loss*inv};
  }

  // Couche dense : accumule dW (+= d·xᵀ), db (+= d) et calcule DX = Wᵀ·d
  static _back(W, x, d, GW, GB, DX){
    DX.fill(0);
    for(let i=0;i<W.length;i++){
      const di=d[i];
      GB[i]+=di;
      const row=W[i], gr=GW[i];
      for(let j=0;j<row.length;j++){ gr[j]+=di*x[j]; DX[j]+=row[j]*di; }
    }
  }
  // Gradient à travers l'activation : dz = dy · f'(z)
  static _actGrad(act, z, dy, dz){
    if(act==='relu'){ for(let i=0;i<dz.length;i++) dz[i]=dy[i]*(z[i]>0?1:0.01); }
    else if(act==='sigmoid'){ for(let i=0;i<dz.length;i++){ const s=1/(1+Math.exp(-Math.max(-10,Math.min(10,z[i])))); dz[i]=dy[i]*s*(1-s); } }
    else { for(let i=0;i<dz.length;i++) dz[i]=dy[i]; }
  }

  // Applique Adam si les gradients sont finis ; renvoie false sinon (mise à jour refusée)
  _apply(brain, g, lrScale=1){
    if(!this._finiteGrads(...g.gW, g.gb)) return false;
    this.adamT++;
    const savedLr=this.lr; this.lr=savedLr*lrScale;
    this._adam(this.adamS.arb3, brain.arb3.W, g.gW[2], brain.arb3.b, g.gb[2]);
    this._adam(this.adamS.arb2, brain.arb2.W, g.gW[1], brain.arb2.b, g.gb[1]);
    this._adam(this.adamS.arb1, brain.arb1.W, g.gW[0], brain.arb1.b, g.gb[0]);
    this.lr=savedLr;
    let sq=0; for(const r of g.gW[2]) for(const v of r) sq+=v*v;
    this.gradNorm=Math.sqrt(sq+1e-8);
    this.lossHist.push(g.loss);
    if(this.lossHist.length > 30) this.lossHist.shift();
    return true;
  }

  // Fin de round : tout le tampon, avec le bonus/malus de victoire sur le dernier pas
  learn(brain, terminalReward=0){
    const n = this.buffer.length;
    if(n < 15){ this.buffer = []; return null; }
    if(terminalReward !== 0) this.buffer[n-1].reward += terminalReward;
    const adv = this._computeAdvantage();
    const g = this._gradients(brain, this.buffer, adv);
    this.buffer = [];
    if(!this._apply(brain, g)) return null;
    this.roundsLearned++;
    return {loss: g.loss, gradNorm: this.gradNorm};
  }

  // Mise à jour en ligne toutes les ~60 frames (pendant le jeu)
  // nStep : taille du mini-batch, lrScale : fraction du lr de base
  learnOnline(brain, nStep=60, lrScale=0.3){
    const n = Math.min(nStep, this.buffer.length);
    if(n < 15) return null;
    const batch = this.buffer.splice(0, n);
    // Retours tronqués n-step, bootstrappés sur la baseline courante
    const G = new Array(n);
    let run = this.baseline;
    for(let t = n-1; t >= 0; t--){ run = batch[t].reward + this.gamma * run; G[t] = run; }
    const adv = this._advantageFromReturns(G);
    const g = this._gradients(brain, batch, adv);
    if(!this._apply(brain, g, lrScale)) return null;
    this.onlineLearned = (this.onlineLearned||0) + 1;
    return {loss: g.loss, n};
  }

  serializeAdam(){
    return {
      t        : this.adamT,
      s        : JSON.parse(JSON.stringify(this.adamS)),
      baseline : this.baseline,
      retVar   : this.retVar,
      rounds   : this.roundsLearned,
      lossHist : this.lossHist.slice(),
      online   : this.onlineLearned||0,
    };
  }

  loadAdam(d){
    if(!d) return;
    this.adamT         = d.t        || 0;
    this.adamS         = d.s ? JSON.parse(JSON.stringify(d.s)) : {arb1:{}, arb2:{}, arb3:{}};
    this.baseline      = d.baseline || 0;
    this.retVar        = d.retVar;
    this.roundsLearned = d.rounds   || 0;
    this.lossHist      = d.lossHist || [];
    this.onlineLearned = d.online   || 0;
  }

  uiSummary(){
    const last10 = this.lossHist.slice(-10);
    const avgL = last10.length ? (last10.reduce((a,b)=>a+b,0)/last10.length).toFixed(3) : '—';
    return `DL∇ rounds:${this.roundsLearned} live:${this.onlineLearned||0} loss:${avgL} |∇|:${this.gradNorm.toFixed(4)}`;
  }
}

// Rend la main au navigateur puis continue (MessageChannel : sans le délai minimal
// de 4 ms que setTimeout impose aux appels imbriqués)
const _yieldChan = new MessageChannel(); const _yieldQ = [];
_yieldChan.port1.onmessage = () => { const f=_yieldQ.shift(); if(f) f(); };
function yieldThen(f){ _yieldQ.push(f); _yieldChan.port2.postMessage(0); }

// ================================================================
//  SESSION D'ENTRAÎNEMENT (sans interface)
//  Tourne dans le worker (train-worker.js) ou, à défaut, dans la page.
//  Communique uniquement par événements : emit({type:'toast'|'round'|'gen', …}).
// ================================================================
const TR_GEN_WIN_SIZE = 20;  // fenêtre glissante pour le taux de victoire
// Paliers de génération : taux de victoire OU ELO à atteindre selon la génération actuelle
function computeGenTarget(currentGen){
  const tier = currentGen<=5 ? 0 : currentGen<=15 ? 1 : currentGen<=30 ? 2 : 3;
  return {
    winRate : [0.30, 0.40, 0.50, 0.60][tier],
    elo     : [1300, 1450, 1600, 1800][tier],
  };
}
function _curriculumLevel(elo){
  if(elo<1300) return 1;
  if(elo<1500) return 2;
  if(elo<1800) return 3;
  return 4;
}

// ── Sélection évolutive appariée ──
// Après chaque round, des candidats (experts GRU mutés) et le champion jouent en coulisse
// les MÊMES rounds (même carte, mêmes apparitions, même hasard) : un candidat ne remplace
// le champion que s'il fait nettement mieux à conditions égales.
// L'arbitre, lui, apprend par gradient de politique pendant les rounds joués.
// Réglages retenus après comparaison de variantes : mutation des seuls experts + gradient =
// progression la plus régulière ; 16 rounds et une marge de 0.1 évitent d'accepter des
// candidats chanceux (≈25 % acceptés au lieu de 50 %).
const ES_ROUNDS_SOLO = 16, ES_ROUNDS_POP = 8;
const ES_MARGIN = 0.1;

// Round d'évaluation sans rendu : cerveau contre entraîneur scripté.
// Score : défaite -1 (jusqu'à -0.5 si tardive), temps écoulé 0, victoire +1 (jusqu'à 1.5 si rapide).
function simEvalRound(brain, level, def){
  const A=new Arena(def);
  const sp=A.randomSpawns(2);
  const me=new Ent(sp[0].x,sp[0].y,'#60c8ff',true,A), op=new Ent(sp[1].x,sp[1].y,'#ff4444',true,A);
  const ag=new ScriptedAgent(level);
  brain.resetMemory();
  const T=60*20;
  for(let t=1;t<=T;t++){
    aiProcessFor(brain.forward(aiInputsFor(me,op,op.arr,t)), me, op);
    me.face=wrapDX(op.x-me.x)<0?-1:1;
    me.update();
    ag.act(op,me,me.arr,[],t); op.update();
    for(const a of op.arr) if(!a.st && arrowHitsEnt(a,me)) return -1 + 0.5*t/T;
    for(const a of me.arr) if(!a.st && arrowHitsEnt(a,op)) return 1 + 0.5*(1-t/T);
  }
  return 0;
}

// Fait jouer à chaque cerveau les mêmes rounds (Math.random graine pendant chaque round).
// Rend la main toutes les ~12 ms ; renvoie null si isAlive() devient faux.
async function evalPaired(brains, level, rounds, mapId, isAlive){
  const scores=new Array(brains.length).fill(0);
  const seed=(Math.random()*2147483647)|0;
  let t0=performance.now();
  for(let r=0;r<rounds;r++){
    const def = mapId==='random' ? ARENA_DEFS[(seed+r)%ARENA_DEFS.length] : arenaById(mapId);
    for(let b=0;b<brains.length;b++){
      const saved=Math.random;
      Math.random=mulberry32(seed+r*7919);
      try{ scores[b]+=simEvalRound(brains[b], level, def); } finally { Math.random=saved; }
      if(performance.now()-t0>12){
        await new Promise(res=>yieldThen(res));
        if(!isAlive()) return null;
        t0=performance.now();
      }
    }
  }
  return scores.map(v=>v/rounds);
}

class TrainingSession {
  // cfg : {id, name, brain, adamState, elo, fitness, skills, generation, personality,
  //        agents:[{level}], traps, rounds, curriculum, objective, map, popSize, evoParams}
  constructor(cfg, emit){
    this.emit=emit;
    this.cfg=cfg;
    this.stopped=false; this.finished=false;
    if(cfg.evoParams) evoParams={...evoParams, ...cfg.evoParams};
    const b=deserializeBrain(cfg.brain);
    b.fitness=cfg.fitness||0; b.elo=cfg.elo||ELO_BASE;
    b.skills={...(cfg.skills||{})}; b.generation=cfg.generation||1;
    b.pgLearner=new PolicyGradientLearner(0.001, 0.97);
    if(cfg.adamState) b.pgLearner.loadAdam(cfg.adamState);
    this.brain=b;
    this.name=cfg.name||'Cerveau';
    this.personality=PERSONALITIES[cfg.personality] ? cfg.personality : PERSONALITY_KEYS[Math.floor(Math.random()*PERSONALITY_KEYS.length)];
    this.objective=cfg.objective||'balanced';
    this.emotional={rage:0, confidence:0, frustration:0};
    this.consecWins=0; this.consecLosses=0; this.rageShown=false;
    this.hof=[]; this.stag=0;
    this.roundsDone=0;
    this.unsaved={wins:0, played:0};
    this.evoHistoryTail=[]; this.genWin=[]; this.genHistory=[];
    this.curLevel=_curriculumLevel(b.elo); this.curWins=0; this.curTotal=0;
    this.trainerLevel=cfg.curriculum ? this.curLevel : (cfg.agents[0]?.level||1);
    this.agents=cfg.agents.map(a=>new ScriptedAgent(cfg.curriculum ? this.trainerLevel : a.level));
    this.popSize=Math.max(1,cfg.popSize||1);
    this.esGen=0; this.esAccepted=0; this.evolving=false;
    this.arena=null; this.round=null;
    this.emit({type:'toast', msg:trainerQuote(this.trainerLevel,'start')});
  }

  // Message de personnalité du cerveau (victoire, défaite, rage…)
  _personalityToast(event){
    const p=PERSONALITIES[this.personality]; const bank=p?.quotes[event];
    if(!bank?.length) return;
    this.emit({type:'toast', msg:`${p.icon} ${this.name} : "${bank[Math.floor(Math.random()*bank.length)]}"`});
  }

  // Poids de récompense = objectif × biais de personnalité × état émotionnel
  _weights(){
    const key=this.personality, p=PERSONALITIES[key];
    const base={...(OBJECTIVE_WEIGHTS[this.objective]||OBJECTIVE_WEIGHTS.balanced)};
    const bias=p?.rewardBias??{};
    for(const k of Object.keys(bias)){ if(base[k]!==undefined) base[k]*=bias[k]; else base[k]=bias[k]; }
    if(key==='berserk' && this.emotional.rage>60){ base.aggro=(base.aggro||1)*1.5; base.wall=(base.wall||1)*0.5; }
    if(key==='sniper' && this.emotional.confidence>60){ base.aimPrec=(base.aimPrec||1)*1.3; }
    return base;
  }

  // ── Début de round : carte, curriculum, entités, adversaire HoF éventuel ──
  _startRound(){
    const cfg=this.cfg;
    if(!this.arena || cfg.map==='random') this.arena=makeArena(cfg.map, cfg.traps);
    else this.arena.reset();
    if(cfg.curriculum && this.curTotal>=20){
      const wr=this.curWins/this.curTotal;
      if(wr>=0.6 && this.curLevel<4){
        this.curLevel++; this.curWins=0; this.curTotal=0; this.trainerLevel=this.curLevel;
        const td=TRAINER_DATA[this.trainerLevel-1];
        this.emit({type:'toast', msg:`⚔ ${td.name} ${td.title} : « ${td.quotes.start[0]} »`});
      } else if(wr<0.25 && this.curLevel>1){
        this.curLevel--; this.curWins=0; this.curTotal=0; this.trainerLevel=this.curLevel;
        const td=TRAINER_DATA[this.trainerLevel-1];
        this.emit({type:'toast', msg:`↓ ${td.name} ${td.title} reprend l'entraînement.`});
      }
      this.agents.forEach(a=>a.level=this.curLevel);
    }
    const A=this.arena;
    const sp=A.randomSpawns(1+this.agents.length);
    const ents=[];
    const me=new Ent(sp[0].x,sp[0].y,SLOT_COLORS[0],true,A); me.face=sp[0].face; ents.push(me);
    this.agents.forEach((ag,i)=>{
      const col = i===0 ? (TRAINER_DATA[this.trainerLevel-1]?.color||SLOT_COLORS[1]) : SLOT_COLORS[i+1];
      const e=new Ent(sp[i+1].x,sp[i+1].y,col,true,A); e.face=sp[i+1].face; ents.push(e);
      ag.reset();
    });
    this.brain.resetMemory();
    // Self-play : 25 % des rounds contre un cerveau HoF passé (ELO > 1350), à la place du 1er entraîneur
    const roundAgents=this.agents.slice();
    let hofElo=null;
    if(this.hof.length>0 && (this.brain.elo||ELO_BASE)>1350 && Math.random()<0.25){
      const hofBrain=this.hof[Math.floor(Math.random()*this.hof.length)];
      hofBrain.resetMemory();
      hofElo=hofBrain.elo||ELO_BASE;
      const R=()=>this.round;
      roundAgents[0]={ act(ent,ce,arrows){ aiProcessFor(hofBrain.forward(aiInputsFor(ent,ce,arrows,R().timer)), ent, ce); } };
    }
    this.round={
      ents, roundAgents, hofElo, timer:0, over:false, outcome:null, shaped:0,
      traps:A.buildTraps(cfg.traps), trapProj:[],
      weights:this._weights(),
      rs:{ arrowsDodged:0, shotsNearMiss:0, platformChanges:0, lastPlatY:-1, framesAiming:0,
           stationaryFrames:0, lastX:me.x, shotsFired:0, shootOpps:0, preciseShots:0,
           prevAimDiff:null, aimHeld:0 },
    };
    if(this.roundsDone===0) this._personalityToast('roundStart');
  }

  // ── Une image de simulation ──
  _step(){
    const R=this.round, A=this.arena, brain=this.brain, rs=R.rs, W=R.weights;
    R.timer++;
    const ce=R.ents[0];
    const d2=e=>wrapDX(e.x-ce.x)**2+wrapDY(e.y-ce.y)**2;
    let foe=R.ents[1];
    for(let k=2;k<R.ents.length;k++) if(d2(R.ents[k])<d2(foe)) foe=R.ents[k];
    const enemyArrows=[]; for(let k=1;k<R.ents.length;k++) for(const a of R.ents[k].arr) enemyArrows.push(a);
    const inp=aiInputsFor(ce,foe,enemyArrows,R.timer);
    const out=brain.forward(inp);
    // Exploration : actions tirées selon les probabilités du réseau (politique stochastique)
    const act=aiProcessFor(out, ce, foe);
    const edx=wrapDX(foe.x+foe.w/2-(ce.x+ce.w/2)), edy=wrapDY(foe.y+foe.h/2-(ce.y+ce.h/2));
    ce.face=edx<0?-1:1;
    ce.update();

    // ── Récompenses façonnées ──
    let aimDiff=Math.abs(ce.aim-Math.atan2(edy,edx));
    if(aimDiff>Math.PI) aimDiff=Math.PI*2-aimDiff;
    const aimPrecision=Math.max(0,(0.4-aimDiff)/0.4);
    R.shaped+=aimPrecision*0.012*W.aimPrec;
    if(aimDiff<0.2) rs.framesAiming++;
    if(brain.pgLearner){
      const aimDelta = rs.prevAimDiff!==null ? rs.prevAimDiff-aimDiff : 0;
      rs.prevAimDiff=aimDiff;
      rs.aimHeld = aimDiff<0.15 ? rs.aimHeld+1 : Math.max(0,rs.aimHeld-2);
      const pgReward =
          Math.max(0,aimDelta)*0.05*W.aimPrec            // convergence de la visée
        + (rs.aimHeld>5 ? 0.006*W.aimPrec : 0)           // visée stable maintenue
        + (act[5] && aimDiff<0.3 ? 0.05*W.shootWell : 0) // tirer bien visé
        + (act[5] && aimDiff>0.8 ? -0.04 : 0)            // tir raté
        + (act[2] && !ce.gnd ? -0.018 : 0)               // saut en l'air inutile
        + ((inp[20]<0.07&&act[0])||(inp[21]<0.07&&act[1]) ? -0.025 : 0) // marcher dans le mur
        + 0.002*W.survival;
      brain.pgLearner.step(out, pgReward, brain, act);
      if(R.timer%60===0 && brain.pgLearner.buffer.length>=30) brain.pgLearner.learnOnline(brain);
    }
    if(act[5] && aimDiff<0.3 && ce.cd===0) R.shaped+=0.08*W.shootWell;
    else if(act[5] && aimDiff>0.8) R.shaped-=0.04;
    for(const a of ce.arr){
      if(a.st) continue;
      const d=Math.hypot(wrapDX(a.x-(foe.x+foe.w/2)), wrapDY(a.y-(foe.y+foe.h/2)));
      if(d<60){ R.shaped+=0.04*W.arrowNear; rs.shotsNearMiss+=0.3; }
      else if(d<120){ R.shaped+=0.015*W.arrowNear; rs.shotsNearMiss+=0.1; }
    }
    for(const a of enemyArrows){
      if(a.st) continue;
      if(Math.hypot(wrapDX(a.x-(ce.x+ce.w/2)), wrapDY(a.y-(ce.y+ce.h/2)))<80){
        rs.arrowsDodged+=0.1*W.arrowDodge; R.shaped+=0.005*W.arrowDodge;
      }
    }
    if(ce.cd===0) rs.shootOpps++;
    if(act[5] && ce.cd===0){ rs.shotsFired++; if(aimDiff<0.3) rs.preciseShots++; }
    if(Math.abs(ce.vx)<0.5 && ce.gnd){ rs.stationaryFrames++; R.shaped+=0.003*W.stationary; }
    if(W.stationary<-1.0 && Math.abs(ce.vx)>1.5) R.shaped+=0.002;
    if(inp[20]<0.1||inp[21]<0.1) R.shaped-=0.008*W.wall;
    if(act[2] && !ce.gnd) R.shaped-=0.018;
    if((inp[20]<0.07&&act[0])||(inp[21]<0.07&&act[1])) R.shaped-=0.025;
    if(ce.gnd){
      const py=Math.round(ce.y/10)*10;
      if(py!==rs.lastPlatY && rs.lastPlatY!==-1){ rs.platformChanges++; if(PERSONALITIES[this.personality]?.platformBonus) R.shaped+=0.04; }
      rs.lastPlatY=py;
    }

    // ── Entraîneurs ──
    for(let i=0;i<R.roundAgents.length;i++){
      const e=R.ents[i+1];
      const arrows=[]; for(let j=0;j<R.ents.length;j++) if(j!==i+1) for(const a of R.ents[j].arr) arrows.push(a);
      R.roundAgents[i].act(e,ce,arrows,R.traps,R.timer);
      e.update();
    }

    // ── Pièges + blocs friables (seuls les tirs de pièges sur le cerveau comptent) ──
    for(const t of R.traps) t.update(R.ents,R.trapProj);
    A.updateCrumbles(R.ents);
    updateTrapProjectiles(R.trapProj, A, [ce], (j,p)=>{ boom(p.x,p.y,ce.col); R.over=true; R.outcome='lose'; });
    if(R.over) return;

    // ── Touches ──
    for(let i=1;i<R.ents.length && !R.over;i++){
      const arr=R.ents[i].arr;
      for(let j=arr.length-1;j>=0;j--){
        const a=arr[j]; if(a.st) continue;
        if(arrowHitsEnt(a,ce)){ boom(a.x,a.y,ce.col); arr.splice(j,1); R.over=true; R.outcome='lose'; break; }
      }
    }
    if(!R.over){
      for(let j=ce.arr.length-1;j>=0 && !R.over;j--){
        const a=ce.arr[j]; if(a.st) continue;
        for(let i=1;i<R.ents.length;i++){
          if(arrowHitsEnt(a,R.ents[i])){ boom(a.x,a.y,R.ents[i].col); ce.arr.splice(j,1); R.shaped+=0.5; R.over=true; R.outcome='win'; break; }
        }
      }
    }
    if(R.timer>60*35 && !R.over){ R.over=true; R.outcome='draw'; }
  }

  // ── Fin de round : compétences, ELO, paliers, émotions, gradient, sélection ──
  _endRound(){
    const R=this.round, cfg=this.cfg, brain=this.brain, rs=R.rs;
    const won=R.outcome==='win';
    this.roundsDone++;
    this.unsaved.played++; if(won) this.unsaved.wins++;
    this.curTotal++; if(won) this.curWins++;
    if(cfg.curriculum && this.roundsDone%40===0){
      const cycle=['balanced','aim','dodge','aggro','mobility'];
      this.objective=cycle[Math.floor(this.roundsDone/40)%cycle.length];
    }
    const prevFit=brain.fitness||0;
    brain.skills=updateSkills(brain.skills||{}, rs, R.timer);
    brain.fitness=computeGlobalFitness(brain.skills);
    if(won){
      this.hof.push(brain.clone());
      this.hof.sort((a,b)=>(b.elo||0)-(a.elo||0));
      if(this.hof.length>evoParams.hofSize) this.hof.length=evoParams.hofSize;
      this.stag=0;
    } else {
      if(brain.fitness<=prevFit) this.stag++; else this.stag=Math.max(0,this.stag-1);
    }
    const lvl=cfg.curriculum ? this.curLevel : (this.agents[0]?.level||1);
    const oppElo=R.hofElo ?? (AGENT_ELO[lvl-1]||AGENT_ELO[0]);
    const prevElo=brain.elo||ELO_BASE;
    brain.elo=Math.max(400, eloUpdate(prevElo, eloExpected(prevElo,oppElo), won?1:0, this.roundsDone));

    // Palier de génération (fenêtre de victoires ou ELO)
    this.genWin.push(won?1:0);
    if(this.genWin.length>TR_GEN_WIN_SIZE) this.genWin.shift();
    if(this.genWin.length>=TR_GEN_WIN_SIZE){
      const gen=brain.generation||1, target=computeGenTarget(gen);
      const wrw=this.genWin.reduce((a,b)=>a+b,0)/TR_GEN_WIN_SIZE;
      const eloOK=(brain.elo||ELO_BASE)>=target.elo, wrOK=wrw>=target.winRate;
      if(eloOK||wrOK){
        const reason = eloOK ? `ELO ${brain.elo}` : `${Math.round(wrw*100)}% vict. / ${TR_GEN_WIN_SIZE}`;
        brain.generation=gen+1; this.genWin=[];
        this.genHistory.push({gen:brain.generation, elo:brain.elo, fitness:Math.round(brain.fitness||0), round:this.roundsDone, reason});
        this.emit({type:'toast', msg:`🧬 GÉN. ${brain.generation} — ${reason}`, important:true});
      }
    }

    // État émotionnel + répliques
    const E=this.emotional;
    if(won){
      this.consecWins++; this.consecLosses=0;
      E.confidence=Math.min(100,E.confidence+15); E.rage=Math.max(0,E.rage-10); E.frustration=Math.max(0,E.frustration-10);
    } else {
      this.consecLosses++; this.consecWins=0;
      E.rage=Math.min(100,E.rage+12); E.confidence=Math.max(0,E.confidence-8);
      const missRate=rs.shotsFired>0 ? 1-(rs.preciseShots/rs.shotsFired) : 0.5;
      E.frustration=Math.min(100,E.frustration+missRate*18);
    }
    if(this.roundsDone%5===0) this._personalityToast(won?'win':'lose');
    if(E.rage>70 && !this.rageShown){ this.rageShown=true; this._personalityToast('rage'); }
    if(won && E.confidence>75 && this.consecWins>=3 && this.roundsDone%3===0) this._personalityToast('taunt');
    if(this.roundsDone%7===0) this.emit({type:'toast', msg:trainerQuote(this.trainerLevel, won?'lose':'win')});
    if(won && this.curWins>0 && this.curWins%3===0 && this.curTotal<=this.curWins+4) this.emit({type:'toast', msg:trainerQuote(this.trainerLevel,'streak')});

    // Gradient de fin de round (bonus/malus de victoire), puis sélection appariée en coulisse
    if(brain.pgLearner) brain.pgLearner.learn(brain, won?3.0:-3.0);
    if(!this.evolving) this._evolve();

    const pg=brain.pgLearner, last=pg?.lossHist.slice(-5)||[];
    const td=TRAINER_DATA[this.trainerLevel-1]||TRAINER_DATA[0];
    this.emit({type:'round', data:{
      roundsDone:this.roundsDone, won,
      elo:brain.elo, fitness:brain.fitness, skills:{...brain.skills}, generation:brain.generation,
      stag:this.stag, esGen:this.esGen, esAccepted:this.esAccepted,
      trainerName:td.name, trainerLevel:this.trainerLevel,
      winRate:this.curTotal>0 ? Math.round(this.curWins/this.curTotal*100) : 0,
      objective:this.objective,
      loss: pg && pg.roundsLearned>0 && last.length ? last.reduce((a,b)=>a+b,0)/last.length : null,
      genWin:this.genWin.slice(), genHistory:this.genHistory.slice(-12),
    }});
  }

  // Une génération de la sélection appariée (asynchrone, rend la main régulièrement)
  async _evolve(){
    this.evolving=true;
    try{
      const lambda=this.popSize, rounds=lambda>1?ES_ROUNDS_POP:ES_ROUNDS_SOLO;
      const champ=this.brain.clone(), cands=[champ];
      for(let k=0;k<lambda;k++){ const c=champ.clone(); mutateExperts(c, this.stag); cands.push(c); }
      const level=this.cfg.curriculum ? this.curLevel : (this.agents[0]?.level||1);
      const scores=await evalPaired(cands, level, rounds, this.cfg.map, ()=>!this.stopped);
      if(!scores) return;
      this.esGen++;
      let best=0;
      for(let k=1;k<scores.length;k++) if(scores[k]>scores[best]) best=k;
      if(best>0 && scores[best]>scores[0]+ES_MARGIN){
        this.brain.experts=cands[best].experts.map(e=>e.clone());
        this.brain.resetMemory();
        this.esAccepted++; this.stag=0;
      } else this.stag++;
    } finally {
      this.evolving=false;
    }
  }

  // Avance de n images (enchaîne les rounds) ; finished=true quand le nombre de rounds est atteint
  advance(n){
    for(let i=0;i<n && !this.finished && !this.stopped;i++){
      if(!this.round) this._startRound();
      this._step();
      if(this.round.over){
        this._endRound();
        this.round=null;
        if(this.roundsDone>=this.cfg.rounds) this.finished=true;
      }
    }
  }

  // Données à sauvegarder (les compteurs victoires/parties repartent de zéro)
  saveData(){
    const b=this.brain, d={
      brain:b.serialize(), adamState:b.pgLearner?.serializeAdam(),
      elo:b.elo, fitness:b.fitness, skills:{...(b.skills||{})}, generation:b.generation||1,
      personality:this.personality, wins:this.unsaved.wins, played:this.unsaved.played,
    };
    this.unsaved={wins:0, played:0};
    return d;
  }

  // Image légère pour l'aperçu (objets simples, sans référence à l'arène)
  snapshot(){
    const R=this.round, A=this.arena;
    if(!R||!A) return null;
    const plainEnt=e=>({x:e.x,y:e.y,vx:e.vx,vy:e.vy,w:e.w,h:e.h,col:e.col,ai:e.ai,gnd:e.gnd,aim:e.aim,face:e.face,
      cd:e.cd,ht:e.ht,hp:e.hp,hitFlash:e.hitFlash,arr:e.arr.map(a=>({x:a.x,y:a.y,vx:a.vx,vy:a.vy,st:a.st,ang:a.ang,tr:a.tr.slice(-4)}))});
    return {
      mapId:A.id, crumbleOn:A.crumbleOn, crumbles:A.crumbles.map(c=>({state:c.state,t:c.t,dy:c.dy})),
      traps:R.traps.map(t=>({...t})), trapProj:R.trapProj.map(p=>({x:p.x,y:p.y})),
      ents:R.ents.map(plainEnt), particles:particles.slice(-200),
      timer:R.timer, roundsDone:this.roundsDone, trainerLevel:this.trainerLevel,
      elo:this.brain.elo, esGen:this.esGen,
    };
  }
}

// Fusionne un résultat d'entraînement dans la fiche du cerveau (IndexedDB)
async function writeTrainResult(db, id, d){
  const entry=await db.get('brains', id);
  if(!entry) return null;
  entry.brain=d.brain;
  entry.fitness=d.fitness; entry.elo=d.elo;
  entry.peakElo=Math.max(entry.peakElo||0, d.elo);
  entry.skills=d.skills||{};
  entry.personality=d.personality;
  entry.generation=Math.max(entry.generation||1, d.generation||1);
  entry.wins=(entry.wins||0)+(d.wins||0);
  entry.played=(entry.played||0)+(d.played||0);
  if(d.adamState) entry.adamState=d.adamState;
  await db.put('brains', entry);
  return entry;
}

// ── Pilote : fait tourner une session au rythme demandé ──
// post(msg) envoie les événements à la page. Messages reçus via handle() :
//   {type:'start', config}  {type:'speed', speed}  {type:'flush'}  {type:'stop'}
// Vitesses : 1/4/16 images par tick (≈60 ticks/s, avec aperçu), 0 = MAX (sans aperçu).
// Sauvegarde dans IndexedDB tous les 10 rounds, toutes les 15 s, sur 'flush' et à l'arrêt.
class TrainingDriver {
  constructor(post, db){
    this.post=post; this.db=db;
    this.session=null; this.speed=1; this.runId=0;
    this.saveChain=Promise.resolve(); this.lastSave=0; this.lastSaveRound=0;
  }
  handle(msg){
    if(msg.type==='start'){
      this._end(false);
      this.id=msg.config.id;
      this.speed=msg.config.speed;
      this.session=new TrainingSession(msg.config, m=>this._event(m));
      this.lastSave=performance.now(); this.lastSaveRound=0;
      this._loop();
    } else if(msg.type==='speed'){
      this.speed=msg.speed;
    } else if(msg.type==='flush'){
      this._save();
    } else if(msg.type==='stop'){
      this._end(true);
    }
  }
  _event(m){
    this.post(m);
    if(m.type==='round'){
      const s=this.session;
      if(s && (s.roundsDone-this.lastSaveRound>=10 || performance.now()-this.lastSave>15000)) this._save();
    }
  }
  // Sauvegardes enchaînées : deux lecture-modification-écriture ne se croisent jamais
  _save(final=false){
    const s=this.session; if(!s) return this.saveChain;
    const data=s.saveData(), id=this.id;
    this.lastSave=performance.now(); this.lastSaveRound=s.roundsDone;
    this.saveChain=this.saveChain.then(()=>writeTrainResult(this.db, id, data)).then(entry=>{
      if(entry) this.post({type:'saved', id, final, roundsDone:s.roundsDone,
        elo:entry.elo, fitness:entry.fitness, generation:entry.generation, wins:entry.wins, played:entry.played});
    }).catch(e=>this.post({type:'toast', msg:'Sauvegarde entraînement échouée : '+(e?.message||e), important:true}));
    return this.saveChain;
  }
  _end(save){
    if(this.session){
      if(save) this._save(true);
      this.session.stopped=true;
    }
    this.session=null; this.runId++;
  }
  _loop(){
    const id=++this.runId;
    const tick=()=>{
      const s=this.session;
      if(id!==this.runId||!s) return;
      if(this.speed===0){
        const t0=performance.now();
        while(performance.now()-t0<14 && !s.finished) s.advance(1);
      } else {
        s.advance(this.speed);
        const snap=s.snapshot();
        if(snap) this.post({type:'frame', snap});
      }
      if(s.finished){
        this._save(true);
        this.post({type:'done', roundsDone:s.roundsDone});
        s.stopped=true; this.session=null;
        return;
      }
      if(this.speed===0) yieldThen(tick); else setTimeout(tick, 1000/60);
    };
    tick();
  }
}
