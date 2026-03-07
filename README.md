# 🧠⚔️ Neural Arena

**PWA de combat IA évolutive à entraînement distribué**

> Joue contre une intelligence artificielle neuronale, entraîne-la par neuroévolution à chaque partie, et partage les meilleurs cerveaux avec la communauté via GitHub.
---

## 🎮 Concept

Neural Arena est une plateforme de jeu où les humains entraînent collectivement une IA de combat. Le principe est simple :

1. **Tu joues** contre l'IA dans Arena Archer (tir à l'arc en 2D)
2. **L'IA évolue** à chaque partie grâce à la neuroévolution
3. **Tu partages** ton meilleur cerveau sur GitHub
4. **Les autres joueurs** téléchargent ton cerveau et jouent contre
5. **L'espèce IA progresse** collectivement, comme un Folding@home du gaming

Chaque joueur est une "île" d'évolution. Les meilleurs cerveaux migrent entre les îles via GitHub — c'est du **island model** de neuroévolution distribuée.
---

## 🕹️ Jeux disponibles

### Arena Archer

Arène de plateformes en 2D. Deux archers (joueur vs IA) s'affrontent avec des arcs et des flèches infinies. Le premier touché perd le round.

**Physique réaliste des flèches :**
- Trajectoire parabolique (gravité)
- Traînée aérodynamique (drag)
- Inertie héritée du tireur
- Recul au tir
- Collision et encastrement dans les plateformes

**Contrôles clavier :**

| Touche | Action |
|--------|--------|
| `A` / `D` | Déplacement gauche/droite |
| `W` | Saut |
| `↑` / `↓` | Viser haut/bas |
| `Espace` | Tirer |

**Contrôles tactiles (mobile) :**
- **Joystick gauche** — Déplacement + saut (tirer vers le haut)
- **Joystick droit** — Direction de visée + tir automatique
- **Bouton SAUT** — Saut dédié (bas-droite)

---

## 🧠 Architecture neuronale

### Réseau de neurones (MLP)

```
Entrées [16] → Couche cachée [20] → Couche cachée [14] → Sorties [6]
```

**Topologie :** `16 → 20 → 14 → 6`  
**Paramètres totaux :** ~700 poids et biais  
**Activations :** LeakyReLU (couches cachées), Sigmoïde (sorties)  
**Initialisation :** Xavier/Glorot

### Entrées du réseau (16 neurones)

| # | Entrée | Description |
|---|--------|-------------|
| 0-1 | `relX`, `relY` | Position relative du joueur (normalisée par viewport) |
| 2-3 | `myVx`, `myVy` | Vélocité de l'IA |
| 4-5 | `enemyVx`, `enemyVy` | Vélocité du joueur |
| 6 | `aimAngle` | Angle de visée actuel |
| 7 | `onGround` | Au sol (0 ou 1) |
| 8-9 | `nearArrowX`, `nearArrowY` | Position relative de la flèche ennemie la plus proche |
| 10-11 | `nearArrowVx`, `nearArrowVy` | Vélocité de cette flèche |
| 12-13 | `edgeL`, `edgeR` | Distance aux bords du monde |
| 14 | `timePressure` | Pression temporelle (0→1 sur 30s) |
| 15 | `entropy` | Bruit aléatoire (exploration stochastique) |

### Sorties du réseau (6 neurones)

| # | Sortie | Seuil | Action |
|---|--------|-------|--------|
| 0 | `moveLeft` | > 0.5 | Déplacement gauche |
| 1 | `moveRight` | > 0.5 | Déplacement droite |
| 2 | `jump` | > 0.5 | Saut |
| 3 | `aimUp` | > 0.5 | Viser vers le haut |
| 4 | `aimDown` | > 0.5 | Viser vers le bas |
| 5 | `shoot` | > 0.6 | Tirer (seuil plus élevé) |

---

## 🧬 Mécanismes d'évolution

### Neuroévolution par sélection directe

L'IA n'utilise pas de backpropagation. L'apprentissage se fait uniquement par **neuroévolution** :

- **L'IA gagne un round :** le cerveau est sauvegardé comme "best brain", ajouté au Hall of Fame, puis légèrement muté (exploration)
- **L'IA perd un round :** mutation plus agressive, ou crossover avec le meilleur cerveau, ou crossover entre deux cerveaux du Hall of Fame

### Mutations gaussiennes

Les mutations suivent une distribution gaussienne (Box-Muller) plutôt qu'un simple bruit uniforme :
- Beaucoup de petites mutations (ajustements fins)
- Quelques grandes mutations rares (sauts exploratoires)
- Taux de mutation adaptatif : 5% après victoire, 15-25% après défaite

### Crossover

Le crossover utilise un **point de coupure par couche** :
- Pour chaque couche, un point de coupure aléatoire est choisi
- Les neurones avant le point viennent du parent A
- Les neurones après viennent du parent B

### Hall of Fame

Les 15 meilleurs cerveaux sont conservés en mémoire. Quand l'IA perd, il y a une chance qu'elle soit remplacée par un crossover de deux champions du Hall of Fame. Cela évite les pertes catastrophiques de bons comportements.

---

## 💾 Stockage local

Les cerveaux sont stockés dans **IndexedDB** sur l'appareil du joueur :

- Persistance entre les sessions (pas de perte à la fermeture)
- Plusieurs cerveaux peuvent coexister (Laboratoire)
- Backup/restore complet en JSON
- Stats par cerveau : génération, fitness, victoires, parties jouées

### Format JSON d'un cerveau

```json
{
  "format": "neuralarena-brain-v1",
  "game": "arena-archer",
  "name": "MonPseudo",
  "generation": 42,
  "fitness": 850,
  "wins": 34,
  "played": 50,
  "topology": [16, 20, 14, 6],
  "params": 698,
  "brain": {
    "topo": [16, 20, 14, 6],
    "w": [/* poids par couche */],
    "b": [/* biais par couche */]
  },
  "exported": "2025-01-15T14:30:00.000Z"
}
```

---

## 🌐 Système distribué via GitHub

### Architecture

```
Joueur A (mobile)          GitHub Repo             Joueur B (mobile)
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  Joue contre │        │  /brains/    │        │  Joue contre │
│  l'IA locale │───────>│   a_gen50.json│<───────│  l'IA locale │
│              │  PUSH  │   b_gen80.json│  PULL  │              │
│  Évolue le   │        │   c_gen120.json        │  Évolue le   │
│  cerveau     │<───────│              │───────>│  cerveau     │
│              │  PULL  │              │  PUSH  │              │
└──────────────┘        └──────────────┘        └──────────────┘
```

### Fonctionnement

**Lecture (sans authentification) :**
- L'app utilise l'API GitHub Contents pour lister les fichiers dans `/brains/`
- Chaque fichier `.json` est téléchargé et parsé
- Le classement est affiché par fitness décroissante
- Le joueur peut télécharger n'importe quel cerveau en un tap

**Écriture (avec token) :**
- Le joueur crée un [Personal Access Token](https://github.com/settings/tokens/new?scopes=public_repo&description=NeuralArena) (scope `public_repo`)
- Le token est stocké localement dans IndexedDB (jamais transmis ailleurs que GitHub)
- Un tap sur "Push" crée un commit avec le cerveau sérialisé en JSON

### Configuration

1. Crée un repo GitHub public (ex: `ton-pseudo/neuralarena-brains`)
2. Crée un dossier `/brains/` à la racine
3. Dans l'app → Réseau → ⚙ Config → entre le repo et ton pseudo
4. Optionnel : ajoute un token pour pouvoir push des cerveaux

---

## 📱 Installation PWA

### GitHub Pages

1. Fork ou clone ce repo
2. Active GitHub Pages (Settings → Pages → Source: `main` / `root`)
3. Accède à `https://ton-pseudo.github.io/neural-arena/`

### Installer sur mobile

1. Ouvre l'URL dans Chrome/Safari
2. Chrome : menu ⋮ → "Ajouter à l'écran d'accueil"
3. Safari : bouton partage → "Sur l'écran d'accueil"
4. L'app fonctionne ensuite hors-ligne

### Structure des fichiers

```
neural-arena/
├── index.html          # Application complète (single-file)
├── manifest.json       # Manifest PWA
├── sw.js              # Service Worker (cache offline)
└── icons/
    ├── icon-192.png
    └── icon-512.png
```

---

## 🏗️ Structure du code

L'application est contenue dans un seul fichier `index.html` pour simplifier le déploiement. Voici les modules internes :

| Module | Description |
|--------|-------------|
| `Brain` | Réseau de neurones (forward, clone, mutate, crossover, serialize) |
| `Store` | Abstraction IndexedDB (brains + config) |
| `GitHubSync` | Sync automatique avec l'API GitHub (lecture/écriture) |
| `Ent` | Entité de jeu (physique, collision, tir, rendu) |
| `Joystick` | Système de joystick tactile dual (mouvement + visée) |
| Game Loop | Boucle principale (input → AI → physics → render) |
| Evolution | Neuroévolution (mutation, crossover, hall of fame) |
| UI | Navigation multi-écrans (Menu, Jeu, Labo, Communauté) |

---

## 🗺️ Roadmap

### Court terme
- [ ] Améliorer l'IA de base (ajouter mémoire récurrente / GRU)
- [ ] Système ELO pour classer les cerveaux objectivement
- [ ] Mode spectateur (IA vs IA)
- [ ] Replay des meilleurs rounds

### Moyen terme
- [ ] Nouveaux jeux (course, puzzle, combat rapproché)
- [ ] Auto-évaluation au push (le cerveau doit battre le top 3 pour être accepté)
- [ ] Sync automatique périodique (pas seulement manuelle)
- [ ] Statistiques de progression (graphes d'évolution)

### Long terme
- [ ] Backend léger (Firebase/Supabase) pour sync temps réel
- [ ] Matchmaking P2P (joueur vs joueur via WebRTC)
- [ ] Topologies neuronales variables (NEAT / HyperNEAT)
- [ ] Communication inter-agents (langage émergent)
- [ ] API publique pour soumettre des cerveaux programmatiquement

---

## 🧪 Expérimenter

### Créer un cerveau optimisé manuellement

Tu peux éditer un fichier JSON et injecter des poids pré-calculés :

```javascript
// Exemple : forcer le neurone "jump" à s'activer quand une flèche approche
// Le neurone d'entrée #8 (nearArrowY) connecté au neurone de sortie #2 (jump)
// avec un poids fort négatif (flèche au-dessus → sauter)
```

### Observer le réseau en temps réel

Pendant le jeu, la visualisation du réseau neuronal est affichée en haut à droite. Les nœuds verts sont actifs positivement, les rouges négativement. L'épaisseur des connexions reflète le poids synaptique.

### Accélérer l'entraînement

Joue agressivement : les rounds courts (victoire ou défaite rapide) génèrent plus de générations par minute. La pression temporelle intégrée aux entrées du réseau pousse aussi l'IA à agir plus vite au fil du temps.

---

## 🤝 Contribuer

1. Fork le repo
2. Joue et entraîne ton IA
3. Push ton meilleur cerveau dans `/brains/`
4. Ouvre une PR si tu améliores le code
5. Crée une Issue pour proposer de nouveaux jeux

### Convention de nommage des cerveaux

```
pseudo_genXX_fYYY_timestamp.json
```

Exemple : `romain_gen142_f1250_m4x8k2.json`

---

## 📄 Licence

MIT — Utilise, modifie, distribue librement.

---

<p align="center">
  <strong>Chaque partie que tu joues rend l'IA un peu plus intelligente.</strong><br>
  <em>L'évolution est collective.</em>
</p>
