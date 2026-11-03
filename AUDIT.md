# Audit Neural Arena — novembre 2026

Audit complet de `index.html`, `sw.js`, `manifest.json` et `icons/` avant la reprise du développement.
Chaque bug corrigé ci-dessous a été reproduit puis revérifié dans Chromium (Playwright).

## Corrigé dans cette passe

### Critique

| Problème | Effet pour le joueur | Correction |
|---|---|---|
| Un commentaire contenant `</script>` fermait le script avant la fin | ~385 lignes (« PATCH v3 » : couche d'attention, écran récapitulatif) n'ont **jamais été exécutées** et s'affichaient en texte caché dans la page. Aucun cerveau n'a jamais eu d'attention, malgré l'étiquette « GRU×4+ATT ». | Patch supprimé. Le réactiver aurait changé la sortie de tous les cerveaux déjà entraînés. |
| XSS via le dépôt communautaire | Un fichier de cerveau piégé exécutait du JavaScript à chaque lancement (sync automatique), sans clic. Il pouvait voler le token GitHub (scope `repo` = tous les dépôts privés) et se republier depuis le compte de la victime. Même chose via un fichier importé. | Échappement de tous les champs affichés, nombres forcés en nombres, validation stricte de la structure des cerveaux (`validateBrainStructure`), métadonnées nettoyées (`cleanBrainMeta`). L'aide recommande maintenant un token *fine-grained* limité au seul dépôt. |
| Les entraîneurs ne tiraient plus après le 1er round (PvP, salle d'entraînement, fond du menu) | La mémoire de l'agent était réinitialisée avec des champs manquants, d'où `NaN` et plus aucun tir. En entraînement, le cerveau n'apprenait jamais à esquiver. | `ScriptedAgent.reset()` unique. |
| En PvP avec un entraîneur, rien n'était sauvegardé | `store.get('brains', null)` levait une erreur silencieuse : ELO, poids et état Adam étaient perdus à chaque round. | Les entraîneurs sont exclus de la sauvegarde mais comptent comme adversaires (ELO fixe par niveau). |
| Changer de cerveau pendant un entraînement écrasait l'autre cerveau | Les poids du cerveau A étaient écrits dans B. | L'id est figé au démarrage et le sélecteur est bloqué pendant l'entraînement. |

### Important

- **Boucle de jeu** :
  - deux boucles pouvaient tourner en même temps (double clic, quitter puis rejouer), ce qui donnait un jeu deux fois trop rapide ;
  - la physique dépendait de la fréquence de l'écran : jeu 2,4× plus rapide en 144 Hz.

  La boucle est maintenant unique, à pas fixe de 60 pas/s.
- **Clavier** :
  - impossible de taper espace, `w`/`z` ou les flèches dans les champs (token, pseudo, repo) ;
  - touches bloquées après un alt-tab ;
  - écouteurs de la caméra libre ajoutés à chaque partie.
- **Visée IA** : l'angle n'était jamais ramené dans [-π, π]. Le calcul de la récompense de visée pouvait alors donner des valeurs énormes, que le réseau exploitait en tournant sur lui-même. `aimSnap` était aussi ignoré en PvP.
- **ELO multi-joueurs** : il n'était pas à somme nulle et dépendait de l'ordre de traitement. Il est maintenant calculé par paires, sur l'ELO d'avant le round.
- **PvP, poids sauvegardés** : les poids sauvegardés n'étaient jamais ceux qui avaient joué (une mutation non testée de plus à chaque round). Le cerveau en mémoire est maintenant resynchronisé.
- **Bouton « ⏭ Suivant »** : il comptait tous les cerveaux comme perdants (pénalité + mutation forte). Il passe désormais le round sans rien appliquer.
- **Salle d'entraînement** :
  - le cerveau sauvegardé était une mutation pas encore testée ;
  - la génération faisait +650 en 100 rounds (vu en test : 12 rounds → génération 21) ;
  - passer en vitesse MAX pendant un round bloquait l'entraînement pour toujours ;
  - le mode MAX gelait l'interface à chaque round ;
  - l'adversaire « Hall of Fame » remplaçait l'entraîneur définitivement ;
  - le score de population était divisé deux fois par la durée ;
  - les compétences repartaient de zéro à chaque session ;
  - les sauvegardes pouvaient se chevaucher.
- **Policy gradient** :
  - l'avantage était normalisé épisode par épisode, si bien que le bonus ±3 de fin de round ne distinguait plus victoire et défaite. Il utilise maintenant une baseline glissante ;
  - ajout d'une protection contre les mises à jour `NaN` ;
  - l'état Adam n'est plus partagé par référence entre individus ;
  - en PvP, l'apprentissage en ligne n'est plus perdu quand le round se termine juste après une mise à jour.
- **Gauntlet de création** :
  - la physique utilisait la grande arène (plateformes invisibles) alors que l'écran montrait un sol plat ;
  - aucune limite de candidats ;
  - le rookie ne pouvait pas toucher le candidat.

  Les entités ont maintenant leur propre carte de collision, le test s'arrête à 30 candidats en gardant le meilleur, et les flèches du rookie éliminent le candidat.
- **Export / partage** : ils envoyaient d'anciens poids (`bestBrain`, jamais mis à jour). C'est maintenant le cerveau actif qui part.
- **Backup** : le fichier « Backup tout » ne pouvait pas être réimporté. La restauration est ajoutée, avec une liste blanche de champs.
- **Notifications** : après la première erreur JS, plus aucun toast ne s'affichait de la session.
- **Simulation du menu** : elle continuait de tourner en arrière-plan pendant le jeu (CPU et batterie).
- **IndexedDB** :
  - les écritures étaient confirmées avant le commit ;
  - aucun message en cas de blocage ou d'échec.

  Le service worker est désormais enregistré même si IndexedDB échoue, et `navigator.storage.persist()` est demandé.
- **Service worker** :
  - il mettait en cache les réponses de l'API GitHub (y compris celles obtenues avec le token) et les réponses d'erreur ;
  - la navigation hors-ligne échouait dès qu'il y avait des paramètres dans l'URL.
- **Upload** : le nom de fichier venait du pseudo brut (`#`, `?`, `/` cassaient le chemin) et la branche était forcée à `main`.

### Nettoyage

- Supprimé :
  - le « PATCH v3 » mort ;
  - le module de tournoi inaccessible (~480 lignes, aucun bouton ne l'ouvrait) ;
  - la classe `Brain` legacy ;
  - `checkHitsMulti`, `newBrain`, `hallOfFame`, `bestBrain`, `stagnationCounter`, `multiAgents`, `freeCamVx/Vy` ;
  - le tableau `acts` recopié à chaque frame ;
  - les réglages sans effet (K-factor, pénalité défaite, seuil stagnation).
- Le « PATCH v4 » (cache `_lastX/_lastZ` de `DenseLayer`) est intégré à la classe. `init()` est appelé en toute fin de script.
- Une seule fonction de désérialisation.
- Documentation des 26 entrées du réseau à jour.
- Étiquettes de topologie calculées.
- Icônes : 3 copies identiques de 1,8 Mo (1024 px) remplacées par de vraies icônes 192 et 512, plus une version maskable (5,4 Mo → 0,9 Mo).

## Passe 2 — arènes, apprentissage, gameplay

### Arènes à la manière des jeux d'arène sur un seul écran

- **Moteur en tuiles** : arène de 40×24 tuiles de 20 px (800×480) qui tient entièrement à l'écran. Il n'y a plus de caméra qui suit ni de caméra libre.
- **Bords rebouclés** : sortir à gauche fait réapparaître à droite, tomber dans un puits fait réapparaître en haut. L'IA, les entraîneurs, les flèches et les touches en tiennent compte.
- **4 cartes originales** :
  - Crypte ;
  - Forge ;
  - Jardins suspendus ;
  - Beffroi.
- **Validation automatique des cartes** :
  - rebouclage cohérent ;
  - chaque point d'apparition est posé sur un sol ;
  - toutes les zones sont accessibles en sautant.
- **Éléments de décor** :
  - plateformes traversables (on peut s'en laisser tomber avec la touche bas) ;
  - blocs friables qui tombent vraiment ;
  - rendu par thème mis en cache.
- **Pièges** :
  - placés selon la géométrie de chaque carte ;
  - lanceur, téléporteur (dans les deux sens) et zone électrique corrigés.
- **Choix de l'arène** en PvP et à l'entraînement. « Aléatoire » change de carte à chaque round ; à l'entraînement, c'est ce qui force l'IA à généraliser.
- **Vérification par simulation** : 40 rounds IA contre entraîneurs ont donné 0 entité coincée, 0 NaN, des passages par les bords dans les deux axes, et des rounds tous conclus par une touche.

### Apprentissage

| Mesure | Avant | Après |
|---|---|---|
| Passe avant du réseau | 84 µs | 13 µs (résultats identiques au bit près) |
| Copie d'un cerveau | 1,1 ms | 0,02 ms |
| Entrées IA (capteurs) | 19 µs | 1,6 µs |
| Gradients d'un lot | 3,4 ms | 0,24 ms (poids identiques) |
| Image simulée en entraînement | ~234 µs | ~46 µs |

- **Exploration réelle** : à l'entraînement, les actions sont tirées selon les probabilités du réseau, et le gradient apprend des actions réellement jouées (avec un bonus d'entropie). En partie, l'IA reste déterministe.
- **Sélection appariée en coulisse** : le champion et un candidat dont les experts GRU sont mutés jouent les mêmes 16 rounds (même carte, mêmes apparitions, même hasard). Le candidat ne remplace le champion qu'avec une avance nette. L'arbitre apprend par gradient. Cela remplace la marche aléatoire (une mutation après chaque round) et la population en round-robin, trop bruitée.
- **Choix des réglages** : plusieurs variantes ont été comparées sur le taux de victoire en jeu déterministe contre SERA. Un cerveau neuf passe de 0-30 % à environ 55-60 % en quelques centaines de rounds, soit 10 à 20 secondes en mode MAX. Les variantes sans gradient progressent mal.
- **Entraînement sans fin** :
  - sauvegarde quand l'écran se verrouille ou que l'onglet se ferme ;
  - pastille « entraînement en cours » sur les autres écrans.
- **Fiche du cerveau** : l'entraînement compte maintenant victoires et parties.

### Gameplay

- **Touches simultanées** : appliquées en même temps, ce qui permet une égalité. Une brève invulnérabilité suit chaque coup.
- **Entraîneurs** : ils gardent une zone d'esquive au lieu de coller l'adversaire (issue #1), et la visée de KURO n'est plus appliquée deux fois.
- **PvP** : le cerveau n'est plus muté au hasard après chaque round ; il apprend de ses parties par gradient.
- **Contrôles** :
  - visée souris possible sur un PC tactile ;
  - aide clavier ZQSD/WASD.
- **Mobile** :
  - en paysage, l'arène prend toute la hauteur et les joysticks se superposent sur les côtés ;
  - en portrait, une invitation propose de tourner l'écran.

## Reste à faire (par priorité)

1. **Plafond d'apprentissage** : vers 55-60 % contre SERA. La sortie `aimSnap` (visée automatique sur la position actuelle) domine le jeu. Pistes :
   - la limiter (temps de recharge, viser la position prédite) ;
   - ajouter un self-play plus fréquent ;
   - entraîner aussi la projection de sortie des experts par gradient.
2. **Mécaniques du genre**, à décider. Toutes changent les entrées de l'IA et demandent une migration des cerveaux :
   - carquois limité et ramassage des flèches plantées ;
   - écrasement d'un adversaire en lui sautant sur la tête ;
   - esquive ou rattrapage de flèche.
3. **Évaluation de la sélection** : elle se fait sans pièges et en 1 contre 1 ; on pourrait y inclure les pièges choisis.
4. **Plusieurs humains en local** : seul le premier humain reçoit les commandes.
5. **Onglet en arrière-plan** : le navigateur y ralentit fortement les minuteries. Un entraînement vraiment continu demanderait un Web Worker.
6. **Fitness en PvP** : elle ignore encore les vraies stats (esquive, précision).
7. **Pièges et entrées** : les projectiles de pièges ne font pas partie des entrées du cerveau.

### Technique

- Remplacer les `onclick="..."` inline par `addEventListener`, ce qui permettrait une CSP sans `unsafe-inline`.
- Accessibilité : `user-scalable=no` combiné à des polices de 4–6 px rend le Labo et les Stats illisibles sur mobile.
- Héberger la police localement : l'`@import` Google Fonts bloque le rendu.

### Issue #1 (ouverte) — état

- Entraîneurs trop collés au contact : **fait** (zone d'esquive).
- Fond de l'arène : **fait** (4 arènes à thème).
- Entraînement infini en arrière-plan : **fait** (mode sans fin, sauvegarde à la mise en veille, pastille). Limite : le navigateur ralentit un onglet caché.
- Sauvegarde sur Google Drive : à faire.
- Radar de fitness par expert : à faire.
