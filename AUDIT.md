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

## Reste à faire (par priorité)

### Gameplay et apprentissage

1. **Policy gradient** : les actions ne sont pas échantillonnées (`p > 0.5` déterministe), et les sorties 3/4 (rotation de visée, continues) sont traitées comme des Bernoulli. Il faudrait échantillonner pendant l'entraînement et sortir 3/4 de la perte.
2. **PvP** : le cerveau mute après chaque round sans sélection (marche aléatoire). À envisager : ne garder la mutation que si elle fait mieux, ou ne muter qu'entre deux sessions.
3. **Fitness en PvP** : elle ignore les vraies stats (esquive, précision) ; seuls la survie et la victoire comptent.
4. **Mode population** : chaque individu ne joue qu'un round par génération, donc la sélection est très bruitée.
5. **L'entraînement continue quand on quitte l'écran.** L'issue [#1](https://github.com/ShinjiEbi/neuralarena/issues/1) le demande justement, mais il faudrait un indicateur visible et une sauvegarde sur `beforeunload` (on perd jusqu'à 9 rounds si on ferme la page).
6. **Fenêtre de création** : quand un cerveau actif existe, les curseurs H/out/arb/xavier n'ont pas d'effet. Le candidat est un clone muté du cerveau actif ; c'est maintenant signalé dans l'aperçu.
7. **Plusieurs humains** : seul le premier humain reçoit les commandes alors que le lobby en autorise plusieurs.
8. **Pièges** :
   - le piège d'effondrement est seulement visuel ;
   - le 2ᵉ piège d'effondrement ne se déclenche jamais ;
   - le lanceur est réarmé par n'importe quelle entité ;
   - les projectiles de pièges ne font pas partie des entrées du cerveau.
9. **Touches simultanées** : l'index le plus bas gagne. Si tout le monde meurt dans la même frame, l'écran affiche « TEMPS ÉCOULÉ ». Il n'y a pas de frames d'invulnérabilité après un coup.
10. **Stats d'entraînement** : l'entraînement ne met pas à jour victoires/parties dans la fiche du cerveau.
11. **Visée de KURO** : elle est tournée deux fois par frame.
12. **Laptops tactiles** : la visée souris est désactivée, et le clavier n'a pas de touche de visée.

### Technique

- Remplacer les `onclick="..."` inline par `addEventListener`. Cela permettrait une CSP sans `unsafe-inline`.
- Factoriser la logique en double :
  - test de touche flèche/rectangle (5 copies) ;
  - boucle interval/MAX ;
  - dessin des plateformes ;
  - rétropropagation de `learn`/`learnOnline`.
- Performance :
  - `drawDungeonBG` reconstruit étoiles et dégradés à chaque frame ;
  - `clone()` construit un cerveau aléatoire puis le jette (~2 ms).
- Accessibilité : `user-scalable=no` combiné à des polices de 4–6 px rend le Labo et les Stats illisibles sur mobile.
- Héberger la police localement : l'`@import` Google Fonts bloque le rendu.

### Issue #1 (ouverte)

- Les entraîneurs collent trop au contact : ajouter une zone d'esquive autour des joueurs.
- Refaire le fond de l'arène.
- Entraînement infini en arrière-plan.
- Sauvegarde sur Google Drive.
- Radar de fitness par expert.
