# Brief13 — smoke20, préparation autorisée à12Gio

Autorité : `/Users/nathanjullia/.pi/agent/tasks/briefs/sireto-site-target-13.md`.
Cette décision remplace pour la préparation le STOP8Gio du brief12 ; pas de full3000.
Code du jalon : `git log -1 --format=%H -- scripts/train3000_frozen_v142_supervisor.py`.

- Préparation admise à12Gio seulement si hydraulic confirmée Stopped et libre+inactive≥9Gio ; sinon pas de worker et attente60s renouvelable. Marge256Mio, arrêt terminal à12Gio, disque≥10Gio et TMP privé conservés.
- Autres workers du superviseur :8Gio, VM Stopped, libre+inactive≥7,5Gio ; une seule escalade génération12Gio, même admission≥9Gio. Aucun contrôle de la VM.
- Redémarrage/état incertain VM pendant tout worker : SIGTERM à l'enfant possédé, attente de sa sortie, conservation des fichiers/reçus, puis réadmission seulement VM Stopped. Pas de worker concurrent.
- Quatre fidélités20/20/écart0 existantes réutilisées après contrôle des références ; les constructeurs scientifiques, poids, textes, features, sources et folds ne changent pas.
- Chronométrage externe des fonctions originales d'extraction/copie et encodage SITE ; aucun résultat ni calcul original modifié.
- Le code actuel réemploie les corpus/banques documentaires gelés et les20 tâches smoke complètes, mais ne partage PAS les nouvelles extractions entre lots. L'ETA ne doit pas annoncer un coût unique par commune : exposition dédupliquée dans chaque vrai lot20, puis sommée sur les149 lots restants. Les temps extraction/copie/encodage SITE sont pondérés par nombres de sites ; les autres coûts sont extrapolés par lot. Hypothèse descriptive, pas une garantie de délai/SLO ; attentes ressources exclues.
- Au plus3 cycles pour des bugs de plomberie ; une seule attente bloquante du contrôleur détaché par lancement, sans polling agent. Échec de fidélité/recette ou garde12Gio : STOP, pas de plafond supérieur ni de corpus réduit.
- `ready.json` et commande future exacte uniquement après préparation achevée et20 sorties native100 uniques avec reçus possédés exit0. Aucun label décodé, apprentissage, réseau, action VM, modification de caches526/pending/travaux étrangers ou lancement --full.
