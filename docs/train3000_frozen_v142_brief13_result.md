# Brief13 — smoke20 complet, ready scellé, aucun3000 lancé

## Fait
Code `8eb4cf1b` : préparation12Gio uniquement hydraulic Stopped et libre+inactive≥9Gio,
attente60s sinon ; arrêt de l'enfant possédé et réadmission si VM redémarrée/incertaine.
Génération8Gio puis une escalade12Gio selon la règle citée par le lead ; marge256Mio,
un worker/thread, disque10Gio, TMP privé, recette et protections inchangés.
Instrumentation externe des fonctions originales pour mesurer extraction/copie/encodage.

## Preuve réelle
Commande exécutée : `.venv/bin/python -u scripts/train3000_frozen_v142_supervisor.py --prepare-ready`.
Un lancement détaché PID86125, une attente bloquante106789,062s, sortie possédée0.
Quatre interruptions VM : enfants arrêtés par SIGTERM puis attente/réadmission,
37920s de pas60s dans CE lancement ; aucune action sur hydraulic.
Les47820s du rapport automatique sont cumulés avec les anciens essais, pas le seul brief13.
Préparation complète20/20 et20 générations native100 avec reçus possédés exit0 :
**100 candidats uniques par dossier,2000 candidats**, folds2/3/4, aucun label décodé.
Fidélités pilotes quatre familles20/20/écart0 conservées ; numérique51 bit à bit20/20.
Gel3681 SHA/148479301179octets conforme ; signatures toujours identiques après le smoke.

| Mesure (processus réussis, démarrage inclus) | Préparation | Génération |
|---|---:|---:|
| Temps20 dossiers |26878,775s|341,343s|
| Secondes/requête |1343,939|17,067|
| Pic RSS processus/arbre |7124926464octets (6,636Gio)|9814081536octets (9,140Gio)|

Trois générations ont requis l'escalade8→12Gio ; leurs essais8Gio coûtent49,259s actifs
supplémentaires : génération opérationnelle observée390,601s, soit19,530s/requête.
Les temps réussis excluent les reprises partielles VM (41393,955s actifs perdus) et attentes.
Le pic prudent RSS+Metal enregistré pendant la préparation réussie est8237645824octets ;
les pics RSS du tableau ne doivent pas être confondus avec un plafond physique12Gio consommé.

## ETA / choix
`ready.json` projette47,291jours préparation +14,223h génération =47,884jours utiles
pour3000, dont20 déjà réalisés ; reste2980 :47,568jours, hors attentes/interruption VM.
Avec les essais8→12Gio observés inclus dans la projection génération :16,275h pour3000,
et **47,653jours utiles restants** au total. Estimations descriptives, pas un SLO.
L'ETA tient compte des corpus/banques gelés copiés/réutilisés et des20 sorties scellées.
Les nouvelles extractions ne sont PAS partagées entre lots dans le code actuel :
124038385 expositions nouvelles restantes, dédupliquées par vrai lot20, pas un coût
unique par commune. Ajouter un cache inter-lots n'était pas nécessaire pour mesurer
le code existant et n'a pas été fait. Extraction/copie/SITE pondérés par nombres de sites ;
autres coûts par lot. Aucun délai calendaire borné si la VM redémarre régulièrement.

## Ready / commande exacte (fournie, NON exécutée)
Sceau : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/ready.json`.
Sceau de complétion : même dossier, `ready_complete.json`, commit automatique `a587ade6`.
Les SHA sources, reçus,20 IDs et sorties ont été revérifiés après terminaison.
```bash
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/train3000_frozen_v142_supervisor.py --full >> /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/full3000.log 2>&1 < /dev/null &
```

## Reste / risques
Full à autoriser séparément ; aucun3000 lancé, aucune évaluation qualité ni apprentissage.
Coefficient-OOF seulement, pas encoder-OOF ; aucune nouvelle prétention de recall/SLO.
Les20 dossiers ne garantissent ni le plafond12Gio ni les durées pour tous les3000.
Reçus/essais/sources/caches526/labels/jeux fermés/pending/travaux étrangers conservés.
Preuves chiffrées et références complètes : `train3000_frozen_v142_brief13_result.json`.
