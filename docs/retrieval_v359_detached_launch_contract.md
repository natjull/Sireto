# V359 — lancer un nouveau run complet, pas reprendre le STOP V352

Accord humain « Continue »,6octobre2026. Transport/restarts du V352 éprouvé,
liés au nouvel orchestrateur356, au worker354 corrigé353, à l'autorité357 et
à l'évaluation358. Même panel328, budgets, dénominateur, scores et cap100.
Voir `retrieval_v356_checkpoint_contract.md` pour les protections complètes.

## Lancement après commit de freeze

Depuis la racine SIRETO :

```sh
.venv/bin/python -m scripts.launch_retrieval_v359_resumable prepare --control /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v359_persistent_launcher/job_20261006_a --generation /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v344_corroborated_population/autoresume_v356/run_20261006_a
.venv/bin/python -m scripts.launch_retrieval_v359_resumable start --control /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v359_persistent_launcher/job_20261006_a
```

Deux dossiers neufs ; prepare vérifie les hashes sans lancer de calcul, start
crée le superviseur avec session POSIX indépendante, stdinNULL/closefds,
claim unique, heartbeat et journaux CATNAT. Un seul producteur/thread,
caffeinate temporaire uniquement, aucune modification permanente système.
Réessais internes bornés ; ne pas réexécuter start après lancement. STOP,
drift, budgets épuisés, cibles commencées : pas restart. Aucune importation des
388 sorties précédentes ou des quatre lignes du probe355 ; recomputation
complète assumée pour conserver le contrat, aucun changement de recette.

`state.json` et `workflow.log` dans le root contrôle donnent l'avancement.
RUNNING/heartbeat ne valent pas propre exit0/PASS ni un recall. COMPLETE exige
génération356 ALL2 000 puis véritable mesure358 postseal. Les phases cibles
restent interdites avant ALL500 lots ; exact/op séparés, fold1fermé/avalgelé.
Les crashs du superviseur lui-même/reboot/volume restent des limites, pas
garantie de reprise après extinction ni prétention de SLO/99,2% validé.
