# Recherche définitivement figée — V142, 17/10/2026

Décision Nathan : « réinspire-toi des modèles XGBoost […] fige le retrieval
Définitivement ». **Plus aucune expérience de recherche.** Toute modification
future de recette, coefficients, sources, index, caches ou candidats exige
une **nouvelle décision écrite de Nathan**, y compris un éventuel fit production.
L'objectif initial99,2 % reste suspendu, pas atteint ni supprimé.

## Recette et sceau

- Script principal : `scripts/embeddinggemma2_v142_bano_native_ablation.py`,
  commit `ae7c3e05` ; helpers/derniers commits et SHA complets dans le JSON voisin.
- H = `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` ; run =
  `H/v142/bano_native_20261013_night/all_new_sources_max_bano`.
- Seule variante : `outputs.json["learned_native_ooc"]`, 526×100 ordonnés,
  exact@100 **507/526**, top1 exact **353/526**. Aucun candidat régénéré.
- SHA256 outputs : `fb691a70311392c103955e1b4fe71ee03fbdabbb2dcc87d2fb32e9bf6c5d3218`.
- Preuve nationale sans prépool : 51 preuves numériques +7 composantes,
  coefficients V71 figés (base28 V32 inchangée), quantification native héritée ;
  max BANO/OSM/nom fournisseur/adresse temporelle par établissement, max RNA/BANATIC
  par entreprise ; pas de double vote. Réemploi uniquement si entrées équivalentes.
- Tantivy : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/`
  `hierarchical_retrieval_v3/4b332899d8698354/tantivy`, 82 772 691 documents.
- `docs/retrieval_frozen_v142.json` : **3681 SHA256 réellement recalculés**,
  148 479 301 179 octets lus en163,20 s : sorties/reçus, cinq modèles58, cinq bases28, affectation,
  352 fichiers index, caches numériques/activité/bonus, contextes sources et code.
  Portée : rejouer les entrées scellées et leurs autorités directes, pas reconstruire
  tous les ancêtres ; ni jeux fermés ni shards de labels d'apprentissage parcourus.
- Commande de scellement : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_freeze_v142.py`.

## Version production : elle n'existe pas dans cette recette

Dans `embeddinggemma2_v32_learned_query.py::calibrate`, partition interne =
`sha256(group_seed + '|' + SIREN_humain)[:8] % 5` (entier big-endian).
V142 charge `v71/.../calibration/assignment.json` puis `models[str(assignment[qid])]`.
Ces partitions internes0–4 ne sont **pas** les folds population0/1 fermés.
Les cinq `model_0.json` à `model_4.json` excluent les entreprises de leur
partition ; aucun `full_fit` unique n'est produit par ces scripts/manifests.
Une entreprise inconnue n'a ni label ni affectation : ce routage ne peut servir.

Il faudrait un seul fit V32 puis résiduel V71 sur les groupes éligibles du pilote
2/3/4 (en conservant les exclusions de contrôle encodeur), sans moyenne des cinq
modèles, et une construction de preuves en ligne sans vérité ni caches propres
aux526. Estimation conditionnelle **45–75 min** pour ce seul fit/minage : reçu
V173 comparable284 requêtes =2571,82 s, extrapolé375 éligibles ≈3396,12 s ;
optimiseurs seuls <0,1 s. Ce n'est pas une mesure V142 full-fit, ni le coût des
sources/préparations, d'une régénération ou d'une validation production.
**Rien de cela n'est calculé ici.** Une telle version ne conserverait pas par
construction les listes507/526 : elle demanderait autorisation puis validation
indépendante, SLO/mémoire et audit sources/encodeur (actuellement non full-OOF).

L'étape2 autorisée le17/10 apprend seulement à choisir dans les100 immuables.
Les labels restent inchangés ; folds2/3/4 seuls, aucun web, VM ou jeu fermé.
