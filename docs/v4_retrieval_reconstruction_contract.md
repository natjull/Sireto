# Contrat — Reconstruction retrieval V4 à 100 candidats

## But

Mesurer le retrieval gelé sur la nouvelle vérité V4 avant tout nouvel
entraînement. Le bon SIRET actif doit être présent dans au moins 99,0 % des
pools, avec 100 candidats au maximum par requête.

Cette étape ne mesure ni le classement final, ni la décision `AUTO_MATCH`.

## Entrées gelées

- noyau V4 courant :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759/` ;
- expansion V4-Fresh :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f/` ;
- anciens pools candidats :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068/` ;
- partitions actives V7 : `data/candidates_v7_all/` ;
- overlay fermé historique :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/stores/legacy_closed_overlay_c33b80855f560074_e39fddd/` ;
- politique d’admission déterministe définie dans
  `docs/retrieval_selective_recall100_contract.md`.

Chaque entrée est contrôlée par son manifeste et son SHA-256. Une
incompatibilité arrête le build.

## Périmètres

### Fit

Le fit exact combine :

- les 4 932 `MATCH_EXACT` du noyau V4, anciens train et dev réunis ;
- les 819 `MATCH_EXACT` de `fit_addition`.

Total attendu : 5 751 requêtes exactes.

Les listes de candidats historiques sont réutilisées sans modification, puis
réétiquetées avec la vérité V4 courante. Les 819 requêtes fraîches passent par
les mêmes canaux et la même admission gelée.

### Dev indépendant

Le dev contient uniquement les 305 `MATCH_EXACT` de `dev_new`. Il ne partage
aucun SIREN exact avec le fit.

### Données exclues

- `UNRESOLVED` : hors métrique et hors apprentissage ;
- `AMBIGUOUS` : hors gate Recall@100, car il n’existe pas de SIRET unique à
  chercher ; ces scènes seront reconstruites après franchissement du gate pour
  entraîner l’accepteur ;
- `holdout_sealed` : aucune lecture de benchmark, de label, génération de
  candidats, prédiction ou métrique dans cette étape ;
- ancien test final : définitivement fermé.

## Politique de retrieval gelée

Pour les requêtes fraîches :

1. calcul des canaux V7 actifs et overlay à profondeur interne 5 000 ;
2. fusion réciproque pondérée gelée ;
3. quotas overlay gelés ;
4. ordre déterministe avec SIRET croissant en cas d’égalité ;
5. sortie tronquée à 100 candidats.

Aucune règle, aucun poids et aucun quota ne peut être modifié après lecture des
résultats du nouveau dev. Le bon SIRET n’est jamais injecté.

## Artefact de sortie

Le build immuable produit sur le SSD :

```text
retrieval_v4/<build_id>/
  fit_exact.parquet
  dev_exact.parquet
  misses_fit.parquet
  misses_dev.parquet
  summary.json
  manifest.json
```

Une ligne contient au minimum l’identifiant de requête, le SIRET de vérité, la
liste ordonnée des candidats, le rang éventuel de la vérité, le nombre de
candidats et la provenance `historical_reuse` ou `fresh_frozen_retrieval`.

## Contrôles obligatoires

- 5 751 requêtes fit exactes et 305 requêtes dev exactes ;
- zéro doublon d’identifiant ;
- zéro chevauchement de SIREN exact entre fit et dev ;
- zéro requête à plus de 100 candidats ;
- zéro candidat dupliqué dans une liste ;
- zéro positif injecté ;
- zéro lecture du holdout ou de l’ancien test ;
- Recall@100 publié séparément pour le noyau historique, l’ajout frais, le fit
  combiné et le nouveau dev ;
- nombres bruts et intervalles de Wilson à 95 % publiés.

## Gate et décision

Le gate est franchi uniquement si :

- Recall@100 SIRET exact du fit combiné ≥ 99,0 % ;
- Recall@100 SIRET exact du nouveau dev ≥ 99,0 % ;
- tous les contrôles ci-dessus passent.

Verdict :

- `GO_RANKER_V4` : les deux recalls et tous les contrôles passent ;
- `PIVOT_RETRIEVAL_V4` : le plafond géographique permet 99 % mais l’admission
  perd trop de vérités ;
- `STOP_V4_DATA` : les entrées, les séparations ou le plafond géographique
  rendent le protocole invalide.

Les misses peuvent être décrites après la mesure pour décider du prochain
travail, mais aucune correction issue de leur analyse ne peut être présentée
comme validée sur ce même dev.
