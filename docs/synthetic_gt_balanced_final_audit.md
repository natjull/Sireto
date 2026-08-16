# Audit final unique du corpus GT synthétique équilibré

Cet audit est exécuté une seule fois après que le registre compté atteint au
moins 20 000 variantes promues. Il est indépendant du runner de production :
il ne génère aucun texte, ne relance aucun retrieval et ne modifie aucun batch.

Le script `scripts/audit_synthetic_gt_balanced_final.py` reconstruit les
artefacts enregistrés et applique quatre contrôles :

1. validation déterministe de 100 % des lignes, contrats, inspirations,
   décisions critic et chaînes de hashes agentiques ;
2. vérification de chaque audit full-SIRENE déjà scellé, de son ledger et des
   snapshots SIRENE gelés, sans refaire les scans déjà exécutés par batch ;
3. publication séparée des vues SIRET exacte et équivalence opérationnelle,
   sans utiliser la seconde pour satisfaire le gate exact ;
4. distribution cumulative du synthétique, des 17 054 lignes réelles, des
   seuls folds train 2/3/4 et de leur union disponible. Le rapport distingue
   les volumes bruts des poids de scène et ne prétend pas qu'une ligne est
   entraînable avant son retrieval top 100 naturel.

## Échantillon de réalisme borné

Le même audit fige au maximum 200 lignes. Il couvre d'abord chaque cellule
peuplée `difficulty × augmentation_stratum × name_relation ×
location_relation`, puis répartit le reliquat proportionnellement. Si les
cellules fines dépassent 200, la strate de repli retire seulement la relation
de localisation. L'ordre est un SHA-256 déterministe avec le sel publié dans
le rapport.

Chaque ligne doit recevoir exactement une décision :

- `PASS` ;
- `BORDERLINE`, avec justification ;
- `CERTAIN_FALSE_REALISM`, avec justification.

Deux faux réalismes certains ou plus donnent `PAUSE_DOWNSTREAM`; zéro ou un
donne `PASS`. Ce verdict intervient après la production et ne peut arrêter ni
modifier le runner. Les cas borderline sont publiés séparément et ne sont pas
silencieusement assimilés à des erreurs certaines.

## Exécution post-production

La première invocation matérialise le seul échantillon déterministe et un
rapport `PENDING_BOUNDED_REALISM_REVIEW` :

```bash
python scripts/audit_synthetic_gt_balanced_final.py \
  --output /Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/\
synthetic_gt_corpus/balanced_v1/final_audit_sample_v1
```

L'auditeur renseigne un JSONL contenant le `schema_version`
`sireto-synthetic-gt-realism-review-1`, le `sample_id`, la `decision` et la
`reason`. La finalisation relit exactement le même registre et le même sel :

```bash
python scripts/audit_synthetic_gt_balanced_final.py \
  --realism-review /chemin/realism_review.jsonl \
  --output /Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/\
synthetic_gt_corpus/balanced_v1/final_audit_v1
```

Une ligne manquante, supplémentaire ou dupliquée dans la revue est refusée.
Le dossier final contient `report.json`, `realism_sample.jsonl`, la revue et
un manifeste de hashes non auto-référent.
