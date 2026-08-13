# V4.12 — Accepteur avec concurrence métier explicite

Date : 13 août 2026  
Périmètre : développement déjà consommé uniquement. Aucun test final ouvert,
aucune promotion produit.

## Expérience

Le script `scripts/evaluate_v412_acceptor_business_competition.py` reconstruit
les scènes de l'accepteur depuis un classement candidat complet. Il accepte
aussi un classement partiel en overlay sur un classement OOF complet, ce qui
permet de rejouer immédiatement l'expérience sur un nouveau ranker.

Il ajoute 24 signaux query-level sans modifier le contrat canonique :

- preuves séparées du nom légal et de l'enseigne ;
- nombre de SIREN concurrents à la même adresse ;
- nombre de concurrents dont le nom légal ou l'enseigne correspond ;
- marge du top 1 sur le meilleur autre SIREN à la même adresse ;
- catégorie juridique et cohérence avec les mots du CRM ;
- cohérence activité NAF / rôle métier ;
- avantage de date de début parmi les entités co-localisées.

Les poids trusted `2`, `5` et `10` sont comparés par nested OOF. Pour chaque
fold externe, le seuil est calibré uniquement par OOF interne sur les quatre
autres folds de composantes SIREN. Le dossier externe n'est donc utilisé ni
pour entraîner le modèle, ni pour choisir son seuil.

## Résultat sur le ranker trusted courant

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_business_competition/ac5ccbd0fce134de`

| Accepteur | Bons top1 AUTO | Faux AUTO | Acceptation des bons top1 |
|---|---:|---:|---:|
| 80 features canoniques, poids 10 | 87 / 216 | 0 | 40,28 % |
| 80 + 24 features métier, poids 10 | **113 / 216** | **0** | **52,31 %** |

Le gain honnête nested OOF est de **26 décisions fiables**, soit **+12,03
points** d'acceptation des top1 corrects. Une calibration unique sur tout l'OOF
consommé accepterait 125/216 sans erreur observée, mais cette valeur n'est pas
retenue comme résultat principal car son seuil utilise tout le développement.

Verdict : `PIVOT_FEATURES`. Le gain est réel, mais le gate 65 % n'est pas
atteint.

## Rejeu sur l'ensemble ranker et les labels localement identifiables

Ranked overlay :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f/ranked_candidates.parquet`

Artefact accepteur :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_business_competition/be2c96c72b46d761`

L'overlay qualité contient 241 `MATCH_EXACT`, 31 `AMBIGUOUS` et 7
`UNRESOLVED`. Les sept `UNRESOLVED` et les six dossiers devenus ambigus sont
exclus de la cible positive ; la correction de `10613` est incluse.

| Accepteur | Bons top1 AUTO | Faux AUTO | Acceptation des bons top1 |
|---|---:|---:|---:|
| 80 features canoniques, poids 10 | 75 / 219 | 1 ambigu | 34,25 % |
| 80 + 24 features métier, poids 10 | **87 / 219** | **0** | **39,73 %** |

L'ensemble améliore le Hit@1 du ranker, mais rend ses scores moins directement
calibrables par l'accepteur actuel. Les features métier suppriment le faux AUTO
et récupèrent douze décisions, sans atteindre le gate.

## Conclusion

Les informations manquantes identifiées par l'audit améliorent bien
l'accepteur ; ce n'était donc pas seulement un problème de seuil. Elles ne
suffisent toutefois pas à accepter 65–75 % des top1 corrects avec zéro erreur
observée.

La prochaine relance utile est le même script sur le ranked role-aware corrigé,
sans nouveau tuning d'architecture :

```bash
python scripts/evaluate_v412_acceptor_business_competition.py \
  --trusted-labels reports/v412_review_local_identifiable_labels_279.csv \
  --ranked-candidates /chemin/vers/nouveau_ranked_candidates.parquet \
  --base-ranked-candidates /chemin/vers/ranked_oof_complet.parquet
```

Le gate reste : au moins 65 % des top1 corrects en `AUTO_MATCH`, zéro négatif
ou ambigu automatique sur ce développement de petite taille, puis validation
sur une population indépendante avant toute revendication à 99,8 %.
