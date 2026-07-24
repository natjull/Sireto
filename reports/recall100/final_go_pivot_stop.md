# Retrieval SIRET Recall@100 — décision GO / PIVOT / STOP

## Décision

**PIVOT.**

La cible de 99,0 % n'est pas atteinte par une admission déterministe limitée à
100 candidats. La meilleure configuration observée sur le dev atteint
2 495 / 2 565, soit **97,27 %**. Le gate dev échoue de 45 requêtes et aucune
nouvelle variante n'a été exécutée sur le test.

Ce résultat n'est pas un `STOP`, car les sources et canaux audités voient le
bon SIRET pour 2 558 / 2 565 requêtes, soit **99,73 %**, lorsqu'ils peuvent
inspecter jusqu'à 5 000 résultats par canal. Le signal existe, mais la fusion
déterministe testée ne sait pas le comprimer en 100 candidats.

## Contrat expérimental

- benchmark gelé : `c33b80855f560074` ;
- split évalué : dev, 2 565 requêtes ;
- métrique principale : Recall candidat au SIRET exact ;
- plafond de sortie : 100 candidats ;
- aucun positif injecté ;
- ranker, decider, risk model et accepteur historiques inchangés ;
- calcul : Mac M4 Pro et SSD `/Volumes/CATNAT_DATA` ;
- aucun GPU loué, cloud ou API payante.

## Résultats

| Variante | Succès | Recall SIRET | Statut |
|---|---:|---:|---|
| Sparse gelé @100 | 2 379 / 2 565 | 92,75 % | baseline |
| Admission déterministe @100 | 2 495 / 2 565 | 97,27 % | gate échoué |
| Oracle des canaux internes @5 000 | 2 558 / 2 565 | 99,73 % | diagnostic non déployable |

L'admission respecte le plafond : maximum 100 candidats, moyenne 99,94 et
zéro requête au-dessus de 100.

L'intervalle de Wilson à 95 % de l'admission observée est
[96,57 % ; 97,83 %]. Il ne contient pas la cible de 99 %. L'oracle interne
n'est pas une sortie à 100 : c'est l'union diagnostique de listes allant
jusqu'à 5 000 résultats par canal.

## Attribution des pertes

Sur les 70 erreurs de l'admission :

- 7 vérités ne sont vues par aucun canal interne jusqu'à 5 000 ;
- 63 vérités sont bien trouvées en amont, puis éliminées lors de la réduction
  à 100.

Les sept requêtes non vues sont `218`, `724`, `725`, `2021`, `9079`, `11369`
et `16995`.

Le défaut dominant est donc désormais **l'admission**, pas l'absence du SIRET
dans les sources.

## Segments

| Segment | Sparse gelé @100 | Admission @100 | Delta |
|---|---:|---:|---:|
| Actifs | 96,34 % | 98,17 % | +1,83 pt |
| Fermés | 77,41 % | 93,43 % | +16,02 pt |
| Mégapoles | 86,06 % | 95,76 % | +9,70 pt |
| Multi-sites | 91,53 % | 97,23 % | +5,70 pt |
| CP seul | 94,87 % | 100,00 % | +5,13 pt |

Aucun segment critique ne régresse, mais aucun de ces gains ne compense
l'échec du gate global.

## Ce qui a été appris

1. Le sparse historique n'est pas seulement du TF-IDF : sa fusion précoce
   masque des candidats vus séparément par les canaux mots et caractères.
2. Le store V7 excluait 62 vérités fermées à cause du filtre legacy
   `dateDebut >= 2016-01-01`. Un overlay immuable de tous les fermés exclus a
   été construit sans lire les labels ; il retrouve les 62 vérités du dev.
3. La recherche au niveau SIREN puis l'ordonnancement de ses établissements
   apporte un signal complémentaire important sur les cas multi-sites.
4. Des quotas et une fusion RRF pondérée améliorent fortement la baseline,
   mais ne distinguent pas assez bien le bon SIRET parmi un très grand pool.
5. Augmenter seulement le nombre de candidats internes rend la cible visible,
   au prix d'environ 1,3 seconde de p95 cumulée pour les deux sources et de
   centaines de mégaoctets d'artefacts diagnostiques. Ce n'est pas encore une
   architecture de production.

## Pivot recommandé

Le prochain contrat doit tester une **admission apprise dédiée au retrieval** :

```text
canaux haute couverture
  → pool interne large et adaptatif
  → score d'admission candidat
  → top 100 strict
  → ranker/decider historiques, toujours gelés pendant l'ablation
```

Cette tête d'admission doit être distincte du ranker métier aval, être entraînée
uniquement sur train, puis gelée avant lecture du dev. Elle peut exploiter les
rangs par canal, leur accord, les égalités nom/adresse, la géographie, l'état,
les agrégats SIREN et la concurrence intra-SIREN. Sa seule cible est la présence
du vrai SIRET dans les 100 candidats, pas le Hit@1.

Ce pivot requiert un nouveau contrat : le contrat actuel impose un retrieval
évalué indépendamment d'un ranker. Il ne doit pas être introduit
silencieusement comme une optimisation de la fusion existante.

## Artefacts reproductibles

- résultat final dev :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f` ;
- audit V7 à K interne 5 000 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_dev_c33b80855f560074_d4255de` ;
- audit overlay à K interne 5 000 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_dev_c33b80855f560074_d4255de` ;
- store overlay :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/stores/legacy_closed_overlay_c33b80855f560074_e39fddd`.

Les volumes correspondants sont environ 2,6 Mo, 421 Mo, 332 Mo et 872 Mo.
