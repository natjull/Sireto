# V4.11 — Développement de l'accepteur compact

## Verdict

**`GO_FREEZE_V411_CANDIDATE`**

Le candidat retenu est `COMPACT_LOGIT`, au seuil
`0.8720916706888049`.

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/9d23bf3deb6b63de`

Ce verdict autorise le gel du bundle et le challenge descriptif des 225 cas
inédits. Il n'autorise ni déploiement, ni certification à 99,8 %, ni
réouverture d'un test final historique.

## Résultats sur `comparison_dev`

| Système | AUTO | Corrects AUTO | Erreurs AUTO | Précision observée | Couverture |
|---|---:|---:|---:|---:|---:|
| Baseline V4.1 | 618 / 746 | 617 | 1 | 99,838 % | 82,842 % |
| `COMPACT_LOGIT` | 614 / 746 | 614 | 0 | 100,000 % | 82,306 % |
| `MONOTONIC_XGB` | 612 / 746 | 612 | 0 | 100,000 % | 82,038 % |

Les deux candidats franchissent les gates préenregistrés. La logistique gagne
par sa couverture supérieure. Face à la baseline, elle accepte deux dossiers
auparavant en REVIEW et refuse six anciens AUTO : le solde est de quatre AUTO
en moins, trois bons en moins et l'unique erreur baseline supprimée.

Les 746 scènes contiennent 634 `MATCH_EXACT` et 112 `AMBIGUOUS`. Le gagnant :

- automatise 614 SIRET exacts corrects ;
- place 20 SIRET exacts en REVIEW ;
- place les 112 cas ambigus en REVIEW.

Toutes les familles critiques passent les contraintes de précision et de
couverture.

## Seuil

Le seuil a été choisi uniquement sur les 710 scènes de `threshold_dev` :

| Modèle | Seuil | AUTO | Corrects | Erreurs | Précision | Couverture |
|---|---:|---:|---:|---:|---:|---:|
| `COMPACT_LOGIT` | 0,872092 | 564 | 563 | 1 | 99,823 % | 79,437 % |
| `MONOTONIC_XGB` | 0,913617 | 568 | 567 | 1 | 99,824 % | 80,000 % |

Pour la logistique, l'erreur est un cas `AMBIGUOUS` automatisé. Ce n'est pas
une violation du contrat : le choix du seuil impose 99,8 % sur
`threshold_dev`, tandis que le gate de 80 % et zéro ambigu est appliqué
ensuite sur `comparison_dev`. C'est néanmoins un signal de fragilité qui
interdit de présenter le zéro erreur du second lot comme une stabilité
acquise.

## Limites statistiques

`614/614` signifie 614 décisions AUTO correctes, pas 614 succès sur les 746
requêtes. La borne basse Wilson bilatérale à 95 % est seulement 99,378 % ; la
borne exacte unilatérale à 99 % est d'environ 99,253 %. Le résultat observé
ne démontre donc pas encore une précision réelle de 99,8 %.

De plus, le ranker C était déjà correct sur les 634 `MATCH_EXACT` de
`comparison_dev`. Ce lot mesure surtout la capacité à séparer les cas exacts
des ambiguïtés ; il éprouve peu l'accepteur face à un mauvais top-1. Les cinq
erreurs OOF du fit et l'erreur de `threshold_dev` restent présentes dans les
données, mais pas dans le lot de comparaison.

Enfin, `comparison_dev` choisit entre deux familles et appartient à un dev
historique déjà consommé. Il ne constitue pas une preuve indépendante de
production.

## Intégrité

- plan gelé : commit `8033934`, hash
  `49299bd98f350abb90f159915a5991d88af25a307ab73b951826beb49cc571b4` ;
- verrou d'exécution : commit `fd70a64`, hash
  `17fe2fdb93522a12459358a36e21f5ed88434df38068d0eabcd3aa358a9c2e4e` ;
- 5 547 scènes fit, 710 scènes seuil et 746 scènes comparaison ;
- aucun chevauchement de requête ou composante entre les populations ;
- scores des deux répétitions identiques bit à bit ;
- modèles et seuils recomputés indépendamment à l'identique ;
- bundle gagnant reproduit exactement les 614 décisions AUTO ;
- retrieval, ranker C, scènes, taxonomie et sources épinglés par hash ;
- aucun challenge, holdout, unseen ou test final ouvert pendant ce run ;
- deux contre-audits indépendants concluent à l'absence de blocker.

## Étape suivante autorisée

1. publier les hashes du candidat gelé ;
2. ouvrir une seule fois les 225 entrées inédites ;
3. les qualifier sans SIRET CRM, sans retrieval et sans score ;
4. geler les labels avec preuves traçables ;
5. exécuter le stack gelé une seule fois ;
6. publier le résultat comme `DESCRIPTIVE_UNSEEN_225`, jamais comme preuve
   représentative.

La preuve finale de la North Star exige toujours un nouvel export CRM
indépendant. Environ 2 300 décisions AUTO sans erreur seraient nécessaires
pour soutenir une borne unilatérale à 99 % au-dessus de 99,8 %.
