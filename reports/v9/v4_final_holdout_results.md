# Évaluation finale V4 — résultats

## Verdict

**`PIVOT`**

Sous-verdict matching : **`TECHNICAL_PIVOT`**.

Le retrieval atteint l'objectif de 99 % avec une marge nette. Le ranker
franchit également son gate. L'accepteur échoue en revanche à la précision
visée, et la qualification stricte n'identifie qu'une fraction insuffisante
des données source.

| Étape | Résultat final | Gate | Verdict |
|---|---:|---:|---|
| Couverture identifiable source | 302/1 345 = 22,454 % | ≥ 80 % | FAIL |
| Recall@100 SIRET exact | 302/302 = 100 % | ≥ 99 % | PASS |
| Hit@1 SIRET exact | 296/302 = 98,013 % | ≥ 96,033 % | PASS |
| Hit@1 SIREN | 296/302 = 98,013 % | publication | — |
| Couverture AUTO | 282/354 = 79,661 % | ≥ 25 % | PASS |
| Précision exacte des AUTO | 280/282 = 99,291 % | ≥ 99,8 % | FAIL |

Le plafond absolu de 100 candidats est respecté. Les 354 scènes sont scorées,
aucun positif n'est injecté, aucun SIREN exact n'est partagé avec fit/dev et
l'ancien test n'est pas lu.

## Conclusion architecture

Le résultat invalide une refonte lourde du retrieval :

- la recherche gelée retrouve **302/302** vérités exactes ;
- le TF-IDF enrichi par quelques canaux déterministes suffit ;
- le ranker produit **296/302** bonnes premières réponses, presque exactement
  son résultat dev de 98,033 % ;
- le problème restant se situe dans le périmètre admissible et dans la
  sécurité de l'automatisation.

La bonne suite n'est donc ni du dense, ni un cross-encoder, ni davantage de
candidats. C'est une correction d'architecture courte en amont et en aval du
ranker.

## Les deux erreurs AUTO

### 1. `fresh:AC010162` — PALAFIS

- CRM : `PALAFIS`, `30 Rte des Creusettes`, Poisy ;
- vérité active : `88517228800023` ;
- prédiction AUTO : `43250243300027` ;
- le SIRET prédit est **fermé** ;
- les deux établissements ont le même nom, le même numéro, la même voie, le
  même code postal et la même commune ;
- le candidat fermé obtient des similarités nom et adresse parfaites et un
  score ranker de 5,057, contre 1,847 pour le SIRET actif.

Le statut administratif n'appartient pas aux 55 features du ranker et aucun
verrou n'interdit actuellement d'automatiser un candidat fermé, alors que la
vérité V4 cible explicitement un établissement actif.

### 2. `fresh:FR012204` — ELGEA

- CRM : `ELGEA`, `1 impasse de la Ferme de Varâtre`, Lieusaint ;
- qualification V4 : `AMBIGUOUS` ;
- **80 SIRET actifs** satisfont la preuve directe disponible ;
- l'accepteur automatise `51085463100049` ;
- l'écart ranker top1/top2 n'est que de 0,053, le plus faible de toutes les
  décisions AUTO finales.

Cette scène était déjà connue comme non saine avant le retrieval. Elle
n'aurait jamais dû dépendre d'un apprentissage probabiliste pour être envoyée
en revue.

## Défaut de calibration

L'accepteur retenu transforme **282 scènes différentes** en une confiance
exactement égale à `1.0`, qui est aussi le seuil AUTO. Cette saturation détruit
une partie de l'ordre de risque :

- probabilité brute logistique PALAFIS : 0,474 ;
- probabilité brute logistique ELGEA : 0,626 ;
- confiance isotonic des deux cas : 1,0.

Les six variantes étaient à égalité parfaite sur le dev. Le choix isotonic
venait uniquement du tie-break déterministe ; il n'était pas soutenu par une
supériorité observée.

## Pivot recommandé

Architecture V4.1 à préenregistrer :

```text
qualification indépendante
  ├─ AMBIGUOUS ou UNRESOLVED → REVIEW
  └─ MATCH_EXACT
       → retrieval V4 inchangé, max 100
       → ranker V4 avec état administratif explicite
       → top1 fermé → REVIEW
       → accepteur sans calibration saturante
       → AUTO_MATCH ou REVIEW
```

Les deux règles structurantes sont :

1. un label mécanique `AMBIGUOUS` ou `UNRESOLVED` entraîne directement
   `REVIEW` ;
2. dans la cible V4 « SIRET actif courant », un top1 fermé ne peut pas devenir
   `AUTO_MATCH`.

Appliquées rétrospectivement au holdout, ces deux règles retireraient
exactement les deux erreurs AUTO et laisseraient 280 AUTO sur 354. Ce chiffre
est seulement un diagnostic de cause : il ne constitue pas une validation,
car les règles ont été formulées après lecture du holdout.

## Problème distinct de couverture

La politique V4 stricte identifie 302 cas exacts sur 1 345 requêtes source,
soit 22,454 %. Même si les exactes représentent 85,311 % des 354 scènes
évaluables exactes ou ambiguës, cela ne satisfait pas le contrat de couverture
source à 80 %.

Il faut traiter cette question séparément du matching :

- améliorer ou réparer les champs CRM ;
- versionner des alias et relations historiques fiables ;
- élargir la preuve de qualification sans utiliser un hit, un rang ou un
  score modèle ;
- garder `UNRESOLVED` en revue tant qu'aucune preuve indépendante n'existe.

Assouplir simplement la qualification sur ce holdout serait une fuite et ne
serait pas une preuve.

## Statistique

- Recall@100, 302/302 : borne Wilson basse 95 % = 98,744 %, basse 99 % =
  97,850 % ;
- Hit@1, 296/302 : borne basse 95 % = 95,734 % ;
- précision AUTO, 280/282 : borne basse 95 % = 97,451 %, basse 99 % =
  96,454 %.

Le résultat observé ne permet aucune garantie de 99,8 % en production.

## Traçabilité

- contrat final : commit `fb6a20c` ;
- runner gelé : commit `8cc9bfa` ;
- autorisation : `7dbd5527374ca0d4` ;
- première évaluation conservée :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4/` ;
- correction instrumentale du verdict : commit `aead6f5` ;
- rapport corrigé :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4_verdict_repair/`.

Le premier runner avait inversé deux booléens d'intégrité :
`old_test_read=false` et `positive_injection=false`. Les métriques brutes
n'ont pas changé ; aucun modèle n'a été recalculé, le holdout n'a pas été relu
et le premier rapport reste conservé.
