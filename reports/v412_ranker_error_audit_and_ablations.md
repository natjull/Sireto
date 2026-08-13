# V4.12 — audit des erreurs ranker et ablations locales

Date : 13 août 2026  
Périmètre : développement déjà consommé uniquement ; test final fermé.

## Point de départ

Le ranker réentraîné sur les 279 REVIEW adjudiqués classait correctement
`216/254` SIRET exacts (`85,04 %`). Les 35 erreurs récupérables ont été
matérialisées dans `reports/v412_ranker_recoverable_errors_35.csv`, avec le
SIRET prédit, la vérité, les raisons sociales, enseignes, activités, statuts de
siège, adresses et rangs de retrieval des deux candidats.

L'audit métier montre quatre familles dominantes :

- opérateur réel confondu avec une holding, un propriétaire ou une structure
  support co-localisée ;
- enseigne opérationnelle confondue avec la raison sociale ;
- activité du site ou catégorie juridique incompatible avec le libellé CRM ;
- plusieurs établissements du même SIREN à la même adresse, dont le rôle
  exact n'était pas représenté dans les features.

L'audit qualité séparé a aussi démontré qu'une partie des erreurs provenait de
labels non identifiables localement ou temporellement invalides. Le canonique
n'est pas modifié : les corrections et exclusions sont documentées dans
`reports/v412_trusted_label_quality_overlay.csv`. Le périmètre strict contient
241 `MATCH_EXACT` localement identifiables.

## Ablations ranker

### Features métier injectées dans XGBoost

Le script `scripts/evaluate_v412_ranker_business_features.py` ajoute les noms
source SIRENE, catégorie juridique, activité, statut employeur/effectif, date
de début, cohérence de rôle et comparaisons entre entités co-localisées.

Sur les 254 labels historiques, la meilleure variante `targeted`, poids 0,5,
atteint `222/254`, mais les anciens contrôles lui attribuaient quatre
régressions. Un contre-audit indépendant a établi que ces quatre labels de
contrôle étaient faux. Sur le périmètre propre de 241 cas, le même type de
ranker plafonne à `219/241` (`90,87 %`) et ne constitue donc pas seul la bonne
architecture.

Artefacts principaux :

- historique :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_ranker_business_features/825f8266f658a093` ;
- labels locaux :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_ranker_business_features_local241/a8e21e7eb9e1c0cf`.

### Reranker XGBoost en deux étages

Le reranker métier appris sur le top 20 préserve `1127/1127` contrôles, mais
retombe à `215/254` sur les cas difficiles. Il est rejeté : ajouter un second
XGBoost n'apporte pas le raisonnement de rôle attendu.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_two_stage_business_reranker/d31b8fa119b28399`.

### Cross-encoders locaux, sans location de GPU

Deux modèles multilingues ont été exécutés localement sur le Mac, top 20 :

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` : meilleur mélange brut
  `222/254`, mais quatre corrections de labels de contrôle étaient initialement
  comptées comme régressions ;
- `BAAI/bge-reranker-v2-m3` : meilleur mélange brut `220/254`, avec un signal
  différent mais insuffisant seul.

Artefacts :

- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_cross_encoder_reranker/8d93c540ffcc3c04` ;
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_cross_encoder_reranker/d19079d68fc0940b`.

Leur usage pertinent est un signal complémentaire, pas un remplacement du
ranker. L'ensemble conservateur qui ne promeut qu'un rang 2 soutenu par des
signaux indépendants atteint `225/254` historiques et `219/241` stricts avant
règles de rôle, avec `1127/1127` contrôles corrigés.

## Accepteur enrichi

Le script `scripts/evaluate_v412_acceptor_business_competition.py` ajoute 24
features query-level de concurrence : autres SIREN à la même adresse, marges
nom légal/enseigne, catégorie juridique, activité et ancienneté. Son évaluation
est nested OOF par composante SIREN.

- ranker trusted historique : `113/216` bons top 1 acceptés (`52,31 %`), zéro
  erreur, contre `87/216` pour les features canoniques ;
- ensemble et labels locaux : `87/219` (`39,73 %`), zéro erreur.

Le gain de représentation est réel, mais le gate de 65 % n'est pas atteint.
Verdict : `PIVOT_FEATURES`, sans baisse opportuniste du seuil.

Artefacts :

- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_business_competition/ac5ccbd0fce134de` ;
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_business_competition/be2c96c72b46d761`.

## Conclusion d'architecture

Les expériences réfutent le simple « fine-tuning du XGBoost » comme réponse
suffisante. Le levier utile est une architecture modulaire : ranker principal,
signaux texte et métier complémentaires, puis règles de rôle générales et
traçables avant un accepteur query-level. Toutes les mesures ci-dessus sont du
développement consommé ; elles ne certifient ni 99,8 % ni la généralisation.
