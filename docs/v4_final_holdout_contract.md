# Contrat final V4 — ouverture unique du holdout

## Objet

Évaluer une seule fois le retrieval, le ranker et l'accepteur V4 sur
`holdout_sealed`, après gel complet du code, des artefacts et des seuils.

Cette expérience ne sert plus à choisir une variante. Aucun résultat du
holdout ne pourra modifier la politique V4, le retrieval, les features, le
ranker, l'accepteur ou son seuil.

## Population

Le manifeste V4-Fresh publie avant ouverture :

- 1 345 requêtes source ;
- 302 `MATCH_EXACT` ;
- 52 `AMBIGUOUS` ;
- 991 `UNRESOLVED`.

Trois dénominateurs sont publiés sans substitution :

1. couverture identifiable source :
   `MATCH_EXACT / toutes les requêtes source` ;
2. part exacte du périmètre évalué :
   `MATCH_EXACT / (MATCH_EXACT + AMBIGUOUS)` ;
3. couverture `AUTO_MATCH` :
   `AUTO_MATCH / (MATCH_EXACT + AMBIGUOUS)`.

Les 302 `MATCH_EXACT` forment le périmètre Recall@100 et Hit@1. Les 52
`AMBIGUOUS` participent à l'évaluation de l'accepteur et toute décision
`AUTO_MATCH` les concernant compte comme une erreur. Les 991 `UNRESOLVED`
restent `REVIEW` et ne deviennent jamais des négatifs inventés.

## Artefacts gelés

Avant ouverture, un manifeste de gel doit publier les SHA-256 de :

- ce contrat et des scripts finaux ;
- manifeste V4-Fresh ;
- manifeste du gate retrieval V4 ;
- dataset et modèle du ranker V4 ;
- dataset, modèle, calibrateur et métadonnées de l'accepteur V4 ;
- feature orders candidat et scène ;
- seuil accepteur, fixé à `1.0`.

Le manifeste doit aussi reprendre les hashes déjà déclarés des deux fichiers
du holdout, sans les relire. Il doit confirmer la disjonction SIREN
fit/dev/holdout à partir du manifeste V4-Fresh.

## Ouverture unique

Le préparateur final :

1. vérifie l'autorisation et tous les artefacts gelés ;
2. crée un marqueur irréversible avant de lire le holdout ;
3. vérifie les hashes déclarés ;
4. copie uniquement les 302 exactes et 52 ambiguës dans un artefact
   d'évaluation immuable ;
5. conserve les 991 `UNRESOLVED` uniquement dans les comptes agrégés ;
6. ne lit jamais l'ancien test.

Pour les ambiguës, le premier SIRET direct est transporté comme sonde
technique exigée par l'auditeur de canaux. Cette sonde ne modifie ni le
retrieval, ni l'ordre candidat, ni les métriques Recall@100. Aucune métrique
de « recall ambigu » n'est publiée.

Un échec reproductible d'instrumentation peut être documenté, mais le premier
artefact et le marqueur restent conservés. Aucune seconde expérience ne peut
remplacer le résultat initial.

## Pipeline gelé

Pour les 354 scènes évaluées :

1. canaux actifs V7 et overlay à profondeur interne 5 000 ;
2. admission déterministe V4, 100 candidats maximum ;
3. 55 features candidates sans sémantique ;
4. ranker `ranker_1aebeada820d92a7_6236365` ;
5. 80 features de scène ;
6. accepteur `acceptor_2b8a9c994e0944be_9ec88c8` ;
7. `AUTO_MATCH` si et seulement si la confiance est supérieure ou égale au
   seuil gelé `1.0`, sinon `REVIEW`.

Il n'y a ni positif injecté, ni règle post-hoc, ni fallback web, ni GPU, ni
service payant.

## Métriques obligatoires

Publier les nombres bruts, taux et intervalles de Wilson à 95 % et 99 % :

- couverture identifiable source ;
- part exacte du périmètre évalué ;
- Recall@100 SIRET exact ;
- Hit@1 SIRET exact ;
- Hit@1 SIREN ;
- couverture `AUTO_MATCH` globale et sur les seules exactes ;
- précision SIRET exacte parmi les `AUTO_MATCH` ;
- nombre d'ambiguës automatisées ;
- nombre d'erreurs exactes automatisées ;
- nombre maximal de candidats et nombre de listes au-dessus de 100.

Toutes les erreurs sont exportées ligne à ligne. Une vérité absente du pool
est une erreur end-to-end pour le retrieval, le ranker et la décision.

## Gates et verdict

Les contrôles d'intégrité sont obligatoires :

- hashes et identifiants compatibles ;
- 302 exactes, 52 ambiguës et 991 unresolved ;
- zéro SIREN exact partagé avec fit ou dev ;
- zéro candidat injecté ou dupliqué ;
- 100 candidats au maximum ;
- toutes les scènes scorées ;
- ancien test absent.

Les gates de performance sont :

- couverture identifiable source ≥ 80,0 % ;
- Recall@100 SIRET exact ≥ 99,0 % ;
- Hit@1 SIRET exact ≥ 96,033 %  
  (pas plus de deux points sous les 98,033 % du dev) ;
- couverture `AUTO_MATCH` ≥ 25,0 % sur exactes + ambiguës ;
- précision SIRET exacte observée des AUTO ≥ 99,8 % ;
- au moins 25 décisions AUTO.

Verdict :

- `GO` si tous les contrôles et tous les gates passent ;
- `PIVOT` si les données sont valides mais qu'au moins un gate de performance
  échoue ;
- `STOP` si l'intégrité, l'indépendance ou la reproductibilité du test est
  compromise.

Un sous-verdict technique est également publié afin de distinguer un échec de
qualification d'un échec du matching :

- `TECHNICAL_GO` si retrieval, ranker et accepteur passent leurs gates ;
- `TECHNICAL_PIVOT` sinon.

Même avec zéro erreur AUTO, ce holdout ne permet pas de revendiquer une
garantie de 99,8 %. Il produit une estimation indépendante unique.
