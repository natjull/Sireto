# Contrat — Matching aval sélectif SIRET

## Objet

Le retrieval sélectif est désormais une entrée gelée : pour chaque requête
admise, il fournit au plus 100 SIRET. Cette phase doit :

1. classer le bon SIRET le plus haut possible ;
2. accepter automatiquement uniquement les premiers choix suffisamment sûrs ;
3. envoyer tous les autres dossiers en `REVIEW`.

La métrique produit reste le **SIRET exact**. Un bon SIREN mais un mauvais
établissement est une erreur.

## Périmètre autorisé

- calcul sur le Mac M4 Pro et `/Volumes/CATNAT_DATA` uniquement ;
- XGBoost et modèles linéaires locaux ;
- retrieval, qualification V3 et admission à 100 gelés ;
- train et dev actuels utilisables pour construire et choisir les modèles ;
- nouveau holdout indépendant obligatoire pour l'évaluation finale ;
- aucun accès au test sélectif consommé pour développer, comparer ou choisir
  une variante ;
- aucun GPU loué, LLM, Places ou source web dans le chemin d'inférence.

Les anciens `ranker`, `decider` et `risk model` sont des baselines en lecture
seule. Ils ne sont jamais désignés implicitement par « latest ».

## Architecture gelée pour l'expérience

```text
CRM
  → retrieval sélectif gelé, 0 à 100 SIRET uniques
  → features candidat canoniques
  → ranker final SIRET
  → top-1/top-2 + preuves candidat + informations de scène
  → accepteur exact-SIRET
  → AUTO_MATCH ou REVIEW
```

Le ranker répond à « quel SIRET choisir ? ». L'accepteur répond à « peut-on
automatiser ce choix ? ». Ces deux tâches restent séparées.

## Données

Le dataset aval doit être construit à partir des sorties réelles du retrieval
gelé :

- 100 candidats maximum par requête, après déduplication exacte du SIRET ;
- aucune injection du positif dans les scènes, métriques ou prédictions OOF ;
- une vérité absente du pool reste une erreur end-to-end ;
- les requêtes sans candidat produisent une scène explicite ;
- `AMBIGUOUS` et `UNRESOLVED` restent dans les données de l'accepteur avec
  cible AUTO incorrecte ;
- seuls les `MATCH_EXACT` dont le positif est réellement présent peuvent
  entraîner le ranker ;
- séparation par SIREN pour train/dev et pour chaque fold OOF ;
- manifeste obligatoire : hashes des entrées, snapshot, retrieval, ordre des
  features, seed, code et volumes.

Une injection du positif est interdite dans cette première expérience, y
compris pour le fit ranker. Elle ne pourra faire l'objet que d'une ablation
ultérieure explicitement isolée.

## Features

### Ranker

Première expérience :

- 44 features déterministes V7, c'est-à-dire les 47 features baseline sans les
  trois features sémantiques ;
- 11 features de provenance et d'agrégats retrieval/SIREN ;
- exclusion des 7 features expérimentales V8.

Une seconde ablation pourra ajouter les trois features sémantiques réparées,
calculées par la même fonction au train et à l'inférence. Elle n'est pas
requise pour obtenir une première réponse architecturale.

### Accepteur

L'accepteur reçoit :

- les 20 features de scène V9 ;
- pour chacune des 20 preuves candidates historiques auditées : valeur top-1,
  valeur top-2 et différence top-1 moins top-2 ;
- la cible `is_exact_siret_correct`.

Les preuves couvrent notamment le nom, l'adresse, le numéro de voie, le code
postal, la commune et la concurrence sémantique. Elles doivent être transportées
avec les prédictions du ranker et calculées par le même code au train et au
service.

## Expériences courtes

### E1 — Ranker final

Comparer sur dev, sur les mêmes pools :

1. ordre brut du retrieval gelé ;
2. ancien ranker explicitement épinglé, lorsque son contrat de features est
   compatible ;
3. ancien decider explicitement épinglé comme score candidat ;
4. nouveau ranker final.

Publier Recall@100, Hit@1 SIRET, Hit@1 SIREN, nombres bruts, résultats
conditionnels au hit retrieval et résultats end-to-end. Le nouveau ranker est
retenu s'il a le meilleur Hit@1 SIRET dev sans dégrader une famille critique
de plus de 2 points.

### E2 — Accepteur exact-SIRET

Construire les scènes train uniquement à partir des prédictions OOF du ranker
retenu. Comparer régression logistique et XGBoost. La calibration et le choix
du seuil utilisent deux moitiés disjointes de dev.

Publier les courbes de couverture aux seuils observés de précision SIRET
99,0 %, 99,5 % et 99,8 %. Le point principal est la couverture maximale à
99,8 % de précision observée ; ce chiffre n'est pas présenté comme une garantie
statistique.

## Gates avant nouveau holdout

Les conditions train/dev sont toutes obligatoires :

- aucun SIREN partagé entre fit et validation OOF ;
- aucun doublon `(query_id, candidate_siret)` ;
- aucun pool de plus de 100 candidats ;
- aucune requête silencieusement supprimée des métriques ;
- Hit@1 et AUTO mesurés en SIRET exact ;
- modèle, ordre des features et seuil désignés par un bundle explicite ;
- train et inférence produisent exactement les mêmes features sur un fixture ;
- aucun choix de modèle ou de seuil basé sur le test consommé ;
- aucune régression segmentaire supérieure à 2 points pour le ranker retenu ;
- courbe risque-couverture publiée avec nombres bruts.

Si E1 n'améliore pas le meilleur scorer historique compatible, verdict
`PIVOT_RANKER`. Si E1 gagne mais qu'aucun accepteur n'atteint 99,8 % observé
avec au moins 25 % de couverture sur dev, verdict `PIVOT_ACCEPTEUR`.

## Évaluation finale

Le test sélectif actuel reste fermé. Avant toute évaluation finale :

1. constituer et geler un nouveau holdout indépendant ;
2. publier son hash et vérifier la disjonction SIREN ;
3. geler le bundle ranker, l'accepteur et le seuil ;
4. exécuter une seule évaluation ;
5. publier couverture, précision exacte, erreurs, intervalles de confiance et
   segments ;
6. conclure `GO`, `PIVOT` ou `STOP`.

Un `GO` nécessite :

- aucune baisse de précision exacte face à la baseline reproductible ;
- aucune régression segmentaire de plus de 2 points ;
- aucune violation de contrat ou de budget ;
- suffisamment de cas AUTO audités pour qualifier correctement la portée
  statistique du résultat.
