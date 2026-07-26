# Contrat V4-Fresh — expansion indépendante du benchmark

## Objet

Le CRM source `data/entrainements.csv` contient 6 330 identifiants de service
absents du benchmark fermé V9. V4-Fresh utilise exclusivement ces lignes pour :

1. compléter le noyau d'apprentissage V4 strict ;
2. créer un nouveau dev ;
3. créer un holdout scellé qui ne sera pas lu par les entraînements.

La qualification réutilise sans modification la politique
`active-direct-current-v4.0`.

## Définition d'une ligne fraîche

Une ligne est fraîche si son `SERVICE ID` :

- est non vide ;
- est unique dans le pool fresh ;
- n'apparaît dans aucun `crm_record_id` du benchmark fermé
  `c33b80855f560074`.

Les 14 SIRET historiques invalides ou absents restent acceptables en entrée :
le SIRET V4 est déterminé par la preuve active nom–adresse, pas par la valeur
historique.

## Interdictions

- aucun split test historique ;
- aucun résultat E1/E2/E2b ;
- aucun rang, hit, score ou décision modèle ;
- aucun assouplissement de la règle V4 ;
- aucun `UNRESOLVED` transformé en négatif ;
- aucune ouverture du holdout fresh par un entraînement.

## Qualification

Chaque ligne fresh est convertie au schéma CRM canonique puis examinée contre
tous les établissements actifs de sa partition géographique. Les labels sont
ceux de V4 :

- un candidat actif direct unique : `MATCH_EXACT` ;
- plusieurs candidats actifs directs : `AMBIGUOUS` ;
- aucun : `UNRESOLVED`.

## Séparation gelée

La clé de groupe est :

- le SIREN V4 pour `MATCH_EXACT` ;
- sinon le SIREN historique valide ;
- sinon le `SERVICE ID`.

Toute clé déjà présente dans le noyau V4 historique est forcée dans
`fit_addition`. Pour les nouvelles clés, le premier octet de
`SHA-256("42:" + clé)` détermine le rôle :

- `0..127` : `fit_addition` ;
- `128..191` : `dev_new` ;
- `192..255` : `holdout_sealed`.

Une même clé ne peut apparaître dans plusieurs rôles. Les SIREN exacts doivent
être totalement disjoints entre le fit combiné, le nouveau dev et le holdout.

## Gate

V4-Fresh passe si toutes les conditions suivantes sont vraies :

- noyau V4 historique + `fit_addition` contient au moins 5 000
  `MATCH_EXACT` ;
- `dev_new` contient au moins 300 `MATCH_EXACT` ;
- `holdout_sealed` contient au moins 300 `MATCH_EXACT` ;
- zéro SIREN exact partagé entre les trois rôles ;
- zéro identifiant de service déjà présent dans le benchmark ;
- zéro SIRET fermé parmi les nouveaux `MATCH_EXACT` ;
- suite complète de tests verte.

Le holdout peut être qualifié et hashé, mais son contenu ligne à ligne et ses
métriques modèle restent fermés. Le manifeste peut publier uniquement ses
volumes de labels et ses hashes avant l'autorisation finale.

## Suite autorisée

Si le gate passe :

1. reconstruire le dataset candidat sur le fit combiné et `dev_new` ;
2. exclure tous les `UNRESOLVED` du fit accepteur ;
3. entraîner ranker et accepteur avec prédictions OOF groupées par SIREN ;
4. geler bundle et seuil ;
5. ouvrir une seule fois `holdout_sealed`.

Ce holdout mécanique ne suffit pas, à lui seul, à garantir statistiquement
99,8 %. Une validation indépendante des décisions AUTO restera nécessaire.
