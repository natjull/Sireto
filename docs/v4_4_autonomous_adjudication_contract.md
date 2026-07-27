# Contrat V4.4 — adjudication autonome fondée sur les preuves

Statut : préenregistré avant collecte des nouvelles preuves publiques.

## Décision de fonctionnement

L'utilisateur n'est pas le validateur et aucune tâche d'analyse ligne par ligne
ne lui est déléguée. Codex mène l'adjudication, conserve les preuves et exclut
les cas sur lesquels il ne peut pas conclure.

Cette méthode permet de construire un corpus de travail défendable. Elle ne
transforme pas une validation IA en certification humaine de précision.

## Population

La première population contient les 172 décisions `AUTO_MATCH` non résolues
de la file V4.3 :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_3_hard_labels/0f832305ab199267/hard_label_queue.parquet`

Les 370 `REVIEW` sont conservés pour une seconde passe si le gate AUTO produit
assez de labels fiables.

## Sources autorisées

- snapshot SIRENE complet déjà épinglé ;
- API Recherche d'entreprises officielle ;
- Annuaire des entreprises et autres sites publics officiels ;
- site officiel de l'entité lorsqu'il associe explicitement nom, adresse et
  identité ;
- historique local du SIREN et des établissements ;
- documents publics datés dont l'URL est conservée.

Le score, le rang, le SIRET CRM historique ou une simple adresse commune ne
constituent jamais une preuve de vérité.

## Règle d'adjudication

Chaque cas reçoit l'un des statuts suivants :

- `TOP1_CORRECT` ;
- `TOP1_WRONG` ;
- `AMBIGUOUS` ;
- `UNRESOLVED`.

`evidence_validated=true` exige :

1. au moins deux preuves non issues du modèle ;
2. une cohérence explicite sur l'identité, pas seulement l'adresse ;
3. aucune contradiction non résolue entre les sources ;
4. les références et la date de collecte dans l'artefact.

Une preuve peut être une concordance SIRENE forte complétée par une source
publique indépendante. Deux variantes du même enregistrement SIRENE ne
comptent pas comme deux sources.

Les jugements sémantiques de Codex peuvent expliquer une décision, mais ne
remplacent pas les deux preuves.

## Interdictions

- demander à l'utilisateur de valider les lignes ;
- utiliser un moteur payant ou louer un GPU ;
- modifier le retrieval, le ranker, l'accepteur ou le seuil ;
- traiter une absence de résultat web comme une preuve négative ;
- fabriquer un SIRET exact lorsque seul `TOP1_WRONG` est démontré ;
- entraîner sur `UNRESOLVED` ou sur une décision sans références.

## Gate

`GO_RETRAIN_AUTO` si la passe AUTO produit :

- au moins 75 `TOP1_CORRECT` avec `evidence_validated=true` ;
- au moins 50 `TOP1_WRONG` avec `evidence_validated=true` ;
- au moins 30 cas validés issus du tirage aléatoire ;
- zéro décision fondée uniquement sur le modèle ou l'adresse.

`PIVOT_MORE_EVIDENCE` si les preuves sont fiables mais trop peu nombreuses.

`STOP_AUTONOMOUS_LABELING` si les sources publiques ne permettent pas une
adjudication suffisamment sûre sans humain.

Ce gate autorise un entraînement expérimental, jamais une revendication de
certification humaine ou un déploiement.

## Livrables

- cache immuable des réponses officielles ;
- table de preuves par cas et source ;
- adjudications avec raisonnement et références ;
- synthèse des cas retenus et exclus ;
- rapport et handover avec verdict explicite.
