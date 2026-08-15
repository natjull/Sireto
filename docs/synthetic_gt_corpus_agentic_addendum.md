# Avenant agentique — corpus GT synthétique SIRETO

Date de gel : 15 août 2026
Version : `agentic-text-generation-v2`
Statut : obligatoire avant toute génération de contenu CRM.

## Décision normative

Le contenu des champs `crm_name`, `crm_address`, `crm_postcode`, `crm_city`
et `crm_insee` synthétiques est rédigé directement par Luna dans une boucle
agentique. Il est interdit à tout code du dépôt de choisir ou d'appliquer des
fautes, suppressions, permutations, abréviations, templates, variantes OCR,
formes juridiques ou autres corruptions. Aucun `faker`, moteur de bruit,
combinaison probabiliste ou générateur stochastique n'est autorisé.

Le code est autorisé uniquement à :

- lire les sources et calculer les profils de phénomènes observés dans le
  train `crm_ok_gt` ;
- sélectionner et assigner les seeds par lots disjoints, sans fuite SIREN ;
- produire des fiches officielles et des demandes de génération ;
- réceptionner et valider le JSON de Luna ;
- appliquer les contrôles d'identité, de fuite, de diversité excessive, de
  provenance, de sibling et de localité ;
- sélectionner des hard negatives existants dans SIRENE ;
- écrire les réponses acceptées, les décisions, les ledgers, checkpoints,
  hashes et statistiques.

Le code ne peut jamais modifier un champ CRM fourni par Luna pour le rendre
valide. Une réponse invalide est rejetée ou mise en `SILVER`; elle n'est pas
réparée automatiquement.

L'unique entrypoint autorisé est
`scripts/run_synthetic_gt_agentic_loop.py`. Il utilise SQLite en mode WAL,
des leases expirables et un journal d'événements append-only. Les anciens
prototypes ne constituent pas une seconde boucle concurrente ; le nom
historique `build_synthetic_gt_corpus.py` est uniquement un shim vers cet
entrypoint.

## Boucle obligatoire

Pour chaque seed :

1. `GENERATOR` reçoit la fiche officielle SIRENE/RNE, le résumé des
   phénomènes réellement mesurés dans le train et les contraintes de
   l'identifiant cible. Il répond avec exactement trois variantes rédigées,
   leurs familles observées et un résumé de transformation.
2. `CRITIC` reçoit une nouvelle vue indépendante de la fiche officielle et
   des variantes brutes. La justification du `GENERATOR` est absente de son
   entrée. Il vérifie séparément réalisme CRM, diversité non triviale,
   conservation du SIRET/établissement, siège versus établissement,
   invention gratuite et meilleure correspondance sibling. Il rend directement
   `ACCEPTED`, `SILVER` ou `REJECTED` ; cette décision est l'adjudication
   agentique de la ligne.
3. Le `SUPERVISOR` parent reçoit les décisions CRITIC et les contrôles
   déterministes en lecture seule. Il ne rédige aucun volume et ne peut
   qu'accepter une ligne déjà `ACCEPTED`, ou durcir vers `SILVER`/`REJECTED`.

Le pilote doit comporter un petit lot complet, avec trois réponses par seed,
les trois décisions, les digests des entrées/sorties, les versions de prompts
et une mesure du temps/tokens par rôle. Le corpus final ne contient jamais les
réponses `SILVER` ou `REJECT`.

## Preuve d'origine

Chaque ligne publiée conserve `agent_response_sha256`, `batch_id`,
`prompt_version_generator`, `prompt_version_critic`, `generator_decision`,
`critic_decision`, `supervisor_decision`, `corruption_families_observed` et
`transformation_summary`. Un audit statique doit confirmer l'absence de
fonction de génération de texte dans le chemin d'exécution et un audit de
réponse doit montrer que les champs finaux sont byte-for-byte ceux de la
réponse agentique validée.

## MAPS_ASSISTED

La branche Maps suit la même boucle. `GENERATOR` formule le CRM à partir de la
fiche SIRENE et de la réponse Places autorisée ; `CRITIC` conteste
indépendamment la liaison et classe. Les gardes CP/commune,
numéro/voie, nom/enseigne, unicité locale et sibling restent obligatoires.
Seul `EXACT_HIGH_CONFIDENCE` adjudicated peut devenir une donnée
`MAPS_ASSISTED`; `SILVER_AMBIGUOUS` et `REJECTED` restent séparés.
