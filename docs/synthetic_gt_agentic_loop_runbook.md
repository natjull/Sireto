# Runbook — boucle agentique GT synthétique SIRETO

## Rôle du code

`scripts/run_synthetic_gt_agentic_loop.py` ne produit et ne réécrit aucun
texte CRM. Il maintient une file durable SQLite, attribue des leases disjoints
aux workers Luna, valide leurs réponses et publie uniquement les variantes
acceptées. Les réponses brutes sont conservées avec leur SHA-256.

Le moteur mécanique historique `scripts/build_synthetic_gt_corpus.py` ne fait
pas partie de ce chemin d'exécution.

## États

```text
PENDING_GENERATOR
  -> LEASED_GENERATOR
  -> PENDING_CRITIC | READY_SUPERVISOR | PENDING_GENERATOR(retry)
  -> LEASED_CRITIC
  -> READY_SUPERVISOR | PENDING_ADJUDICATOR
  -> LEASED_ADJUDICATOR
  -> READY_SUPERVISOR
  -> COMPLETED
```

Un lease expiré retourne dans son état `PENDING_*`. SQLite utilise WAL,
`BEGIN IMMEDIATE`, `synchronous=FULL` et une contrainte unique sur
`(run_id,target_siret)`.

La commande `abandon` abandonne uniquement la tâche louée et remet toujours
la seed dans le même état `PENDING_*`. Elle ne l'envoie jamais au superviseur.
Le superviseur refuse par ailleurs de terminer une seed qui ne possède pas
exactement trois variantes.

## Répartition des agents

- parent : coordinateur et `supervise`, sans génération de volume ;
- `luna-g1`, `luna-g2` : deux workers `GENERATOR` en thinking low ;
- `luna-c1` : worker `CRITIC` indépendant en thinking low ;
- `ADJUDICATOR` : uniquement pour les lots où le critique ne rend pas trois
  `ACCEPT`.

Le pilote conserve `critic_mode=all`. Le mode ciblé n'est initialisable
qu'avec `--allow-easy-supervisor`, après validation explicite du pilote.

## Commandes de la boucle

```bash
python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" init \
  --run-id pilot-v1 --seeds seed_cards.jsonl

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" lease \
  --run-id pilot-v1 --role GENERATOR --worker-id luna-g1 \
  --limit 8 --output generator_tasks.jsonl

# Luna lit les tâches et écrit generator_responses.jsonl directement.

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" submit \
  --role GENERATOR --worker-id luna-g1 --input generator_responses.jsonl

# En cas d'échec du worker, réinscrire la seed sans la publier :
python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" abandon \
  --task-id TASK_ID --role GENERATOR --worker-id luna-g1 \
  --reason worker_failed

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" lease \
  --run-id pilot-v1 --role CRITIC --worker-id luna-c1 \
  --limit 16 --output critic_tasks.jsonl

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" submit \
  --role CRITIC --worker-id luna-c1 --input critic_responses.jsonl

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" supervise \
  --run-id pilot-v1 --limit 1000

python scripts/run_synthetic_gt_agentic_loop.py --db "$DB" status \
  --run-id pilot-v1
```

Les tâches `CRITIC` ne contiennent ni `transformation_summary`, ni familles,
ni justification du générateur. Le critique voit seulement la fiche source
et les cinq champs CRM produits.

## Contrôles fail-closed

- folds autorisés : 2, 3, 4 uniquement ;
- splits autorisés : `train`, `train_synthetic` ;
- relation `SIRET[:9] == SIREN` obligatoire ;
- trois variantes exactement : `v1`, `v2`, `v3` ;
- aucune fuite de SIRET/SIREN dans les champs CRM ;
- aucune réparation du JSON invalide ;
- doublons exacts ou purement cosmétiques renvoyés au générateur ;
- familles `name/address/orthographic` validées avant le premier lease ;
- contrat explicite `v1=name`, `v2=address`, `v3=orthographic` ;
- contexte du préflight précédent transmis au retry, sans réécriture par le code ;
- deux tentatives maximum par défaut ;
- un `REJECT` critique ne peut jamais être promu par le superviseur ;
- seuls les fichiers `accept.jsonl` sont entraînables.

Les fichiers `silver.jsonl` et `reject.jsonl` restent des artefacts d'audit.
