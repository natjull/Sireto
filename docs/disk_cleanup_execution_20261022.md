# Ménage SSD SIRETO — 22 octobre 2026 : arrêt partiel, sans forçage

## Autorisation et périmètre

Nathan a autorisé la suppression de B (« Oui supprime », puis délégation aux agents pi), selon le brief `sireto-disk-3.md` et le plan v2 du commit `0af77259a2555e50f13574090b5fd9fa729e6329`.
Le script et le plan n’ont pas été modifiés. Statut final : **`STOP_PARTIAL_PERMISSION_DENIED`**, pas un nettoyage B complet.

`H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`.
Journal complet, privé sur SSD : `$H/cleanup_20261022.log` ; SHA256 final :
`e64e1b97698deff84cdd24f80bfec34ad52be7a62d3046674281f2b257ecbc64`.
Ce document ne publie que commandes, agrégats et diagnostics ; aucun CSV, aucune donnée nominative ni sortie individuelle de production n’est ajouté à Git.

## Contrôles et application

- Deux inventaires `lsof -nP -Fpcfan`, avec filtrage des chemins SIRETO du SSD : **0 fichier ouvert en écriture hors PID 8288**. Inventaire des processus : aucun calcul SIRETO actif identifié hors les outils de cette tâche ; PID 8288 dormant, aucun fichier B ouvert détecté. Aucun signal envoyé, aucun contrôle de VM, aucune intervention sur CatNat ou les processus tiers. Limite : visibilité des processus accessibles à l’utilisateur courant.
- `bash scripts/disk_cleanup_apply_v2.sh` : **exit 0**, 51,108 s, **587 unités explicites**, **395 979 546 624 octets attribués à B = 368,784691 Gio**, total identique au plan. Aucun fichier A exact ni chemin sous une racine A intégrale n’est ciblé ; les enveloppes B mixtes conservent leurs exceptions A/C/pending.
- `bash scripts/disk_cleanup_apply_v2.sh --apply --confirmed-by-nathan` : une seule invocation, **exit 1 après 93,185 s**. Le préflight complet des 587 unités a passé, puis des suppressions B ont eu lieu avant l’erreur ci-dessous. Le compteur final de fichiers supprimés n’a pas été émis : aucun compte de suppressions n’est inventé.

### Blocage observé

```text
PermissionError: [Errno 13] Permission denied:
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v13_stop_receipt/005fb19823721480/manifest.json
```

Le répertoire parent `005fb19823721480` appartient à l’utilisateur courant et porte le mode **0555** ; le manifeste porte le mode **0444**, sans flag d’immutabilité affiché. L’absence de droit d’écriture sur le parent bloque `unlink`.
**Aucun chmod/chown/sudo sur les cibles, aucune relance, aucun retrait manuel ni contournement des gardes.** Le ménage s’est arrêté à cette erreur. Le journal privé créé pour cette tâche est en mode0600.

## Espace réellement observé

`df -h /Volumes/CATNAT_DATA` :

| Mesure | Disponible | Occupé | Capacité |
|---|---:|---:|---:|
| Avant ménage | 230 Gi | 1,6 Ti | 88 % |
| À l’arrêt du ménage | 325 Gi | 1,5 Ti | 83 % |
| Après smoke, 22/10/2026 00:15:11 +02:00 | 325 Gi | 1,5 Ti | 83 % |

Valeurs précises de `df -k` : **241 017 608 KiB disponibles avant**, **340 268 044 KiB à l’arrêt**, **340 264 204 KiB après smoke**.
Gain disponible observé à l’arrêt : **94,652592 Gio** ; gain net après smoke : **99 246 596 KiB = 94,648930 Gio**.
Il s’agit de la variation du volume, pas d’une mesure exclusive des blocs libérés par chaque suppression ; activité indépendante et comportement APFS peuvent intervenir. Ne pas soustraire ce gain physique du total B pour annoncer une taille restante précise.

## Protections après le ménage partiel

- **3 680/3 681 fichiers V142 présents**, avant et après le smoke : **0 nouvelle absence, 0 différence** sur présence/inode/taille/mtime/mode/allocation. Le fichier `H/v6/deadline_recovery_20261008_a/fit/best_adapter.pt` était déjà absent ; il n’a pas été supprimé par cette tâche. Pas de rehash intégral des 138 Gio du gel.
- **10 630 exceptions C/pending préexistantes inchangées** sur les mêmes attributs ; aucune racine d’unité C de taille positive manquante.
- Toutes les autorités A exactes du plan restent conformes avant smoke, en excluant seulement le compteur de liens physiques dont la baisse est normale après retrait d’alias B. Les 11 empreintes de référence/modèles sont conformes après ménage ; les modèles restent identiques après smoke.
- Aucun fichier A/C supprimé, aucune action sur PID 8288, CatNat ou la VM `hydraulic`, aucun changement des fichiers étrangers du dépôt.

## Smoke de production réel

```bash
.venv/bin/python -u scripts/sireto_serve_v1.py serve \
  --input "$H/serve_v1/pilot_2000/cli_input_10.csv" \
  --output-dir "$H/serve_v1/pilot_2000/cleanup_20261022_smoke10"
```

Exécution après l’arrêt, sans nouvelle tentative de suppression : **exit 0, 58,831 s**. Même entrée880octets que le smoke précédent, SHA256 `2808adbb084d6f186154cf9e17f2a9637c12ea93dbf928c4832a0e0b74c85c00`.
Comparaison au précédent `pilot_2000/cli_csv_smoke10` : **10/10 identiques** pour les 100 candidats, classement100, scores R3, features/score de risque, NB, top4 et statuts ; écarts maximaux numériques **0**. Exports CSV et JSONL **identiques octet pour octet**, **0 nouveau vecteur Gemma**.
Seules les écritures normales demandées du serving ont été produites : nouveau namespace confidentiel et journaux/reçus de workers, avec ajout au registre mutable `serve_v1/workers.json`. Aucun ancien résultat individuel ni poids n’a été remplacé. Aucun test unitaire ni réentraînement.

## Suite

**B reste partiellement à traiter.** Une reprise nécessite un réaudit borné de l’état restant et une décision explicite sur les permissions des répertoires scellés : le plan signé v2 n’est plus rejouable tel quel après ces suppressions et les nouveaux reçus/handover. Ne pas forcer ses gardes et ne pas répéter la commande d’application.
Commit isolé du résumé et de la ligne handover, sans push : `git log -1 --format=%H -- docs/disk_cleanup_execution_20261022.md`.
