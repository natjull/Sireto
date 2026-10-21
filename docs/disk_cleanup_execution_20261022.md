# Ménage SSD SIRETO — 22 octobre 2026 : B terminé après reprise bornée

**Clôture : `B_COMPLETE`**, reprise exit0, aucun fichier B restant dans l’allowlist, gain disponible cumulé observé **364,986164 Gio**, production10/10identique. Les résultats de la première application sont conservés ci-dessous comme historique ; la reprise autorisée par `sireto-disk-4.md` est détaillée en dernière section.

## Première application — autorisation et périmètre (commit `6b875eb7`)

Nathan a autorisé la suppression de B (« Oui supprime », puis délégation aux agents pi), selon le brief `sireto-disk-3.md` et le plan v2 du commit `0af77259a2555e50f13574090b5fd9fa729e6329`.
Lors de cette première application, le script et le plan n’avaient pas été modifiés. Son statut était **`STOP_PARTIAL_PERMISSION_DENIED`**, pas un nettoyage B complet ; il est désormais supersédé par la clôture de la reprise ci-dessous.

`H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`.
Journal de cette première application, privé sur SSD : `$H/cleanup_20261022.log` ; SHA256 inchangé :
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

## Reprise autorisée — brief `sireto-disk-4.md`

### Réaudit borné et gardes adaptées

Nathan a confirmé la suppression de toute B et autorisé le déverrouillage des seuls chemins B restants. Aucune intervention sur CatNat, Oùbusiness, la VM `hydraulic` ou PID8288 ; aucun signal envoyé. Aucun écrivain SIRETO visible hors les outils de cette tâche, notamment au contrôle `lsof` immédiatement avant l’application ; aucun fichier B ouvert par PID8288 détecté. Aucun fichier tiers du dépôt modifié.

Le premier dry-run inchangé a refusé **exit1 : handover modifié**, comme attendu après le commit `6b875eb7`. Le réaudit a retrouvé **184 unités déjà absentes**, **49 enveloppes ne contenant plus que des exceptions protégées**, et **354 unités B restantes / 316 857 fichiers-liens / 290 060 410 880 octets = 270,139809 Gio** recensés. Aucun nouveau fichier B postérieur à l’audit initial n’a été détecté. Décompte dédupliqué sur les inodes B restants et excluant les inodes A/C partagés connus ; ce n’est pas une promesse de gain physique.

Liste exhaustive des354unités, tailles nettes B, snapshots et exceptions : **`$H/cleanup_resume_20261022.json`**, privé mode0600, SHA256 **`06b3b2f717c8c81e4c1520be4adb6064627fd1aacc196aba241657d82bfb5311`**. Le journal contient également les354lignes `DRY_RUN` et les354lignes `APPLY_CANDIDATE`.

Adaptation minimale nécessaire dans `scripts/disk_cleanup_apply_v2.sh` :
- Le plan historique `docs/disk_cleanup_plan_v2.json` et son SHA256 **`e905191c50046c167a600d6839a39af1fb33dc103066e59b2469e2002379ec9d`** restent **strictement inchangés** ; les **587 tuples explicites chemin/scope sont inchangés**.
- Un état de reprise signé fournit uniquement un **sous-ensemble** de ces tuples, les snapshots/tailles du reste, le HEAD audité `6b875eb7`, et le SHA du handover à ce réaudit. Chaque reprise est refusée si un tuple est ajouté/dupliqué ou une exception C modifiée. Les protections A exactes, racines A, parents réservés, C et pending restent celles du plan original.
- Pour les autorités A, les seuls écarts admis au plan original sont les compteurs `nlink` diminués par les alias B déjà retirés et le handover explicitement publié. Tous les autres attributs A restent contrôlés contre le plan original puis contre le réaudit signé.
- Le nouvel inventaire A3 contient4913JSON au lieu de4894, après le smoke déjà demandé en phase3 et ses reçus de workers. Les533autorités de métadonnées directes ont toutes leur SHA initial ; les3131gardes de sources/documents restent identiques. Le contrôle d’inventaire A3 est repinné à cet état précis, jamais désactivé. Le résumé d’exécution est ajouté aux seuls fichiers propres autorisés par le garde Git.
- Aucune autre logique d’unlink, d’exclusion ou de contrôle du script n’a changé. Aucun contournement global des SHA, aucune expansion de cibles, aucun nouvel entraînement.

### Déverrouillage et application

`bash scripts/disk_cleanup_apply_v2.sh` adapté : **exit0, 33,275 s**,354unités, aucune entrée A/C/pending parmi les fichiers candidats ; les enveloppes mixtes gardent leurs exceptions.

Seuls deux répertoires **B purs**, et non leur parent commun, ont reçu `chmod u+w` (**0555→0755**) :
- `$R/audits/retrieval_v13_stop_receipt/005fb19823721480` ;
- `$R/audits/retrieval_v13_stop_receipt/bdcba90d29bd7da0` ;
avec `$R=/Volumes/CATNAT_DATA/SIRETO_RECALL100`. Chaque commande, mode avant/après et succès est journalisé. **Choix plus restrictif que `chmod -R` : aucun chmod sur les fichiers**, car `unlink` ne requiert que le droit d’écriture du répertoire parent et un fichier B peut partager son inode avec A/C. Aucune permission A/C/pending ni celle d’un parent partagé n’a été changée : **3555 répertoires protégés/partagés contrôlés, 0 différence**.

`bash scripts/disk_cleanup_apply_v2.sh --apply --confirmed-by-nathan` : **une invocation, exit0, 107,590 s, 316 857 fichiers/liens B effectivement supprimés**. Contrôle indépendant de toutes les587unités originales : **0 fichier B restant**. Aucun sudo, chmod récursif, suppression hors allowlist ni suppression de données A/C/pending.

### Espace final mesuré

| Moment | Disponible `df -h` | Disponible précis `df -k` | Occupé / capacité |
|---|---:|---:|---:|
| Avant toute suppression, phase3 | 230 Gi | 241 017 608 KiB | 1,6 Ti / 88 % |
| Avant reprise, phase4 | 325 Gi | 340 264 184 KiB | 1,5 Ti / 83 % |
| Après suppression phase4 | 595 Gi | 623 737 184 KiB | 1,2 Ti / 69 % |
| Après smoke final phase4 | **595 Gi** | **623 733 340 KiB** | **1,2 Ti / 69 %** |

Gain net observé pendant la reprise : **270,337254 Gio** ; gain cumulé depuis l’avant phase3 : **364,986164 Gio**. Ces différences de disponibilité du volume ne sont pas une attribution exclusive par fichier ; liens physiques conservés, APFS, métadonnées et activité indépendante expliquent qu’il ne faille pas les assimiler au budget B initial368,784691Gio.

### Préservation et smoke final réel

- **3680/3681 fichiers V142 présents avant/après smoke, 0 nouvelle absence, 0 différence** sur inode/mode/taille/mtime/allocation. Seul le fichier déjà absent `H/v6/deadline_recovery_20261008_a/fit/best_adapter.pt` manque toujours ; restauration hors brief. Aucun rehash intégral des138Gio du gel annoncé.
- **10630 exceptions C/pending inchangées**, aucune racine C positive manquante, toutes les autorités A exactes conformes hors variation normale `nlink`, **3555 répertoires protégés/partagés sans modification de mode**,11empreintes de référence conformes, modèles inchangés après smoke. Ces contrôles précèdent la publication de cette mise à jour de journal autorisée.
- Commande réelle `.venv/bin/python -u scripts/sireto_serve_v1.py serve --input "$H/serve_v1/pilot_2000/cli_input_10.csv" --output-dir "$H/serve_v1/pilot_2000/cleanup_resume_20261022_smoke10"` : **exit0, 58,854 s**. Même entrée et SHA que le smoke précédent.
- Comparaison à `cli_csv_smoke10` : **10/10 identiques**, mêmes100candidats et classements100, tableaux `pred/risk_features/score/tier/top4/nb` identiques, écarts maximaux **0**, CSV/JSONL identiques octet pour octet, **0 nouveau vecteur Gemma**. Écritures limitées au nouveau namespace de smoke et aux reçus normaux/registre mutable des workers ; aucun ancien résultat ni modèle remplacé. Aucun test unitaire ni évaluation scientifique supplémentaire.

Journal complet privé de la reprise : **`$H/cleanup_resume_20261022.log`**, SHA256 **`88afe606dc9b7f5207b61a218a3dd223dfe92580b50c640554de9bea3facbe92`**. Il contient refus initial, liste et tailles, deux chmod, dry-run/application, contrôles, smoke et mesures `df`. L’ancien journal reste inchangé.

**Reste : aucun fichier B à supprimer dans l’allowlist validée.** Les répertoires réservés ou contenant A/C/pending restent volontairement présents. Les nouveaux reçus et ce handover périment à nouveau les gardes de reprise : ne pas rejouer ce script ni élargir ses cibles ; tout futur ménage requiert son propre audit/autorisation. Aucun push.
Commit isolé de clôture (résumé, script borné et handover seulement) : `git log -1 --format=%H --grep='docs: finish authorised B cleanup with bounded resume'`.
