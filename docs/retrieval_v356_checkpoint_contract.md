# V356–359 — relance complète du retrieval corrigé

Autorisation humaine « Continue » après correction V353, lot réel V355 validé
et premier dépassement20s déclaré. Nouveau run depuis zéro, pas migration du
STOP388/2000 V349/V352 et pas admission du probe V355. Dossiers historiques
immutables, aucun ancien payload/checkpoint/STOP modifié ou importé.

V356 est la copie versionnée de l'orchestrateur V349, remplaçant uniquement
le worker par V354 (RegexQuery corrigé V353). Une facade privée réutilise
les contrôles V344 sans modifier ses globals/fichiers/config. Le lancement
gèle explicitement la politique V353 et tous les anciens+nouvelles sources
de code. Chaque lot exige aussi les traces de ce compilateur, langage inchangé
et zéro partition supprimée. Les succès originaux gardent leurs expressions.
Même panel328 de2 000 requêtes, folds2/3/4 uniquement, mask329, preuves286,
heads77/228 et sparse8 gelés, collecteur natif unique100 avant tout aval.
Les2 000 sont des requêtes, jamais un pool de candidats. Pas nouvelleBAN,
index/fit/score/politique/aliasCRM ni positif injecté. Modèles aval gelés.

V348 fournit les mêmes checkpoints atomiques fsync/rename et flock exclusif.
V356 : trois tentatives par lot maximum, première tentative complète valide
seule admise, reçus des échecs conservés et hashés ; aucune sélection de qualité.
Retries uniquement sur coûts/ressources déclarés ou interruptions propres,
autre erreur/drift/intégrité/cancel fatals. STOP_KEEP_DENOMINATOR après épuisement,
aucune requête ignorée. Pas récupération d'un root STOP terminal.

V359 est la copie versionnée de V352 : un superviseur détaché, trois contrôleurs
max après SIGKILL/SIGTERM seulement, sans STOP terminal/phasecibles commencée,
nettoyage de son propre groupe NEW-session et reçu lié au lancement/PIDparent.
Pas de reprise CLI seule, de deux producteurs, reboot infini, lancement récurrent,
TCC/FDA/launchd/énergie permanents ou dépenses externes. Caffeinate-i temporaire.
Mort du superviseur/reboot/logout/volume coupé restent des limites connues.

Budgets V342 inchangés : 3Gio RSS/80s worker/20s query (indexopen inclus),
quiet30, pression1/libre≥50%, unthread. Dix secondes avant retry puis quiet30.
Le probe355 a dépassé20s au premier essai et passé14,188s au second : pas de
garantie de temps à froid ni de réussite complète. Temps des échecs conservés,
exclus de la latence des succès et explicitement non-SLO, pas hausse automatique.

V357/V358 : copies versionnées des seules autorités350/évaluation351 liées
au nouveau root, mêmes kernels324/ledger/sidecar/politique/exact-op/filtresArrow
folds234 ET2 000 contrôles. ALL500 producteurs ownexit0/PASS, hashes pré/post,
epochs, traces, dénominateur et2 000 sorties scellés AVANT tout label-side.
LIVE/STOP/partial interdisent les cibles. Aucune injection/statut filtreGT.
Fold0 contaminé nonlu/fold1fermé. PanelDEV nonpristine : pas fullOOF23571,
attributionfine/Wilson/généralisation/SLO/READY ni99,2% garanti.

Code/config/contrat gelés avant départ. Milestones commits et handover,
aucun push ni inclusion des modifications utilisateur préexistantes.

## Validation exhaustive sans répétition du même scope global

Mesure lecture seule du probe réel355 : dix validations autonomes de lot,
0,135–0,145s chacune (~69s pour500), avant même les doubles contrôles349
et les multiples admissions d'autorité350. Sans correction, la phase finale
risque de dépasser80s. On n'élargit PAS ce budget et on ne saute aucun lot.

Pendant UNE admission complète356 uniquement, le lancement/code/config/sources
query-only/inventaire sont validés intégralement avant ET après les500 lots.
Chaque checkpoint, launchglobal/shard, payload, manifest, supervision, chaque
reçu d'échec, le choix de première tentative valide et tous les contrôles
originals344 `_stage` (exit0/PASS/hash/epochs/full4/cap100) restent vérifiés.
Le shardlaunch doit être identique au scope global. Les validations autonomes
et la génération conservent leurs vérifications complètes ordinaires.
Pas cache persistant, autorité hors transaction, lecture GT anticipée ni
source absente. Mutation pré/post/epoch/receipt/missinglot = fatal.

357 utilise les résultats de cette admission exhaustive une fois par appel
d'autorisation, sans relancer deux fois la même admission dans cet appel.
Chaque recheck refait ALL500, puis vérifie le manifest hashé original.
358 conserve les kernels, cibles et barrières sans changement. Cette borne
mesurée n'est pas un SLO national garanti ; aucune baisse de dénominateur.
