# V348–352 — redémarrage automatique et checkpoints vérifiés

Accord humain6octobre2026 : demander surtout un mécanisme de redémarrage
automatique après STOP344. Nouvelle politique d'exécution UNIQUEMENT, jamais
modification de recette, sources, modèles, budgets ou dénominateur. Défaut
conservateur : STOP après trois échecs, aucune ligne ignorée ou inventée.
Question humaine ouverte sur continuer en comptant les échecs ; en l'absence
de réponse, STOP_KEEP_DENOMINATOR reste obligatoire. Pas de hausse20s/80s/3Gio.

349 orchestre les workers344 IMMUTABLES, dans des shards neufs sous le
namespace344/autoresume_v349. Chaque shard porte EXACTEMENT le même launch
query-only2000/recipe/hash du run, tous les folds234 et la même sélection328.
Worker344 réalise chaque fois le même lot4, same ONEcollecteur global100
286+mask329/heads77/228/sparse8/XY/successions/régions, même pression1/free50%/
quiet30/unthread/3Gio/80sworker/20squery. Pas nouvelleBAN/encoder/fit/score/aval.

Seuls les lots complets souslimites avec parentownexit0 confirmé, workerPASS,
payload complet/hashes sourcepre-post/epochs/contrôles/cap100 sont commités.
Checkpoint écrit avec flush/fsync puis rename atomique sous verrou exclusif
fcntl de contrôleur. Crashavant commit mais APRÈS reçu réel : revalidation
possible ; manifest sans reçu propre : NONadmis, nouvelle tentative fraîche.
Un checkpoint modifié/source/code/contrôle/receipt driftté est fatal, pas retry.
Chaque reprise revérifie l'intégralité des sources et producteurs concernés.

Trois tentatives par lot maximum, dix secondes avant retry puis quiet30.
Timeout/coût/RSS/pression déclarés ou interruption de processus propre peuvent
être réessayés, mêmes limites. Exceptions inconnues/compteursinvalides/drift
sont fatales. On garde les tentatives échouées mais JAMAIS leurs lignes de
sortie, ni un sous-ensemble positif, ni la meilleure recette/qualité. Première
tentative valide uniquement. Checkpoints futurs349 seulement, aucun fragment
STOP historique333/344/347 importé ou réétiqueté.

352 supervise le contrôleur détaché, jusqu'àtrois exécutions si crash sans
STOP terminal. Nettoie uniquement le groupe NEW-session de son enfant avant
reprise. STOP déclaré/budget épuisé/drift/évaluation commencée : pas restart.
Reprise d'un contrôleur349 déjà lancé requiert le reçu352 lié au launch et
au PID du parent, après nettoyage du groupe propre ; reprise CLI seule refusée.
Un autre contrôleur est exclu parflock ; pas parallèle GPU/cloud/dépense,
TCC/FDA/droits/énergie permanents modifiés. SIGKILL/reboot/logout du superviseur
lui-même ne peuvent pas être guéris par ce processus ; pas garantie générale
de reprise après extinction. Caffeinate-i temporaire prévient seulement veille
inactive. Aucun KeepAlive/scheduler ni redémarrage infini.

Seul ALL500 producteurs valides/TOUS2000 outputs scellés permet350 puis351
cibles. Kernels exact/op324 inchangés, ledger/sidecar/politique originale
et filtresArrow234ET2000query_id, aucune injection ou filtrestatut. Temps des
tentatives réussies reste descriptif, failed_attempts/controller_runs publiés
et temps ratés explicitement EXCLUS de cette latence ; aucun claim SLO prod.
PanelDEV nonpristine ; fold0nonlu/fold1fermé/avalgelé, fullOOF23571/99,2%/
Wilson/généralisation/attributionfine/SLO/READY toujours à démontrer.
Code/config/contrat figés avant exécution, milestones commits+handover.
