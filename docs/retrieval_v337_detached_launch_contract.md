# V337 — transport détaché depuis le contexte autorisé

Supplément physique au contrat333, sans modification de code/scoring333,
334/335/336 ni sources/modèles/budgets. Accord humain Oui pour sécuriser
une exécution persistante, une seule génération toujours NONdémarrée.

Le bootstrap336 via plistCATNAT a été refusé pour ownership/permissions.
Une copie exacte du plist dans un dossier temporaire privé a permis son
chargement, mais xpcproxy a refusé file-write-create service_stdout.log sur
CATNAT (journal système23h34m48). Exit78 avant le programme Python : aucun
one_shot_claim/state/worker/génération333 créé. Jobinactif bootouté ; pas
retry du retrieval ni sorties partielles. Cause interruption330 reste inconnue.

V337 ne change PAS les droits/TCC/FDA/volume, n'utilise pas root et ne
réessaie pas le service refusé. Il lance un enfant depuis le contexte CLI
explicitement autorisé, avec ses permissions héritées, une nouvelle session
POSIX, stdinDEVNULL et stdout/stderr ouverts par le parent dans CATNAT.
Le shell/outil peut terminer sans fermer son terminal/session de travail.
Vérifier SID/PGID==PID et parentPID1 APRÈS retour du starter ; cela prouve
le détachement, pas une survie garantie à toute panne ou politique OS.

Un marqueur de start O_EXCL et un marqueur de job O_EXCL interdisent tout
second démarrage/retry/rebootstrap. Aucun calendrier/KeepAlive/restart.
Heartbeat five-seconds, owned Popen exit réel du workflow, nettoyage de son
seul groupe de processus en cas d'arrêt normal/signal, caffeinate-i temporaire.
Capot/logout/reboot/volume retiré/SIGKILL restent des limites, jamais une
permission de reprise. Aucun artefact STOP/LIVE/partiel évalué.

V337 pinne TOUTE l'implémentation333 originale, le runtime, son code et ce
supplément avant exécution, et les re-vérifie après le workflow. V333 garde
son freeze739beefe et lance ses500 lots de4, tous exit0/PASS avant334/335GT.
Le panel328 COMPLETE et tous2000contrôles sont réutilisés, jamais fragments
330 ; pas nouvelle BAN/encoder/tirage/fit/score. Limites214 inchangées2Gio/
80sworker/20squery/quiet30/unthread/pression1/free50%, cap100direct natif.
Fold0 nonlu/fold1fermé/avalgelé ; exact/op séparés, DEV nonpristine, SLO/
fullOOF/généralisation/99,2%/READY nonvalidés. Code mesuré reste immuable,
aucun autre producteur lourd, aucun retry après un nouvel échec retrieval.
