# V352 — superviseur détaché avec redémarrages bornés

Reprend le transport347 éprouvé et le package328 admis. Nouvelle session,
stdinDEVNULL/closefds/logsCATNAT/PPID1 aprèsstarter, un claim de superviseur.
Àl'intérieur, trois contrôleurs349 maximum et checkpointreprise349 vérifiée.
Chaque contrôleur est Popen propre, ownexit attendu, journaux séparés ;
contrôleur tué SIGKILL/SIGTERM nettoyé avant lancement du suivant, jamais deux
producteurs. Exit positif inconnu fatal, jamais une permission de retry.

Tout le code344/342/343 et349/348/350/351/352 et contrats/configs/runtime est
pinné. Pas augmentation20s/80s/3Gio et pas changement des scores. STOPterminal
ou phasecibles déjàcommencée bloque le restart ; aucune variante selon recall,
aucune réutilisation de sortiepartielle ni ancienfragment STOP. Retry worker
3maximum déclaré dans349, première tentative valide seule admissible.

Wrapper352 vérifie après succès ALL2000 génération349 et mesure351, sans
résultat qualité avantseal/permission. Heartbeat ne prouve pas ownexit/PASS.
SIGKILL/reboot/logout/volume coupé restent limites, aucune reprise de
superviseur morte garantie ; caffeinate-i temporaire, pas scheduler/launchd/
KeepAlive/TCC/FDA ou réglage permanent. Fold0/1 fermés, aval gelé.

## Utilisation (nouveau run seulement ; pas lancé pendant l'implémentation)

Depuis la racine SIRETO, préparer puis démarrer le job détaché :

```sh
.venv/bin/python -m scripts.launch_retrieval_v352_resumable prepare --control /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v352_persistent_launcher/job_20261006_a --generation /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v344_corroborated_population/autoresume_v349/run_20261006_a
.venv/bin/python -m scripts.launch_retrieval_v352_resumable start --control /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v352_persistent_launcher/job_20261006_a
```

Les deux dossiers doivent être neufs. Le premier appel vérifie les hashes et
prépare seulement le contrôle ; le second déclenche le calcul. `state.json`
et `workflow.log` dans le dossier de contrôle donnent l'avancement. Les
restarts internes sont automatiques, sans relancer ces commandes ni créer un
second job. Les anciens92 outputs STOP344 ne sont pas importés : les reprises
portent sur les futurs checkpoints349 validés. Aucun lancement long effectué
pour tester ce mécanisme ; la requête93 bloquante reste à diagnostiquer.
