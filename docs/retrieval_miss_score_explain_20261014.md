# Notes des 19 ratés V142 — 14 octobre 2026

**État : `STOP_NO_SCIENTIFIC_CONCLUSION`.** Fidélité : **1/19** requêtes. Durée totale : **140,34 s**.

**Mesuré.** Pour `10719` : mêmes 100 SIRET dans le même ordre, **écart maximal de note 0**. Le premier est `48190475300029`, note **10,986982** ; le 100e est `78995219900022`, note **6,473458**. La note du label n'a pas été calculée.

La deuxième requête, `13765`, a été arrêtée avant comparaison. Pic observé de l'enfant : **5,998 Gio** (6 440 255 488 octets). La garde, appliquée par prudence à la **somme parent + enfant**, a déclenché à **6,024 Gio** (6 467 813 376 octets), pour un plafond de **6 Gio**. Arrêt par SIGTERM confirmé, code enfant −15 ; aucun processus du calcul ne reste actif.

**Lu dans le script.** Moteur natif Tantivy et univers du run inchangés, coefficients gelés, quantification 20 bits. Chaque requête est isolée dans un processus, avec caches de requête réduits et sans représentation textuelle géante de la requête. Aucun `run`/`evaluate`, aucune génération complète ni nouveau recall. **Non exécuté / inconnu :** décomposition des 19 labels, rangs exacts et comparaison des cinq modèles. Le reste du script d'explication n'est pas validé par ce smoke.

| query_id | nom CRM court | note label | note 100e | écart | 3 signaux en retard | passe avec un autre modèle ? |
|---|---|---:|---:|---:|---|---|
| 10719 | CONVERGENCE MEDIAS | inconnu | 6,473458 | inconnu | non conclu | inconnu |
| 13765 | FOYER TY BREIZ | inconnu | inconnu | inconnu | non conclu | inconnu |
| 16145 | SIP PLAINE COMMUNE MAISON DES PROJETS | inconnu | inconnu | inconnu | non conclu | inconnu |
| 4422 | OPHLM | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:1c6805a5f39956c1909201e3e901b453 | SAS 5EME ETAGE | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:2ca6ec4d142492c19345fa989a83c988 | GROUPAMA SUPPORTS ET SERVICES | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:36e9555c94cdb18e4a726cea0dd614e1 | CEGEDIMCLOUD SMARTRX VILLEURBANNE | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:37fdb9d2e6510a987a11890caa9adf45 | INFODESCA | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:486d54cd450122e56cd41661faa6f305 | ATOL | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:5d3a6e667b24ee373fb8d4840ab19393 | TRESCAL | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:87d277db7512a10b43cb3231956f584c | NAOS | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:98eab05da0fcc3139cdbb4fa0c94518b | NAOS | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:ad20d5983bf6c000ce6b9e1761aa0f6f | CAPITAL SANTE INSTITUT | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:c78d13bc213dcb0d923aaabdf4052dcd | CERGY PONTOISE RUE DES VERGERS | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d6171348a29c3e0781b2dd602f54d06b | CAMPHES | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d7f022fe5339efeed38937fedae2a511 | ADOS LOISIRS | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d8f3afc5e08bb3556ca449fb15ba2895 | CUISINES SCHMIDT HAUCONCOURT | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:e59aa4eb6d0f28ef4b5ceccc7b8cd35b | NAOS | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:fe1206af413a0560ebdc49be6b534e76 | SYNDICAT MIXTE AGENCE LANDAISE POUR | inconnu | inconnu | inconnu | non conclu | inconnu |

## Agrégat des familles
**Aucune conclusion scientifique : arrêt avant fidélité complète ou décomposition validée.**

Blocage mesuré : **garde mémoire sur `13765` avant validation du replay**. Aucun écart de score n'est établi pour cette requête ; ce n'est pas une preuve d'infidélité du moteur. Pas de nouvelle tentative, de plafond augmenté ou de modèle simplifié.

Artefacts conservés : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v172/miss_score_explain_20261014/`. Le JSON voisin contient les reçus, les octets observés et le premier replay validé. Vérification après arrêt : **82 artefacts du manifeste inchangés** et empreinte des sorties originales conforme au sceau.

## Commande réellement exécutée et suivi

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/retrieval_miss_score_explain.py \
  > /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v172/miss_score_explain_20261014.launch.log 2>&1 < /dev/null &
# Lecture du journal uniquement ; ne pas relancer le calcul.
tail -n 20 /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v172/miss_score_explain_20261014.launch.log
```

Résultat : `REPORT STOP_NO_SCIENTIFIC_CONCLUSION 1 /19`. Le premier enfant a fini à code 0 en **27,33 s**, le second a été arrêté en **17,87 s**. La durée totale inclut la préparation des copies filtrées des contextes. Aucune explication ni aucun agrégat de signaux n'a été calculé.

**Inconnu.** Rang exact des labels hors100 : « > 100, rang non calculé ». Toutes les notes de label, leurs retards et leur éventuelle admission avec un autre modèle restent inconnus. Aucun jeu fermé ou label modifié.
