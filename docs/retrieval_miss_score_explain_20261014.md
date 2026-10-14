# Notes des 19 ratés V142 — 14 octobre 2026

**État : `COMPLETE_FAITHFUL_MISS_SCORE_DIAGNOSTIC`.** Fidélité : **19/19** requêtes.
Pic résident observé : **3.867 Gio** ; plafonds enfant seul **8/4 Gio** (complète/allégée). Durée : **906.08 s**.

Fidélité par méthode : `{'complete_historical': 1, 'light': 18}`. Le replay historique de `10719` est conservé.

Contrôle : mêmes 100 SIRET et notes alignées **par SIRET**, tolérance inchangée 1e−5. L’ordre est enregistré séparément. Le premier arrêt de reprise concernait seulement deux ex æquo inversés sur `ad20…` : mêmes100, erreur par SIRET 1,08e−6 ; ce contrôleur trop strict a été corrigé, sans changement du moteur ni de ses notes.

**Mesuré.** Même moteur Tantivy, mêmes univers, cinq coefficients gelés, quantification 20 bits. Un processus par requête ; aucune relance complète, aucun nouveau recall.

| query_id | méthode / pic Gio | note label | premier : SIRET / note | 100e : SIRET / note | écart | 3 signaux en retard | autre modèle : bat le seuil enregistré ? |
|---|---|---:|---|---|---:|---|---|
| 10719 | allégée / 2.254 | 2.702122 | 48190475300029 / 10.986982 | 78995219900022 / 6.473458 | 3.771336 | own_address_proof (+0.435896); own_geo_quality (+0.364951); site_x_geo (+0.333202) | non |
| 13765 | allégée / 3.591 | 4.414280 | 20000554400024 / 8.638061 | 91997684500017 / 4.786485 | 0.372205 | own_address_proof (+0.220217); name_x_address (+0.209415); own_street_similarity (+0.182942) | non |
| 16145 | allégée / 1.841 | 3.157493 | 48089858400036 / 4.668259 | 10472898500011 / 3.411345 | 0.253852 | parent_cosine (+0.250355); site_cosine (+0.149371); parent_x_address (+0.124279) | non |
| 4422 | allégée / 2.133 | 3.536363 | 40278771700022 / 6.079288 | 31888357600031 / 3.674513 | 0.138149 | parent_cosine (+0.363895); succession_dense_context (+0.210747); succession_site_context (+0.114608) | oui (0) |
| NEWCRM:1c6805a5f39956c1909201e3e901b453 | allégée / 3.825 | 4.741258 | 84287677300016 / 8.836025 | 83333442800012 / 4.847869 | 0.106612 | parent_cosine (+0.172119); site_cosine (+0.080218); parent_x_address (+0.064871) | oui (1) |
| NEWCRM:2ca6ec4d142492c19345fa989a83c988 | allégée / 2.170 | 2.683944 | 42761930900037 / 6.983894 | 41536670700016 / 5.394906 | 2.710962 | site_cosine (+0.616162); own_geo_quality (+0.335342); site_x_geo (+0.323948) | non |
| NEWCRM:36e9555c94cdb18e4a726cea0dd614e1 | allégée / 2.697 | 3.848045 | 71980481700067 / 5.800704 | 79207080700021 / 3.968051 | 0.120006 | own_street_similarity (+0.182181); own_address_proof (+0.133569); parent_x_address (+0.036759) | non |
| NEWCRM:37fdb9d2e6510a987a11890caa9adf45 | allégée / 2.196 | 3.416977 | 49996975600026 / 5.515627 | 89514233900012 / 3.735051 | 0.318074 | succession_dense_context (+0.180521); succession_site_context (+0.096745); parent_cosine (+0.061679) | non |
| NEWCRM:486d54cd450122e56cd41661faa6f305 | allégée / 2.452 | 2.625890 | 81968686600024 / 6.600632 | 53745225200043 / 4.392061 | 1.766171 | parent_cosine (+0.529417); site_cosine (+0.300546); parent_x_address (+0.276319) | non |
| NEWCRM:5d3a6e667b24ee373fb8d4840ab19393 | allégée / 1.757 | 3.624580 | 56204705000819 / 7.727541 | 82061109300015 / 4.048359 | 0.423779 | site_cosine (+0.334400); own_street_similarity (+0.170403); parent_x_address (+0.151287) | non |
| NEWCRM:87d277db7512a10b43cb3231956f584c | allégée / 2.711 | 3.264978 | 30844902412688 / 7.446741 | 89134182800012 / 5.556520 | 2.291541 | own_address_proof (+0.424309); own_geo_quality (+0.416593); site_x_geo (+0.379931) | non |
| NEWCRM:98eab05da0fcc3139cdbb4fa0c94518b | allégée / 1.605 | 2.903915 | 38497078600313 / 6.019317 | 35319084600033 / 3.841687 | 0.937771 | succession_dense_context (+0.368517); succession_site_context (+0.339702); own_geo_quality (+0.104685) | non |
| NEWCRM:ad20d5983bf6c000ce6b9e1761aa0f6f | allégée / 3.867 | 2.012496 | 98946449000071 / 7.125662 | 31728846200011 / 5.078611 | 3.066115 | site_cosine (+0.666242); parent_cosine (+0.357871); succession_dense_context (+0.301022) | non |
| NEWCRM:c78d13bc213dcb0d923aaabdf4052dcd | allégée / 2.763 | 4.574586 | 34811254100043 / 6.055947 | 38083687400020 / 4.750906 | 0.176319 | succession_dense_context (+0.335518); parent_cosine (+0.259930); succession_site_context (+0.210980) | oui (0) |
| NEWCRM:d6171348a29c3e0781b2dd602f54d06b | allégée / 1.411 | 3.314436 | 50953693400011 / 4.768494 | 35035253000017 / 3.653134 | 0.338698 | parent_identity_x_parent_geo (+0.095411); parent_portfolio_name_x_geo (+0.073040); site_x_geo (+0.069926) | non |
| NEWCRM:d7f022fe5339efeed38937fedae2a511 | allégée / 1.563 | 3.431245 | 78506433800011 / 6.737812 | 62971149000017 / 4.097994 | 0.666750 | own_address_proof (+0.214619); own_street_similarity (+0.200704); char_addresses (+0.171943) | non |
| NEWCRM:d8f3afc5e08bb3556ca449fb15ba2895 | allégée / 1.617 | 2.727318 | 48405866400067 / 5.600386 | 84969916000024 / 3.398498 | 0.671179 | own_street_similarity (+0.210272); own_address_proof (+0.185759); own_geo_quality (+0.119102) | non |
| NEWCRM:e59aa4eb6d0f28ef4b5ceccc7b8cd35b | allégée / 2.375 | 3.660820 | 90271669501114 / 5.625860 | 91756070800047 / 4.354993 | 0.694173 | parent_cosine (+0.358780); parent_x_address (+0.289962); own_address_proof (+0.224345) | non |
| NEWCRM:fe1206af413a0560ebdc49be6b534e76 | allégée / 1.409 | 2.728859 | 51041188700043 / 5.315753 | 53420160300014 / 3.015384 | 0.286526 | own_address_proof (+0.125299); own_street_similarity (+0.103667); name_x_address (+0.085868) | non |

## Agrégat des familles
Nombre de requêtes où la famille figure parmi les trois retards principaux (familles dédupliquées par requête).

- Entreprise, acteurs et continuité économique : **12/19**.
- Ressemblance Gemma site/entreprise : **10/19**.
- Adresse propre et interactions : **9/19**.
- Géographie propre et interactions : **6/19**.

Les 58 contributions sont coefficient × valeur. Les 51 valeurs numériques sont avant quantification des deux blocs : elles ne sont pas arrondies séparément. Les bonus et le résidu de quantification/arithmétique native referment la note.
Le bonus de nouvelles sources est un **maximum**, pas une somme : ses alternatives ne sont pas créditées plusieurs fois.
Pour la méthode complète, le JSON teste aussi le vrai top100 national des cinq modèles. Pour le repli allégé, « autre modèle » compare **uniquement les notes du label et du 100e enregistré** : dépasser celui-ci ne prouve PAS une admission au nouveau top100 national. Celle-ci reste inconnue, sans classement d’un pool101 présenté comme moteur. Les mêmes préparateurs BANO/OSM recalculent leurs bonus avec ces poids et leurs univers complets.
La méthode allégée exécute les fonctions natives et projette seulement leurs jointures SIRET/SIREN sur les documents demandés. Les bornes de quantification et les agrégations parent restent calculées sur tous les sites ; les statistiques lexicales sont nationales. La fidélité vérifie les 100 notes, pas la composition d’un nouveau top100 national.
Le premier candidat, le seuil, les 58 signaux, les bonus et les cinq modèles sont détaillés dans le JSON voisin. Aucune règle de choix de modèle n’est proposée.

**Inconnu.** Rang exact des labels hors100 : « > 100, rang non calculé ». Les écarts expliquent la compétition mesurée, pas une causalité ni un gain de recall. Aucun jeu fermé ou label modifié.

## Exécution et vérifications finales

- Commande réellement exécutée, détachée : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/retrieval_miss_score_explain.py` ; journal `H/v172/miss_score_explain_20261014/resume_20261015_b.launch.log` (H défini dans le brief). Ne pas relancer sur ce namespace existant.
- Reprise B : **906,08 s** ; reprises A + B : **1 062,69 s** (17 min 43 s), hors smoke préalable. Replays A réutilisés, arrêt et reçus conservés ; aucun artefact précédent écrasé.
- Avant les calculs, libre + inactive : **7,446 à 9,178 Gio**, toujours < 10 Gio. Méthode complète donc non admise ; toutes les décompositions sont allégées. Aucun acte sur la VM Lima.
- Vérification en lecture seule des résultats : **19/19 triplets**, **3 306 contributions**, **95 comparaisons de modèles**, **19/19 enfants d’explication sortis à code 0**. Erreur maximale des 1 900 notes : **1,09143e−6**, inférieure à 1e−5.
- **82 entrées du manifeste inchangées** (taille/mtime), empreinte SHA-256 des sorties originales conforme au sceau. Toutes les décompositions se referment dans leur borne de quantification/arithmétique native ; résidu absolu maximal **1,33120e−5**.
- Fragilité mesurée : **3/19** labels battent la note du 100e enregistré avec un autre modèle (`4422`/0, `1c6805…`/1, `c78d13…`/0). Aucune admission dans un autre top100 national n’est établie.


## Historique — arrêt précédent conservé intégralement

# Notes des 19 ratés V142 — 14 octobre 2026

**État : `STOP_NO_SCIENTIFIC_CONCLUSION`.** Fidélité : **12/19** requêtes.
Pic résident observé : **1.821 Gio** ; plafonds enfant seul **8/4 Gio** (complète/allégée). Durée : **156.62 s**.

Fidélité par méthode : `{'complete_historical': 1, 'light': 11}`. Le replay historique de `10719` est conservé.

**Mesuré.** Même moteur Tantivy, mêmes univers, cinq coefficients gelés, quantification 20 bits. Un processus par requête ; aucune relance complète, aucun nouveau recall.

| query_id | méthode / pic Gio | note label | premier : SIRET / note | 100e : SIRET / note | écart | 3 signaux en retard | autre modèle : bat le seuil enregistré ? |
|---|---|---:|---|---|---:|---|---|
| 10719 | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| 13765 | light / 1.648 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| 16145 | light / 1.319 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| 4422 | light / 1.436 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:1c6805a5f39956c1909201e3e901b453 | light / 1.731 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:2ca6ec4d142492c19345fa989a83c988 | light / 1.361 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:36e9555c94cdb18e4a726cea0dd614e1 | light / 1.506 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:37fdb9d2e6510a987a11890caa9adf45 | light / 1.340 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:486d54cd450122e56cd41661faa6f305 | light / 1.398 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:5d3a6e667b24ee373fb8d4840ab19393 | light / 1.338 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:87d277db7512a10b43cb3231956f584c | light / 1.514 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:98eab05da0fcc3139cdbb4fa0c94518b | light / 1.274 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:ad20d5983bf6c000ce6b9e1761aa0f6f | light / 1.821 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:c78d13bc213dcb0d923aaabdf4052dcd | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d6171348a29c3e0781b2dd602f54d06b | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d7f022fe5339efeed38937fedae2a511 | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:d8f3afc5e08bb3556ca449fb15ba2895 | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:e59aa4eb6d0f28ef4b5ceccc7b8cd35b | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |
| NEWCRM:fe1206af413a0560ebdc49be6b534e76 | non exécutée / 0.000 | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |

## Agrégat des familles
**Aucune conclusion scientifique : arrêt avant fidélité complète ou décomposition validée.**

Blocage mesuré : `RuntimeError('{\'query_id\': \'NEWCRM:ad20d5983bf6c000ce6b9e1761aa0f6f\', \'phase\': \'fidelity\', \'method\': \'light\', \'pid\': 51034, \'exit_code\': 1, \'memory_cap_bytes\': 4294967296, \'guard_scope\': \'CHILD_ONLY\', \'peak_resident_bytes\': 1955151872, \'elapsed_seconds\': 14.268742625135928, \'admission\': {\'free_plus_inactive_bytes\': 9231351808, \'vm_stat\': \'Mach Virtual Memory Statistics: (page size of 16384 bytes)\\nPages free:                                    75251.\\nPages active:                                 491228.\\nPages inactive:                               488186.\\nPages speculative:                              9192.\\nPages throttled:                                   0.\\nPages wired down:                             206930.\\nPages purgeable:                                 262.\\n"Translation faults":                    16374043718.\\nPages copy-on-write:                       896780357.\\nPages zero filled:                       10142473107.\\nPages reactivated:                        1738178154.\\nPages purged:                              105891890.\\nFile-backed pages:                            553465.\\nAnonymous pages:                              435141.\\nPages stored in compressor:                   999315.\\nPages occupied by compressor:                 258933.\\nDecompressions:                           1292050208.\\nCompressions:                             1499030614.\\nPageins:                                  1978430045.\\nPageouts:                                   23163664.\\nSwapins:                                   194494991.\\nSwapouts:                                  206549649.\\n\', \'complete_admitted\': False, \'observed_unix\': 1792016396.2664618}, \'owned_child_exit_confirmed\': True, \'stop_reason\': None, \'child_error\': {\'error\': "ValueError(\'STOP_FIDELITY : top100/notes non reproduits\')", \'peak_rss_bytes\': 1955151872, \'traceback\': \'Traceback (most recent call last):\\n  File "/Users/nathanjullia/Documents/Projets/SIRETO/scripts/retrieval_miss_score_explain.py", line 824, in <module>\\n    worker(args.query_id, args.worker)\\n    ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File "/Users/nathanjullia/Documents/Projets/SIRETO/scripts/retrieval_miss_score_explain.py", line 522, in worker\\n    raise ValueError(\\\'STOP_FIDELITY : top100/notes non reproduits\\\')\\nValueError: STOP_FIDELITY : top100/notes non reproduits\\n\'}, \'log\': \'/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v172/miss_score_explain_20261014/resume_20261015_a/logs/fidelity_light_1c4d17a5ae4b2a8f3d94d5bdf1bfd13ced171d02a8eb49cdd64bad15931d7989.log\'}')`. Artefacts partiels conservés sous `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v172/miss_score_explain_20261014/resume_20261015_a`.

**Inconnu.** Rang exact des labels hors100 : « > 100, rang non calculé ». Les écarts expliquent la compétition mesurée, pas une causalité ni un gain de recall. Aucun jeu fermé ou label modifié.


## Historique — arrêt précédent conservé intégralement

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
