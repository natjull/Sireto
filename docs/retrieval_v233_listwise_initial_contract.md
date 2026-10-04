# V233 : calibration initiale sur le score réel, loss listwise figée

Préenregistrement4 octobre2026, avant mesure. Nécessite23264 complets,
parité217 tous points et trois heads, huit exits réels PASS214. MêmeTRAIN64
disjoint du pilote128, registre234 original complet, modèles77 fold-safe.
Ne pas lancer si232 STOP/partiel. Sources / poids historiques immuables.

Pour chaque fold externe234, exclure requêtes du fold, positifs des SIREN
protégés et chaque négatif de ces SIREN. Exclure positif exact et équivalents
officiels acceptable_operational, filtrés par query_id64 ET folds234. Garder
les concurrents locaux et nationaux naturels, dédupliqués SIRET. Aucun statut
de qualification ne filtre les données d'évaluation produit.

Apprendre les onze poids EFFECTIFS217 non négatifs[0,64]. Contributions
alpha/sparse77, propriétaires/successions et XY fixées, incluses dans les
marges d'apprentissage telles qu'elles seront en service. Prior poids77
OOF protégés, ridge0,005 fixe autour de ce prior, L-BFGS-B500 itérations,
ftol1e-10/gtol1e-7. Loss par requête = log(1+sum exp(score_neg-score_pos)),
moyenne sur les requêtes puis ridge. Cela donne davantage de gradient aux
concurrents qui devancent le positif, sans sélectionner par labels128 ni
tronquer les négatifs TRAIN. Aucune recherche de paramètres/familles ici.

Après fit des trois heads, contrôle OOF sur les64 requêtes TRAIN du fold
tenu seulement. Les points positifs sont supervisés, pas injectés dans une
sortie produit ; le nombre de concurrents qui les devancent sur un échantillon
miné n'est PAS Recall@100 national. Ne jamais appeler ce contrôle un plafond
du rappel. Contrôle extérieur non utilisé pour sélectionner/tuner les poids
ou parameters. Avant émission produit, nouveau contrat/driver figé requis.

Usage futur : mêmes poids compilés dans les postings AVANT un seul
collecteur100, pas pool fixe puis reranker déguisé. Vocab/projections/alpha
77/78 et modèles aval restent gelés. Pas de gate2000/500, de nouveau dense
national à encoder, téléchargement, dépense externe ou autre application.

Guards214 inchangées2Gio/80s/un thread/calme30s/pression normale50% libres.
Le vrai rappel sera uniquement mesuré après128 sorties complètes en process
distinct ; panel réutilisé, aucun READY même100%. OOF complet99,2%, Wilson,
chaque fold, attribution/généralisation/SLO restent obligatoires,fold0/1 fermés.
