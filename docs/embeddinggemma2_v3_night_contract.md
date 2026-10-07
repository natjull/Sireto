# Nuit du 8 octobre — entraînement et recherche initiale100

L'utilisateur autorise huit heures de travail autonome local, sans budgets
d'étape artificiellement courts. Objectif inchangé :99,2% exact SIRET initial100,
pas une performance aval. Pas de garantie de résultat, test4 323/folds0/1 fermés.

V2 s'est arrêtée le7octobre21h37 après82 048/117 158encodages TRAIN, sans fit.
Le garde-fou groupé ne distingue pas temps/RSS/pression ; durée compatible avec
limite30min. Historique inchangé. Nouvelle exécution V3, mêmes données officielles
complètes et même recette dual-proof V2. Le modèle reprend V1best7200.

Budget global8h, sans limite30min pour minage. Shards128vecteurs persistés
atomiquement, hashés et liés données/code/adapter/ordretexte ; reprise exacte
après pression mémoire sans perdre l'encodage terminé. Erreurs d'intégrité ou
numériques restent fatales. Matrice disque mmap pour minage exact même commune.
Banque TRAIN seulement, masque des parents liés/incertains et des sites non
prouvés distincts ; les limites conceptuelles du minage V2 restent publiées.

Mémoire :GPU40%de24Gio, RSS8Gio, admission45%libre, yield25%, supervise20%,
pression système normale. Le minage yield20%, reprise30s ; entraînement96reprises
resource-only jusqu'à deadline. Ces protections visent une vraie saturation,
pas un plafond arbitraire2Gio. Mac/CATNAT, aucun cloud/coût/upload.

Préflight réel512tokens et batch16/micro2, mêmes objectifs75%site/25%identité,
LoRAr8 Q/V/projection, LR3e-5/cosine, deuxepochs au plus. La deadline du fit
réserve2h pour comparaison ; si entraînement incomplet publier epoch/couverture,
ne pas appeler le budget épuisé une convergence. Checkpoint25pas, meilleur
adapter choisi sur256 contrôles validation sans ouvrir test. Réserves8h et
2h sont allocations de cette nuit, pas un SLO ni une garantie.

Après fit : comparer V1/V3 sur les mêmes requêtes validation/SIREN nonentraînés,
avec tous les établissements officiels des communes sélectionnées uniquement
sur CRM/hash/comptages. Noms officiels enrichis pour TOUS les distracteurs,
jamais seulement pour la vérité. Dense multivue initial100 et lexical national
initial100, fusion RRF fixe avant métriques, aucun pool2000/500 ni modèle aval.
Sorties intégrales scellées avant labels ; GT hors partition compté perdu,
exact/opérationnel séparés, intervalles et gains/pertes publiés. Ceci reste un
diagnostic géographique de développement, pas une validation nationale/OOF.

La suite autonome doit exploiter les résultats mesurés et auditer les ratés
sur preuves officielles, pas enchainer plomberie ou grilles sans hypothèse.
Garder une approche multicanal : fine-tuning seul ne reconstitue pas les marques,
successions ou coordonnées absentes. Aucun aliasCRM appris n'est indexé.

Appuis : [fine-tuning officiel Google](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-multimodal-embeddinggemma-with-sentence-transformers),
[gradient cache Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss).
