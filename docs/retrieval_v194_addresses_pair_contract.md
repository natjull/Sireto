# V194 — deux lectures d'adresse nationales, sans mélange des requêtes

Protocole écrit avant mesure. Test physique aux positions 0 et 1 figées,
issues de la population de 128 CRM vérifiée intégralement par V192. Aucun
label, fold0/1, ancien candidat ou token d'expansion n'est lu. Chaque requête
conserve ses propres nom, numéro, suffixe, voie, CP, INSEE et coordonnées BAN
avec leur fiabilité ; le contrôle JSON de la requête est revérifié à l'admission.
Les identifiants query ne servent qu'à l'alignement, jamais de feature.

Ouvrir UNE fois PagedAddresses178 et CertifiedAddressFacts181. Mêmes sources
177/160 officielles, 34 346 446 variantes d'adresse, 44 403 034 sites et
98 804 872 faits certifiés, sans exclusion géographique, d'état ni de numéro.
Pour CHAQUE requête, produire les vecteurs complets float64/bool de couverture
et d'exactitude par le kernel178 inchangé ; conserver les fichiers, fermer
uniquement les maps propres. Aucune propagation de preuve site/parent ici.
Les trois spans physiques couvrent le corpus entier et restent ceux de183.

Les 93 507 epochs originaux et toutes les autorités STOP/source177 restent
contrôlés avant/après les requêtes et avant/après l'audit. Capturer en plus
targets/epochs exacts de TOUS les facts.used181 (53k fichiers, pas uniquement
une partition), tous les CSR et autorités178, plan/ledger/impl177, catalogue,
population/query/vecteurs/modèles/geo/3 décodeurs et les quatre fichiers de
preuve. Un worker séparé FULL-hash chaque fichier capturé ; publier files,
bytes et digest exact. Aucun epoch/stat ne remplace un hash. Les deux preuves
restent en quarantaine avant les vrais exit0, preflights/report153, receipts
et contrôles d'admission. Aucune source ni artefact historique supprimé.

La première requête doit reproduire BIT À BIT les deux fichiers184 déjà
scellés : coverage a4167fa3e78e366ba9a9139e9d33d068bf1972f248719b1416db79d84d5c19cb ;
exact 7e08546b0f671246dfd6e4b7ec236678f411eaaf94a8b954b7c2de2953ad0f6d.
Ce sont des preuves d'adresse, PAS un candidat ou label utilisé pour chercher.

Limites inchangées : worker <=1Gio RSS/80s, requête <=20s, quiet30,
pression kernel1 et mémoire libre >=50%. Swap global télémétrique non
attribué. Mac/CATNAT seulement, aucune dépense ni purge système/autres apps.
Un STOP conserve ses artefacts et ne peut être relancé dans le même scope.
Temps publié : chargement+deux requêtes+contrôles+audit complet, plus temps
parent/quiet séparé. Cache froid/chaud inconnu : aucune garantie de vitesse.

V194 ne produit AUCUN initial100 et ne mesure PAS recall/SLO. Objectif intact :
recherche initiale <=100 et 99,2% SIRET exact AVANT aval gelé, sans pool2000/500.
Suite requise : scoring propre à chaque query/ownfold, puis toutes128 sorties
scellées avant GT séparée ; fullOOF/Wilson/généralisation/loss-sidecar/SLO
toujours dus. Un succès de ce test ne vaut ni READY, ni SOTA, ni preuve de99%.
