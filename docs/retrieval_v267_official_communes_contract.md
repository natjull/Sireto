# V267 : snapshot source officiel communes, pas retrieval ni recall

5 octobre2026 AVANT acquisition. Un GET CONSTANT de geo.api.gouv.fr,
champs nom/code/codesPostaux/siren/codeEpci,types commune-actuelle ET
arrondissement-municipal,zones metro/drom/com,JSON. Aucun CRM/code/nom
spécifique envoyé. Tous enregistrements retournés préservés byte-identiques,
aucune sélection suivant cible/public/état/ville de clients ni labels.
Liste serveur pas auditée indépendamment contre COG : limite explicite,
pas claim index national SIRET complet ni récupération de chaque historique.

Motivation : contexte nominal des villes et liens officiels commune–SIREN/
EPCI utilisables plus tard comme joins, pas features d'identifiant, alias
CRM ou identité d'un équipement déduite automatiquement. Le snapshot ne
contient aucune nouvelle recette d'admission/scoring. Tout futur test
nécessite code/config gelés, initial100 avant aval, postseal GT et ressources.

Sources documentaires officielles :
[API communes](https://geo.api.gouv.fr/decoupage-administratif/communes),
[schéma API du producteur](https://github.com/datagouv/api-geo/blob/master/definition.yml).
Ne réclamer que l'observation fournie par la réponse actuelle, pas inventer
millésime COG/date de validité historique ou identité juridique de site.

Worker physique distinct sous2142GiB/80s/quiet30/p1/free50/thread1,timeout
HTTP30,payload16MiBmax. Budget dépassé/structure inconforme =>STOP sans
troncature/retry ou élargissement. Rows sanity30000..50000 pas certificat
indépendant de complétude. Codes uniques et chaînes typées sur TOUS records,
UNKNOWN SIREN/EPCI permis et publié, aucune entité/site exclu par cette source.
Source brute/hash et metadata HTTP immuables, projection fields publiée,
vrai exit0/PASS contrôlé avant seal. Aucun credential/APIpayante/compute
externe/encoder/fit/GT/fold0/fold1/candidat/aval. Mac/CATNAT seulement.
Exécuter entre jobs nationaux terminaux, pas worker concurrent. Ne vaut pas
recall/SLO/fullOOF/généralisation/READY, objectif99,2 exact initial100 inchangé.
