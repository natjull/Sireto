# V281 : trois snapshots régionaux officiels, SOURCE seulement

Figé avant acquisition. Tables régions COG INSEE2015(préréforme),2016
(transition),2026(actuel), tous27/18/18 records et champs originaux conservés,
pas dict issu CRM ni source filtrée parclients/GT. Endpoints officiels vus
dans les pages primaires, schémas inspectés SOURCE seulement avantprotocole :

- https://www.insee.fr/fr/information/2560698
- https://www.insee.fr/fr/information/2114819
- https://www.insee.fr/fr/information/8740222

Deux ZIP latin1/tabs singlemember reg2015.txt/reg2016.txt ;2026 CSVutf8,
headers stricts/counts/IDs catégoriels contrôlés, aucun fichier ZIP extrait
dans filesystem. Origins byte-identiques+headers Date/ETag/LastModified ou
UNKNOWN, SHAentier ; tous champs persistés sans sélection.63observations
trois années, PAS histoire exhaustive ni succession/équivalence territoriale
certifiée ni catalogue entreprises. Les codes sont contrôle/joins seulement.

GET fixes publics sans CRM/GT/credential/coûtcompute externe. Payload2MiB
guard STOP pas troncature ; exactendpointHTTP200 sansredirect/retry,
producer2142GiB80squiet30p1free50thread1, séquentiel après autre producer
terminal. STOPsansretry/partielréutilisé/hausse. Sourceonly aucun modèle,
candidat, recall/SLO/READY. Futur retrieval nécessite recette indépendante
figée avant mesure allhuman exact/op100, aucune identité revendiquée par
simple région homonyme. Corpus actuel/historique complémentaire, pas GT.
