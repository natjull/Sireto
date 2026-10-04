# V177 — même index complet, budget DuckDB réduit

Protocole avant mesure, 4 octobre 2026. V176 INIT et son premier pack
channel0/bucket0 ont réussi, mais channel1/bucket0 a dépassé le plafond RSS
de 1 Gio (1079443456 octets). Le parent a réellement arrêté son propre
worker ; aucun résultat de cette phase ni index complet V176 n'est accepté.
Les artefacts/code/config V176 restent immuables, aucun retry de STOP.

V177 reprend exactement la validation compacte complète V176, la source
V175 entière, tous les 34346446 aliases et 2440276150 postings, le kernel
natif V134 inchangé, les 256 partitions et toutes ses preuves avant/après.
Seul le budget interne DuckDB passe de 256 à 96 Mio. DuckDB peut davantage
spiller sur disque (plafond inchangé 3 Gio) ; cela ne retire aucune ligne,
aucun token/alias/site/candidat et ne change aucun score/dtype/tri.
Fresh root V177. Première mesure sur le bloc qui a échoué en V176, puis
continuation uniquement si ressources et vraie sortie du processus PASS.

Même politique V153/V164 : 1 Gio RSS, 80 secondes, pression réelle normale,
free >=50 %, swap global télémétrie non attribuée. Mac/CATNAT seulement.
Même autorités scellées, epochs globaux, fullhash des fichiers lus, audits
natifs exhaustifs et final seal de chaque partition complète. Aucun pool
2000/500, aucun label/CRM, aucun fold0/1, aucun modèle aval ouvert.

Ni ce build ni ses tests ne mesurent le recall ou la latence du retrieval.
Complete_retrieval_index et ready_for_fold1 restent false. L'objectif reste
99,2 % exact humain avant les modèles aval et au plus 100 SIRET initiaux.
