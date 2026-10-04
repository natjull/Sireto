# V175 — sceller tous les lots sans monolithe mémoire/IO

Protocole fixé avant mesure,4octobre2026. V168 remaining-parts est terminé,
49imports V166 réussis +527lots propres, mais aucun staging/index complet
n'est encore scellé. Ne pas confondre cette présence avec une validation.

V175 réutilise chaque lot PASS sans recalcul ni import d'un STOP. Sources
fullINIT V168/V163/V160 strictes, mêmes34346446wholealiases/98804872facts/
44403034sites/317683879observations/148075748lexicalobservations. Chaque
intervalle d'alias doit être contigu, exhaustif et unique dans son ordre
officiel ; aucune sélection par requête, GT, score ou candidat.

INIT fixe hashes et epochs de TOUS les fichiers/manifest/supervision/preflight
des576lots. Ce ledger n'est pas un certificat complet. Ensuite36batches
physiques de16lots : phase_record168/166 original (hash intégral de chaque
artifact, statut PASS, vrai exit0 et politique153), champs/proofs exacts,
feature schema/IDs/range/rows, TOUS les fichiers de postings reread (channel
columns) et compteurs reconstruits. Aucun cap de posting, token, alias,
owner ou candidat. Rehash intégral des mêmes artifacts après le comptage.
Source epochs et tous les liens source/implémentation demeurent inchangés.

Chaque batch worker/parent supervisé V153/V164 :1Gio/80s, pression réelle1/
free>=50%, préflight1s (INIT30s), swap global télémétrie non attribuée. Fresh
root, aucune reprise d'un batch STOP. Les lots sources et anciens résultats
mesurés ne sont jamais modifiés. Aucun purge/autreapplication tuée.

Après TOUS les batches, réauditer le prefix et reverse source entier via les
deux enfants168 seal (80s chacun, hashes complets et vrais exits). Les tags
seal de168 sont frais ; ne pas lancer son seal_parts/256packs/ancienremaining.
Combiner les36certificats sous garde153, vérifier source/audit/part refs,
comptages globaux et epochs deTOUS les artifacts depuisINIT jusqu'àla fin.
Les hashes complets before/after de chaque artifact sont acquis par son
batch, pas réinterprétés àpartir d'epochs non certifiés. Refuser trous,
doublons, fichier modifié, STOP, source différente ou compteur incohérent.

Sortie SEALED_V175_ALL_WHOLE_ADDRESS_CHANNEL_POSTINGS, mêmes features et
buckets que seal_parts168. Pas encore CSR lexical, query snapshot national,
initial100, recall ou SLO ; complete_retrieval_index/ready_for_fold1 false.
Elle permettra ensuite une exécution directe exacte174 sur TOUT le corpus
ou les mêmes packs134 avec audit par fichier pertinent plutôt que rehash
global répété256fois. GT/CRM/ancienscandidats/folds0/1/aval restent fermés.
Le dernier recall exact114/128 est inchangé,99,2% non prouvé.
