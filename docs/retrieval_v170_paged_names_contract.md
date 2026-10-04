# V170 — fenêtres physiques plus larges, mêmes preuves exactes

Protocole avant mesure, 4 octobre 2026. Contrat métier/protections V38 et
invariants exacts V169 inchangés. Aval gelé, folds0/1 fermés, aucun GT/hit
ancien ouvert. Aucune garantie de99,2%, aucun READY ou SLO.

V169 session85292 TERMINAL1, QUERY0 STOP_PAGED_NAME_QUERY_COST pendant
scatter. Pic940621824 (~897Mio), worker41,121805s/min73/pression[1]/global0,
garde ressources153 PASS mais requête20s NON complète. Aucun manifest,
initial100, recall ou validation mémoire de la requête entière. Scratch et
trace conservés, root169/code/config mesurés immuables. L'arrêt montre une
limite de coût, pas un dépassement1Gio ou des données manquantes.

V170 ne change que le quantum physique :524288 au lieu de131072, pour
réduire les fermetures/réouvertures et les re-faults d'updates file-backed.
Cela permet plus de références dans une même opération, PAS moins de
références au total. Toutes les variantes/postings/faits/patches/relations
et leurs positions restent identiques, divisions/maximum/OR float64 natifs.
Tests bit-exacts au même oracle y compris quantum524288. Gardes1Gio/80s,
query20s, scratch3Gio et pression normale/free>=50 INCHANGÉES.

Instrumentation de phases seulement : temps/RSS après ouverture, lexical,
CHAQUE bucket owner, scatter et un-hop. Aucune preuve CRM/GT/SIRET publiée
dans ce journal. Les checkpoints ne sont pas des certificats de succès.
Si encore STOP, diagnostiquer la phase et versionner, jamais contourner la
garde ou relancer le même root. Consommer tous les44403034sites et fullhash
avant/après restent nécessaires au PASS coût, qui ne vaut pas recall.

Ordre : tests puis root frais, exactement la même première requête query-only
V115 et mêmes sourcesV134/V149 nationales complètes. Pas de sélection de
requête selon les résultats. Les fichiers query-owned réussis seuls sont
nettoyés, ceux d'un STOP conservés. Reprise adresse16814381 séparée.
