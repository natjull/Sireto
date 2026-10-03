# V143 — Reprise d'adresses sans retrait de données ni garde affaiblie

V142 a STOP au part officiel74 :64 pages swapout global après1s,742,8Mo
RSS. Session91768 TERMINAL1 vérifiée. Ce n'est pas un timeout ou un résultat
national. Conserver V142 code/config/docs/plan/75 succès et part74 STOP
immuables. Ne pas reprendre CE root ni importer export interrompu.

Nouveau root et nouvelle autorité V143. Réutiliser exactement les75 parts
V142 réussis après vérification entière de chaque artefact/manifeste/plan/
launch/ressources/protections. Les sources originales sont encore entièrement
hashées avant la reprise et à la fin. Chaque champ/scope/ID original et
part planifié V142 reste identique. Aucun changement de projection V142.

Seule optimisation du worker : batch8192 au lieu de32768. Même corpus,
normalisation, champs, rôles, ordre/identité de preuve et audits complets ;
pas de plafond de nombre de sites, observations, alias, membres ou sources.
Différences physiques possibles des row groups et de compression, pas des
preuves. Tests bit-exacts entre les deux découpages. Source entière317683879
observations, pas seulement les75 parts réutilisés.

Avant init :30s de stabilité du compteur système swapout et au moins50%
selon memory_pressure -Q ; avant chaque worker :1s de stabilité. Gate
read-only borné55s, sinon STOP sans lancer de worker. Compteur ne doit pas
avoir changé entre la fin du gate et l'ouverture du budget worker. Aucune
app fermée/cache vidé/autre processus tué. Ce gate réduit un risque, ne
garantit pas l'absence d'activité concurrente pendant le worker.

Budgets workers V142 INCHANGÉS :80s internes,90s superviseur,3GioRSS,
ZÉRO swapout global,512Mio/part,32Gio total. Les gates sont publiés à part,
pas dissimulés dans la mesure du worker. Aucun retry automatique d'un part
STOP, même root, ni hausse de seuil. Contrôleur s'arrête au premier défaut.
Toute phase nouvelle doit exporter et relire entièrement son source-part.

Seal final exige tous312 parts importés/nouveaux une fois, dénominateur
source/proofs exact, toutes sorties hashées, sources entièrement rehashées.
Manifeste explicite chaque autorité importée ou construite. Pas de CRM/GT/
fold0/fold1/fit/aval/recall/SLO/READY/push/dépense. Après source complète,
dédup whole-variants/owners puis accès lexical et collecteur initial100.
