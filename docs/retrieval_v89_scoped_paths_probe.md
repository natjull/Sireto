# V89 — traversée guidée par les bornes, même métrique V87

Toutes les règles/scopes/scores/protections de V87 sont inchangés et hashés.
V88 batching certifie une première requête7.48s, mais STOP sur la suivante
à20.56s :1192 targets,1299 observations,300 native/300 noms/300XY,2 passages
natifs8.19s et10 accès ponctuels2.36s. Aucun recall partiel ni labels ouverts.

Avant mesure, changer seulement le plan physique : première page256 (taille
de lecture, pas pool/fanout/depth plafond) puis pages adaptatives sans limite
de profondeur. Si heap100 est rempli, ne pas lire un flux dont la borne
unseen est STRICTEMENT sous son cutoff : natif si frontier_native<cutoff,
noms/XY si64×frontier_name×frontier_site<cutoff. Égalité reste active pour
ties globales. Aucun target retiré du corpus ni chemin tronqué. Ce test
n'abaisse pas le64 hypothétique pour faire artificiellement passer le coût.

Comparer les sorties à un oracle exhaustif sur many-to-many/adaptateurs
réels/ties/closed/missing/weakBAN et gardeown-fact. Budgets inchangés20s,
moyenne16≤4s, worker80s/6Gio/zéro swap, supervision90s et contrôleur900s.
16 sans labels→128 aveugles uniquement si coût passe→évaluateur distinct.
V87/V88 et leurs STOP restent immuables. Jamais READY_FOLD1 sur ce pilote.
