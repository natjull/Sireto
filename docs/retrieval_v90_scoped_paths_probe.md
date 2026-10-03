# V90 — clé scalaire du type du tableau, aucun changement de métier

Même métrique/grades64/noms/portées/XY/BAN/cap/scopes/queries que V87,
collecteur guidé par bornes V89 et pages256 inchangés. Remplacer seulement
les recherches binaires avec int Python par la même clé scalaire du dtype
exact du tableau mmap (uint64 pour SIRET/owners, uint32 pour parents graphe).
Validation ASCII14/9 et range intacte, IDs toujours jointures PAS features.

Diagnostic source-only16 positions réparties sur chaque tableau, alternance
Python-int/typed-scalar, mêmes résultats :12531976 endpoints p50 3.029ms vs
3.229µs ;6087062 parent_owner_keys p50 1.465ms vs2.500µs. Ceci montre un coût
de dtype caché des accès, PAS un facteur garanti de gain end-to-end ni SLO.
V89 avait7 recherches certifiées puis STOP worker80s à81.016s, pic4559814656,
zéro swapout, pendant recherche binaire : pas de recall partiel/labels.

Tests de positions/provenances/fanout/scopes contre adaptateur ancien,
assertion que chaque appel binaire reçoit un scalaire du dtype exact ; oracle
exhaustif initial100 avec weak/missing XY et natif gelé. Pas de nouveaux poids.
Code/config commités avant run, V87/V88/V89 et logs immuables. Même préflight
16 coût≤4s,20s query,80s/6Gio worker/zéro swap,90s superviseur,900s contrôleur.
128 aveugles puis évaluateur distinct seulement si gate passe. Aval gelé,
folds0/1 fermés, aucune dépense externe, pas READY_FOLD1 sur un pilote.
