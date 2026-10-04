# V180 — même preuve structurée V139, cartes propres libérées par chunk

Protocole avant mesure, 4 octobre 2026. Le collecteur direct146 parcourt
tous les44m sites, mais conserver chaque page mmap de CP/INSEE/NUM/HQ/NAF/
XY jusqu'à la fin peut excéder1Gio. Ce wrapper change uniquement la durée
de vie physique de ses propres cartes readonly, pas les données ou scores.

Constructeur StructuredSites139 INCHANGÉ (sources/seal/hashes/types/audit),
mêmes flags/coordinates139. Capture les epochs/dtypes/shapes et refs des
10 tableaux et dictionnaires/labels officiels effectivement consommés.
Avant/après chaque flags/coordinates : toute la source reste inchangée.
Les sorties doivent posséder leur mémoire ; puis seules les cartes de ce
reader qui viennent d'être lues sont fermées et réouvertes. Les autres sont
toujours accessibles, pas de purge globale, de filtrage ou d'approximation.

Même numéro normalisé139, code public5char/corse/DOM, même HQ/NAF ouUNKNOWN,
XYsameCRS/NaN ouUNKNOWN. Les sitesfermés/extra restent accessibles. Aucun
héritage de coordonnées/activité/site frères, pas de réduction de corpus.
verify_after rehash tous ses tableaux et métadonnées/dictionnaires/labels.
Tout échec invalide le reader, jamais publication d'une source partielle.

Tests de parité avec139 et score146 sur une source tiny complète ; ce n'est
pas un benchmark national, une preuve1Gio ou un nouveau recall. Toute
mesure réelle doit rester sous V153/V1641Gio/80s/pression1/free>=50% et sur
Mac/CATNAT, sans dépense. Aucune donnée CRM/GT, fold0/1, downstream changé,
candidat ou résultat initial100 produit par ce seul wrapper. Le gate
initial10099,2% exact/Wilson/OOF/généralisation/SLO reste à franchir.
