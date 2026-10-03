# V147 — toutes les preuves d'adresse accessibles depuis leur site

Noyau de compilation/lecture non mesuré nationalement. Aucun nouveau corpus
national, manifest global, retrieval, recall ou SLO. V142/V144 restent
interrompus ; V145 n'a pas encore de dictionnaire national complet.

## Liaison sans perte

Chaque fait V145 est rattaché à son site explicite, son parent, sa variante
globale d'adresse et ses attributs propres. Le couple bucket dictionnaire /
fact_id conserve l'accès à toutes les observations originales et à leurs
dates, sources, masques et provenance. Ces positions ne sont pas des
features. La jointure SIRET→parent est reconstruite pour CHAQUE fait sur la
namespace V141, sans inventer un site ni dégrader une identité invalide en
contexte.

`stage` partitionne les faits par un quantum physique de positions sites,
pas par pertinence pour une requête. `pack` trie tout un intervalle physique
et produit un CSR site→faits en tableaux typés mmap. Toute colonne et tout
site reconstruisent exactement les entrées, par relecture complète. Les
numéros sont partagés dans un dictionnaire local mais conservés verbatim,
sans deuxième normalisation. Les sources sont hashées avant/après et les
artefacts au chargement. Un intervalle vide garde tous ses sites au CSR.
Aucun plafond de preuves, propriétaires, noms ou historiques.

`ReversePart.batches` accède directement aux plages d'un intervalle de sites
et retourne des `AddressBatch` V146. Même un site ayant2501+ faits garde
TOUTES ses preuves : le chunk est seulement un plafond de RAM temporaire.
Pas de propagation aux frères, pas de tri des preuves par score, ni lookup
de vérité. Les scores de chaque variante entière sont contrôlés une fois
dans `AliasSnapshot` readonly, pas rescannés mondialement à chaque lot de
sites. Le lexique complet qui calculera ces scores n'est PAS fourni ici.

## Tests et portée

Le test intégré source officielle factice de test→V142→V145→V147→V146 vérifie
la parité de sortie initial100 pour différents chunks, sans ancien candidat
ni label. D'autres tests confrontent chaque attribut/provenance, les sites
vides, les numéros Unicode normalisés une fois,2501 faits au même site, les
doublons/absences/alias hors plage, dérives de parent/hashes et l'immutabilité
d'un intervalle terminé. Ces fixtures ne sont JAMAIS un corpus d'évaluation
ou une preuve de recall national. Les scores de variantes y sont contrôlés
manuellement pour vérifier la liaison ; pas prétention de coût lexical réel.

Il manque encore le driver national qui exige TOUTES les parts de projection,
construit le dictionnaire V145 et scelle la totalité des intervalles V147,
ainsi que le lexique d'adresses et le loader query-side V138/V137. Une liste
de parts réussies n'est pas une autorité complète. Les appels à ces noyaux
doivent être supervisés par les budgets versionnés ; aucune garde système
n'est modifiée par V147. Aucun job national lancé dans ce tour.

Prochaine mesure recevable : moteur assemblé sur toutes les autorités,
128 sorties scellées avant vérité, puis OOF all-human complet et gates
99,2% exact / Wilson / généralisation / SLO. Exact/op séparés, erreurs
humaines non retirées du dénominateur, folds2/3/4 seulement, fold0 non lu,
fold1 fermé, modèles aval gelés. Aucun objectif remplacé par les tests verts.
