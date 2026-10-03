# V145 — factorisation des adresses, sans changement de retrieval

Nouveau code non mesuré nationalement. Ce module ne lit ni CRM ni labels,
ne recherche pas de candidats et ne produit aucun recall. Il ne remplace
pas le contrat de recherche initiale V38, ni les autorités V142/V144.

## Objet conservé

Un dictionnaire par bucket physique partage chaque **texte complet**
normalisé une fois. Le hash choisit seulement le stockage : deux textes
distincts ne sont jamais fusionnés même en cas de collision. Le fait conserve
adresse, site, parent, CP, INSEE et numéro. Plusieurs observations du même
fait restent toutes accessibles par leurs identifiants globaux.

`current_exact` est un OR des seuls booléens explicites des observations du
même fait. Les flags source ternaires, dates, sources et valeurs originales
restent dans les autorités liées ; ce OR n'invente ni date de fermeture ni
validité temporelle. Les adresses parent, masquées ou vides restent dans la
source, sans propagation vers des établissements et sans retrait du corpus.
Les positions servent aux jointures, pas aux features.

L'export est contrôlé sur chaque observation admissible : unicité, texte
complet, site/parent, CP/INSEE/numéro, flags agrégés et reconstruction des
liens. Aucun plafond de noms, sources, périodes, faits ou propriétaires.
DuckDB doit être versionné dans tout futur lancement : son hash de stockage
n'est pas supposé stable entre versions.

## Statut et suite

Tests unitaires seulement ; aucun manifest national V145, aucune nouvelle
mesure de rappel. Une projection nationale complète et scellée, l'inventaire
exact de toutes ses parts et les hashes de sources restent des prérequis à
une construction nationale. Les lots réussis d'une construction interrompue
ne valent pas cette autorité complète.

La prochaine intégration devra combiner preuves lexicales de noms/adresses,
géographie souple et relations officielles dans le collecteur initial100.
Pas de pool fixe union→LightGBM sous un nouveau nom. Toute optimisation
d'accès doit conserver les preuves et démontrer l'équivalence du top100.
Le dense est une hypothèse complémentaire à tester pour ses gains uniques,
pas une promesse99% ni un encodage national déclenché par ce module.

Schémas locaux vérifiés : les stocks historiques INSEE contiennent les
anciens noms/enseignes, PAS les anciennes adresses. RNE/BODACC apportent des
preuves d'adresses distinctes avec leur portée. Les relations BODACC présentes
sont ASSET_TRANSFER SIREN→SIREN : ne pas les rebaptiser TUP/absorption ou
succession certaine. Aucun alias appris du CRM ; aucune suppression de label
jugé bruyant dans le dénominateur exact. Exact et opérationnel séparés,
folds2/3/4 seulement, fold0 non lu, fold1 fermé, modèles aval gelés.

## Références, pas garanties locales

[Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
montre l'intérêt d'une baseline lexicale top-k en entity matching ; cela
ne prouve ni notre recall ni notre latence sur le Mac.
[Multilingual E5](https://arxiv.org/abs/2402.05672) décrit des tailles
small/base/large et un entraînement contrastif ; cela ne démontre pas qu'un
petit modèle adapté sur nos paires battra un modèle généraliste ici.
