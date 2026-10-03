# V146 — intégration du score cohérent et collecteur initial100

Code non mesuré nationalement ; pas de nouvelle politique de ressources,
construction nationale, recall ni SLO. Les essais précédents sont immuables.

## Score query-side

`NameSnapshot` combine maxima de variantes ENTIÈRES parent, site et noms
one-hop V137, dans la namespace V141 inchangée. Le nom propre à un site ne
remonte jamais au parent ni aux frères ; la relation n'apporte aucun exact
courant. Les nouveaux propriétaires sans noms explicites restent UNKNOWN,
avec les preuves directes du parent s'il est connu.

`AppendedAttributes` impose la même autorité V136 entre attributs V139 et
append V141. Les nouveaux sites restent UNKNOWN pour numéro, CP/INSEE, siège,
activité et XY : ni interpolation ni héritage de frère/successeur.

Le score initial natif30 V121/V114 reste non négatif. Il calcule un score
complet pour chaque preuve d'adresse avec SON texte, SON numéro et SES
CP/INSEE, puis prend le maximum par site. Il ne combine pas une ancienne
adresse lexicale avec le numéro d'une autre observation. Exact adresse est
alias_exact ET current_exact du même fait. Un fallback indépendant des
adresses utilise les attributs courants explicites. Géométrie et activité
restent celles du site actuel, sans être déclarées historiques ou transférées
par relation. Un code/XY inconnu n'est pas une exclusion. Une preuve sans
couverture lexicale peut encore porter numéro/commune officiels : elle n'est
pas supprimée en raison de son seul score lexical nul.

C'est une NOUVELLE hypothèse de fusion cohérente, pas une assertion de parité
métier avec les anciens scores qui n'avaient qu'un couple CP/numéro par site.
Les tests scalaires confrontent son algèbre aux primitives initiales30.
Aucun poids entraîné dans ce volet, aucun modèle aval appelé/modifié.

## Collecteur, pas reranker

`search` traverse la namespace complète par chunks physiques, score toutes
les feuilles et fournit directement `collect` initial100. Tout trou, doublon
de position, score non fini/négatif, namespace différente ou timeout impose
STOP sans résultat partiel. Les identités sont reconstruites depuis le
catalogue pour sortie/dédup/égalité de score, jamais comme features.

Les collecteurs locaux top-k fusionnent le MÊME score complet. Leur parité
avec un tri global est testée, y compris les égalités entre segments append
non triés mondialement. Ce n'est pas un pool fixe ensuite reclassé : aucun
2000/500, changement de score ou modèle intermédiaire. Heap final<=100,
pas de plafond de preuves, owners, membres ou observations. Ce parcours
complet est aussi un oracle pour de futures bornes sûres ; il n'a PAS encore
la latence SLO et ne prétend pas éviter le scoring de44M feuilles.

## Ce qui manque pour une mesure nationale recevable

Les interfaces testées ne certifient pas les sources d'un appelant. Le loader
national devra vérifier toutes les autorités/hashes/scopes et fournir TOUS
les faits d'adresses par site via `batches(ids,query)`, avec lookup lexical
par variante entière. V145 n'a pas encore de manifest national ; V144 est
STOP112 et la projection complète317683879 n'est pas scellée. Un faux flag
`complete` ou un sous-corpus de test ne vaut jamais preuve nationale.

Il reste à terminer les adresses, construire l'accès lexical/reverse-site,
brancher les noms nationaux V138/graphes officiels et décoder les features
CRM/activité selon les modèles OOF autorisés. Le parcours complet permettra
de contrôler toute optimisation ultérieure d'accès/bornes contre le même
top100, sans filtre dur géographique ou d'état. Puis sceller les128 sorties
avant vérité, publier exact/op séparés, et franchir les gates OOF all-human,
Wilson/généralisation/SLO du contrat actif. Aucun READY ou99% revendiqué.

Fold0 non lu ; fold1 fermé ; folds2/3/4 seuls ; pas d'alias CRM, injection
de positif ou retrait d'erreur humaine du dénominateur. Pas de dépense,
nouveau run Mac national ou adaptation de zéro swap global dans V146.
