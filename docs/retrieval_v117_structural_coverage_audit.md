# V117 — Couverture structurelle avant nouveau moteur

Audit du 3 octobre 2026, sans requêtes CRM ni labels. Il confronte la
proposition utilisateur historique / variantes / géo / lexical / dense /
graphe aux artefacts réellement disponibles. Aucun nouveau recall mesuré,
aucun changement de contrat ni de modèle aval, aucune compilation nationale.

## Constats vérifiés

1. Le moteur principal
   `indexes/hierarchical_retrieval_v3/4b332899d8698354` contient 43 896 818
   documents établissements et 82 772 691 documents au total. Son manifeste
   reprend `temporal_complete=true` de la projection
   `sources/siren_dossier_retrieval_v3_full/manifest.json`, SHA256
   `1df67665f69e39242be72fecbfbb1b92654b99c3d3f0124e4caaa3c37296b7f4`.
   Mais les comptes officiels de cette projection sont seulement
   SIRENE_CURRENT=125 551 755, RNE=81 640 996, BODACC=98 133 358.
   SIRENE_HISTORY et SIRENE_SUCCESSION ne sont pas recensés. V100 confirme
   ces trois seules familles dans son dictionnaire des sources de noms.
   **Le drapeau est contradictoire avec les familles déclarées**, pas une
   preuve de couverture de tous les historiques. Les artefacts historiques
   restent inchangés ; le contrôle V117 refuse cette inférence.
2. De l'historique existe ailleurs : V75 a ajouté 273 796 observations
   d'anciens noms de sites ; V76 a ajouté 4 025 541 observations de noms
   légaux anciens via égalité exacte. Leurs contrats documentent les limites
   amont (déduplication, exclusion des noms actuels, géographie courante,
   parents sans INSEE, etc.). V84 et V96 fournissent successions et membres
   nationaux. Donc ne pas conclure « aucun historique dans le projet ».
   Il manque une intégration complète et homogène de ces preuves dans la
   recette retenue ; V115/V116 déclarent notamment les chemins relationnels
   non inclus, et ne réunissent pas toutes les sources V101.
3. La projection SIRENE courante de `src/xgb_matcher/siren_dossier.py`
   extrait dénomination, usuelles1–3, sigle, nom et nom d'usage ainsi que
   enseignes1–3 et nom usuel d'établissement. Elle **n'extrait pas les
   prénoms** ni le pseudonyme de l'UL. Le stock courant possède
   `prenomUsuelUniteLegale`, `prenom1UniteLegale` et `pseudonymeUniteLegale`.
   Contrôle borné : les 65 536 premières lignes physiques du stock UL
   contiennent 20 213 lignes `statutDiffusionUniteLegale='O'` avec nom et
   prénom non vides (20 213 avec prénom usuel), et 208 lignes d'autre statut
   ou statut inconnu. Requête mono-thread/512Mo, 0,240860s. Échantillon non
   représentatif, aucun identifiant ou nom individuel publié, aucun lien
   CRM/GT effectué. C'est une disponibilité de preuve, pas un gain mesuré.
4. Les schémas physiques des stocks historiques donnent 71 355 318 lignes
   UL et 95 865 102 lignes établissements. **L'historique établissements ne
   contient ni voie, ni numéro, ni CP, ni coordonnées** ; l'historique UL
   ne contient pas les prénoms/sigle courants. Ne pas attribuer aux stocks
   historiques des champs absents. Une adresse courante jointe à un ancien
   nom reste un scaffold actuel, pas une ancienne adresse prouvée.
5. Le builder natif indexe les mots des noms historiques/supporting mais
   seulement les n-grammes des noms courants ; même asymétrie pour les
   adresses. Les champs d'égalité des noms sont courants uniquement.
   La projection historique du moteur principal utilise en outre des caps
   de variantes (historique6/site et12/parent, supporting4/site et6/parent,
   etc., `config/siren_name_portfolio_v1.json`). V94–V100 lèvent ces caps
   pour leur source de noms complète, mais pas pour les champs déjà bâtis
   du moteur natif, ni pour des familles absentes de cette source complète.
6. V43 utilise déjà des coordonnées officielles, conserve les fermés et
   laisse les sites sans coordonnées dans le corpus lexical. Son périmètre
   spatial est Lambert93 métropolitain vérifié, pas tous les territoires.
   V114 étudie activité × lieu/XY ; V115/V116 sont arrêtés au coût avant
   évaluation. Ni une meilleure géométrie ni ces contrôles TRAIN ne prouvent
   un nouveau recall@100.

Lecture légère : petits manifestes, code et métadonnées Parquet ; pas de
rehash intégral de plusieurs Go, pas d'audit exhaustif des valeurs sources
ou du statut de diffusion national. Les comptes historiques sont ceux des
footers, pas des comptes de variantes exploitables.

## Orientation retenue, sous le contrat initial100 existant

La priorité est **la couverture et la cohérence de l'objet recherché**, pas
le remplacement immédiat du modèle dense. Conserver noms uniques normalisés
avec postings vers sujets officiels et observations (source, rôle, période,
diffusion), puis jointures vers SIRET sans cap arbitraire de membres. Ne pas
matérialiser un énorme produit variantesUL × tousSIRET ; dédupliquer les
chaînes, pas supprimer leur portée/provenance. Les tokens d'un nom ne doivent
pas fabriquer une preuve en mélangeant plusieurs variantes distinctes.

Ordre concret :

1. Prototype source-only borné de projection sans perte : variantes EI
   nom+prénom et nomUsage+prénom à partir des champs réellement diffusibles ;
   historique nom et période sans adresse inventée ; provenance et portée
   SIREN/site conservées. Contrôler les champs absents, masqués et UNKNOWN.
   Préserver les observations derrière chaque nom dédupliqué, sans choisir
   seulement sa dernière période. Fixtures de régression avant compilation.
2. Tester coût/volume sur partitions officielles préfixées indépendantes
   des CRM/labels ; vérifier égalité des preuves avant/après déduplication,
   aucun plafond d'alias/fanout, aucun site fermé/non-localisé supprimé.
   Arrêter au budget ; aucune reconstruction nationale tant que l'estimation
   de coût et l'intégrité de ces contrôles ne sont pas satisfaisantes.
3. Ajouter recherche lexicale words + char3/4 sur toutes les variantes et
   recherche géographique confidence-aware. Rayon/département doivent être
   des voies de recherche supplémentaires, pas des filtres irréversibles
   en cas de mauvais lieu CRM/BAN. Un chemin non localisé/global reste ouvert.
4. Joindre les membres et liens officiels pendant la recherche : une
   succession/cession est une hypothèse de continuité, pas une identité
   exacte automatique. Un nom d'enseigne de site ne devient pas l'alias de
   tous les sites frères. Les absorptions/TUP textuelles exigent une source,
   une extraction vérifiée et une portée avant d'être utilisables.
5. Dense comme canal complémentaire : réutiliser d'abord les vecteurs locaux
   compatibles ; changer/fine-tuner l'encodeur seulement si une ablation
   TRAIN/OOF justifie ce coût. Les «40k paires» ne sont pas automatiquement
   un ensemble TRAIN admissible ; exclusions entités/folds restent obligatoires.
6. Comparer pertes, SIRET/SIREN, canaux et combinaison sous cap100, puis
   validation OOF complète, généralisation géographique/entités, Wilson et
   SLO. Un plafond oracle d'union éventuellement diagnostique ne vaut pas
   la métrique produit. Labels ouverts uniquement après scellement des
   sorties ; bruit audité séparément, aucun nettoyage pour améliorer le
   dénominateur ni correction contaminant le développement.

La fusion union→LightGBM proposée nécessite une décision utilisateur car
elle change le contrat initial100 avant reranker. Question asynchrone posée ;
**aucun changement implicite**. Le dictionnaire d'abréviations miné depuis
les CRM annotés est également incompatible avec l'interdiction actuelle
des alias CRM. La normalisation réversible et les variantes établies par
sources officielles restent possibles ; garder aussi la chaîne originale
pour ne pas effacer un nom significatif ou une forme juridique distinctive.

Les promesses «quelques ms», «quelques heures MPS» ou «petit modèle meilleur»
sont des hypothèses, pas des performances démontrées sur ce corpus/Mac.
Pas de dépense externe ni LLM ajouté au retrieval. Le ranker, decider et
modèle de risque existants restent gelés.

## Exécution et verdict

`tests/test_retrieval_source_coverage_v117.py` : 10 tests PASS, 0,03s.
Contrôle réel du manifeste : `DECLARATION_CONTRADICTED`, historique et
successions manquants dans ses comptes. Même cinq familles présentes ne
prouveraient pas l'exhaustivité des périodes/champs/variantes.

Verdict : lacunes structurelles vérifiées, nouveau chantier justifié,
**aucun gain de recall annoncé**. Dernier recall exact mesuré :
V101=114/128=89,0625%, pilote réutilisé uniquement. Aucun label/fold0/fold1
ouvert dans cet audit, aucun job long lancé, objectif99,2% non atteint.
