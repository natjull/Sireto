# V60 — vraie recherche initiale100 avec les mots officiels appris V59

Préenregistrement avant sorties V60. Paramètres V59 figés : tous folds
choisissent ridge10/alpha2 uniquement par validation interne par entités.
Les contrôles externes de tête, nombre de concurrents et marge progressent
sur les3 folds ; cela justifie un pilote, ne vaut PAS recall national.

Exporter bundle serveur sans labels, scènes ou contrôles : poids natifs11,
3 projections384x2048, vocabulaire officiel2048, encoder local V52 hashé,
max128 tokens/prefixnom exactement V52, sorties sparse<=8 mots pondérés.
Aucun refit, réencodage des sites, index nouveau ou donnée V58 issue GT.
Encodeur CPU2 threads, seules128 requêtes CRM autorisées en génération.

Score initial = même score natif V57 + alpha*maximum de couverture des
mots appris sur5 champs de noms, cette composante supplémentaire uniquement
en commune/CP. Ce gate n'affecte pas la recette de base globale. Un
collecteur natif national émet<=100 SIRET directement, pas de réservoir,
reranker, propagation d'alias CRM, feature d'identifiant ou filtre d'état.
Index full43896818 sites, proofs adresses/numéro/site V57 conservées.
XY/BAN, liens temporels et catégorie juridique pas encore intégrés dans
cette ablation : pas prétention à architecture complète/SOTA/objectif.

Population128 panel v38/V42/V49/V57 réutilisé, pas holdout inédit,
disjoint512 train. Premier préflight16 aveugle : moyenne primaire<=4s
expérimentale PAS SLO. Worker80s/6Gio/zéro swap, superviseur90s,
contrôleur360s y compris encodage128, coût individuel20s contrôlé après
retour natif (préemption dure seulement superviseur90s), deux threads.
STOP immuable sans labels si garde échoue, aucune relance élargie.

Évaluer séparément uniquement après128 scellées : exact/opérationnel,
all-human, Wilson, historique/V2/V3, folds, gains/pertes face V42 scellé
ET comparaison supplémentaire à V57/V49 afin de ne pas exagérer un gain.
Latences chargées sans BAN, encodage publié séparément et inclus total ;
SLO non validés automatiquement. Visites de postings non instrumentées.
Gros segments index size/mtime seulement, pas nouveau rehash intégral.
Fold0 non lu/fold1 fermé/aval gelé/aucune dépense. Pas READY_FOR_FOLD1.
