# V3 — attribution données/poids et union initial100

Prédeclare8octobre2026, avant les métriques. Namespace CATNAT/v3 séparé des
V1/V2. Validation2 212 issue du split SIREN immuable ; test4 323/folds0/1fermés.
Ce cohort development n'est pas une preuve OOF nationale indépendante et des
contrôles ont déjà choisi les adapters. Le ratio local ne vaut pas gate99,2%.

Communes triées hash(seed|INSEE), minimum3requêtes CRM,100–20 000sites,
communes complètes acceptées tant que total<=120 000, arrêt dès>=128requêtes.
Si moins128après tous les comptes, publier le vrai nombre, pas rajouter des
communes à partir des ratés. Préparer uniquement les champs CRM requis.
Actifs/fermés et tous les distracteurs présents dans l'index natif ; GT hors
commune reste dans le dénominateur et perdu au canal dense. Le canal lexical
cherche le corpus NATIONAL43896818SIRET sans filtre dur de commune.

TOUS les sites énumérés reçoivent le même enrichissement officielV130,64buckets
facts/names fullhashés avant utilisation. Nom légal parent + aliases de CE site,
pas nom de sibling comme enseigne. Les noms de siblings appartiennent uniquement
aux vues identité du parent. Tokens512packés sans coupure silencieuse.

Variantes figées :

- V1-native160 : recette dense du diagnostic initial, même checkpoint7200,
  troncature160historique explicitement comptée.
- V1-official512 : poids V1, vues officielles nouvelles, score exact exhaustif
  .75maxsim(querynom+adresse,vuesSITE)+.25maxsim(querynom,vuesIDENTITÉparent).
- V3-official512 : mêmes vues et même score, nouveaux poids retenus sur validation.
- Lexical-national : requête initial100V38 multichamp nom/adresse/bonusgéo souple.
- Union pour chacune des3variantes denses et lexical : RRFk60, poids égaux,
  plafond final100 après déduplication. Canaux indépendants<=100, aucune grille
  postrésultat, modèle ranker, pool2000/500 ni hydratation truth-conditioned.

Le score parent/site est une hypothèse de retrieval à tester, pas calibré par
les résultats de cette cohort. Une simple amélioration de loss ne valide rien.
Comparaison appariée même corpus/requêtes pour attribuer effets textes et poids.
Ce canal dense est encore partitionné par commune et sans XY/BAN numérique ;
la fusion avec un lexical national ne rend pas le dense lui-même national.

Prepare peut tourner pendant le minageGPU (tokenizer CPU seulement). Le watcher
attend l'exit réel du fit puis génère les3variantes en processus successifs.
Adapter/hash du launch de fit vérifiés ; baseline si meilleuradapterstep0 à
publier comme absence de gain. Chaque commune terminée est scellée et réutilisable
dans la même recette. Une génération interrompue n'est PAS évaluée partiellement.
Budget partagé de la nuit, reprise resource-only30s, pas watchdog25min.

Avant lireGT : toute population et trois générations complètes, owned exits0,
outputs fusionnés scellés, sidecar officiel politique hashés. Ensuite exact,
SIREN, opérationnel séparés, intervallesWilson95, gains/pertes, folds, label
historique/V2/V3/origine/état et tous GT hors partition. Champs inconnus publiés
UNKNOWN. Pertes structurelles doivent être lues avant toute nouvelle recette.
Pas de résultat SOTA garanti, fold1 fermé, serving/aval inchangés, aucunpush,
dépense ou transfert externe. Incertitude froid/chaud et SLO nonvalidé publiées.
