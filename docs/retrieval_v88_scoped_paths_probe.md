# V88 — exécution physique V87 sans changement de métrique

Toutes les règles de `retrieval_v87_scoped_paths_probe.md` restent identiques :
mêmes poids, mots/IDF, portée, grades, XY/BAN,100 distincts, sources nationales,
requêtes et protections. Même hypothèse64 non apprise, pas un tuning qualité.

V87 a STOP sur la première requête après20s sans sortie certifiée ni labels.
699 observations de chemins,497 targets complétés,300 faits/100 native/100XY ;
borne1.752 égale au cutoff, donc pas de certificat sous budget. Pas de recall.

Adaptation physique avant nouvelle mesure : completer par lots100 à travers
TOUS les fanouts d'une page de noms, au lieu d'une requête native aléatoire
par petit fanout. Chaque grade reste vérifié contre SON fait d'origine ;
frontière avancée seulement APRÈS toute la page, donc aucun pool/depth fixe.
Ordre de visite round-robin native puis XY puis noms (avant : noms d'abord).
Ce scheduling ne change ni la métrique ni les ties globales/certificat strict.
Instrumenter temps de flux natif et nombre/temps d'accès natifs aléatoires.

Budgets inchangés20s/requête, moyenne16≤4s, worker80s/6Gio/zéro swap et
supervision90s/contrôleur900s. STOP est conservé, V87 scellée immuable.
Après tests d'oracle exhaustif sur les vrais adaptateurs, préflight16 sans
labels puis128 aveugles si coût passe, évaluateur séparé sinon aucun recall.
Pas READY_FOLD1, pas garantie d'amélioration ni de SLO.
