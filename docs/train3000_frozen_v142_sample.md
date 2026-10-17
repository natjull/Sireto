# Tirage train3000 V142 — 17 octobre 2026

Mandat : `sireto-site-target-8.md`. Ce mandat autorise explicitement à
matérialiser des preuves query-only dans v174 avec les constructeurs, sources,
versions et paramètres historiques : ce n'est pas un changement de recherche.
Il remplace l'interprétation restrictive « pas de nouveaux caches » de
`docs/train3000_frozen_v142_contract.md`, sans modifier le gel V142 ni le préflight.
Aucun des caches526 n'est étendu ou réécrit.

## Population et custody des jeux fermés

- Ledger scellé, filtres folds2/3/4 avant matérialisation ; aucun filtre de statut.
- 23 571 query_id ; exclusion de4323 tests par complément des trois autorités
  autorisées : train15036, panel query-only de validation2212 et panel2000
  exclu de l'encodeur. Aucun fichier `test_membership.json` ouvert.
- Contrôle962 reconstitué selon la règle V130 : validation moins entreprises
  pilote et contrôles de checkpoint ancêtre64/actuel256. Les IDs des contrôles
  consommés sont reproduits avec les hashes historiques, puis seuls leurs SIREN
  sont projetés. Le filtre entreprise est poussé dans Arrow avant retour de
  données ; aucun SIRET/SIREN du contrôle962 n'est retourné au constructeur.
- Les hashes des ensembles d'exclusion sont publiés dans le JSON, pas leurs labels.
- Exclusion des405 entreprises pilote :859 dossiers. Population restante :
  **17 427 dossiers / 11 483 entreprises**, folds2/3/4 =5763/6028/5636.
- Affectation SIREN historique ; absence du SIREN dans les hashes de fit des
  DEUX modèles V32/V71 affectés : **17 427 OK, 0 échec**. Ce n'est pas full-OOF
  encodeur ; beaucoup d'entreprises appartiennent à son entraînement consommé.

## Tirage avant preuves et résultats

Graine `20261017`, strates fold × taille de commune, allocations proportionnelles
aux plus forts restes, permutation SHA256(graine|strate|query_id), sans inspection
de résultats. Taille = nombre exact de documents SIRET de la commune dans le
même index Tantivy national, par le `zone_query` historique, pas nombre de labels.
Comptage seul, un thread, aucune hydratation de document/score/encodeur ; limite
metadata1Gio. Le VM peut continuer pendant ce census léger, pas pendant les
préparations ni recherches lourdes.

Bandes : code commune absent ; code présent et <1000 sites ; 1000–9999 ; ≥10000.
Le nom de champ `1_999` désigne ici le deuxième groupe : il conserve aussi les
5 codes présents sans document indexé (1 dossier de ce type tiré). Rien n'est
supprimé ou réparé ; les comptes exacts sont conservés par ligne. Cette précision
résout le cas0 non explicité dans les bandes du contrat initial.

- **3000** dossiers : folds2/3/4 =**993/1037/970**, **1233** codes commune.
- Modèles internes0/1/2/3/4 :611/563/579/618/629 dossiers.
- Hash de l'ensemble trié de query_id :
  `f2a359afb834b02d5ca30d905808b62ea2d613eb038acbb9eaa1c82e8df2deb3`.
- 20 pilotes de fidélité et20 nouveaux dossiers smoke choisis par la même
  méthode stratifiée, graines suffixées `|fidelity` et `|smoke`, AVANT les preuves.
- Smoke/fidélité ne sont pas encore exécutés par ce seul milestone de tirage.
- Census/tirage réels : **8,30s, 422,4Mio RSS maximal**. Ce n'est pas un coût
  de préparation/génération et ne permet aucune ETA des3000.

Namespace `H/v174/train3000_frozen_v142_20261017/extension_20261017/` :
`queries.json` ne contient QUE les8 colonnes CRM historiques ; `assignment.json`
ne contient que le modèle OOF affecté ; `label_side_sample.json` reste séparé,
interdit comme entrée des constructeurs, à réserver à l'évaluation après sceau.
`sample.json` et `docs/train3000_frozen_v142_sample.json` portent le même tirage,
les quotas, les autorités et les20+20 IDs ; publication sans écrasement.

## Gates suivants inchangés

Même recette/coefficient/source ; fidélité des quatre familles sur les mêmes20
pilotes, puis vrai smoke20 préparation ET native100. Pas de full3000 avant ETA
mesurée ≤30h ; sinon proposer une taille pour15h. Aucune ETA basée sur ce census.
Workers lourds : VM Running/incertaine → attendre60s, ne jamais contrôler la VM ;
un worker/un thread ; RSS8Gio, une escalade generate12Gio avec ≥9Gio libre+inactive
et VM arrêtée ; stop disque CATNAT<10Gio. Aucun label/jeu fermé/web/fit/push.
Le scellement SHA complet du gel reste obligatoire avant recherche. Le générateur
complet et son commit final restent un jalon distinct, non réalisés par ce script.

Vérification réelle : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train3000_frozen_v142_cohort.py --seal`.
Consultation : `.venv/bin/python scripts/train3000_frozen_v142_cohort.py --status`.
