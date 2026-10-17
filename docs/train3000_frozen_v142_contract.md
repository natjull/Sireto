# Train3000 — candidats V142 figés, 17 octobre 2026

Autorisation : brief Nathan `sireto-site-target-7.md`. Référence immuable :
`docs/retrieval_frozen_v142.{md,json}`, commit `36c7a476`, variante
`learned_native_ooc`. V173 est rejeté. Aucun apprentissage aval ici.

## Population et affectation

- Uniquement les labels humains exploitables du ledger scellé, projetés et
  filtrés sur les folds 2/3/4 avant matérialisation. Aucun filtre de qualification.
- Exclure toutes les entreprises humaines des 526 dossiers du pilote, pas
  seulement leurs query_id ; conserver et compter chaque motif d'exclusion.
- Vérifier les cinq modèles V71 et leurs cinq bases V32 contre les SHA du gel.
  Affectation interne : SHA256(`sireto-native-calibration-20261012-v1|SIREN`),
  huit premiers octets big-endian modulo 5. Vérifier l'absence du SIREN humain
  dans les entreprises de fit V32 ET V71 du modèle affecté ; sinon exclure/compter.
  Cette partition interne n'est pas un fold population fermé ni un routage produit.
- Si les entrées nécessaires existent : tirer 3000 dossiers, graine 20261017,
  proportionnellement aux strates fold × taille de commune (nombre de sites
  officiels indexés : absent, 1–999, 1000–9999, ≥10000), plus forts restes,
  ordre aléatoire déterministe dans chaque strate. Ne pas substituer au nombre
  de sites le nombre de labels disponibles. Sceller le tirage avant génération.

## Fidélité et gates préalables

Les modèles, coefficients, index, sources et caches du gel restent inchangés.
Les labels servent aux groupes/affectations et à l'évaluation, jamais aux preuves
ni à la génération. Aucune injection, substitution de recherche ou preuve manquante
remplacée par zéro. Les hashes du gel doivent être revérifiés avant génération.
Le préflight publie séparément la couverture des entrées par requête et la
faisabilité hors entreprise. Si les entrées scellées ne couvrent pas les nouvelles
requêtes, **STOP_MISSING_FROZEN_QUERY_INPUTS**, sans faux smoke ni lancement complet.
Interprétation conservatrice de « mêmes caches, hashes que le gel » : pas
d'extension/reconstruction silencieuse des caches liés au pilote. Une adaptation
query-only des préparateurs doit être explicitée avant de prétendre à leur fidélité.

## Exécution et sortie prévues après gates

Namespace neuf `H/v174/train3000_frozen_v142_20261017/`. Un seul calcul lourd,
un thread, reprise par artefacts/reçus immuables ; ne toucher à aucun `*.pending.*`.
Réemployer la supervision V173 : attente 60 s si hydraulic Running/incertaine,
aucune action VM ; 8 Gio par worker, une escalade generate à 12 Gio si libre +
inactive ≥9 Gio et VM arrêtée, arrêt terminal au dépassement ; disque ≥10 Gio.
Smoke réel de 20 dossiers avant lancement complet détaché et commit automatique.
Publier temps/requête, pic RSS et ETA mesurés, jamais extrapoler un smoke non exécuté.
Après sorties et reçus code 0 scellés, évaluation descriptive séparée : exact,
SIREN, union exact ∪ site (règle site existante), rang exact/absent ; absence du
label reste une erreur. Aucun jeu fermé, nouveau reclasseur, label réécrit ou push.
Résultats `docs/train3000_frozen_v142_result.{json,md}` et ligne `handover.md`,
y compris un STOP documenté si un gate préalable échoue.
