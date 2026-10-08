# Perplexity contre Gemma — comparatif apparié initial100, 8 octobre 2026

Autorisation « Vas-y check » : mesurer l'apport du modèle brut, pas fine-tuner
ni modifier les modèles aval. Ce DIAGNOSTIC n'est pas le pilote représentatif
Gemma526 ni une garantie de recall99,2%.

Sélection figée query-only avant labels dans censusV15 : une commune par
strate [1,1000),[1000,10000),[20000,50000) sites, au moins3 requêtes ; nombre
de requêtes décroissant, hash `pplx-paired-v1|INSEE` pour ex-aequo. Toutes
requêtes de ces communes et toutes requêtes sans commune conservées. Aucun
choix selon résultats, noms, présenceGT ou coût après tirage. Ce plan maximise
l'information par coût, il N'EST PAS probabiliste/représentatif.

Résultat query-only :39510 (4queries/567sites),66048 (6/1146),71076 (14/29348),
commune inconnue (3/0). TOTAL27requêtes/31061établissements officiels complets,
actifs/fermés conservés. La zone29348 dépasse l'ancien plafond20k mais reste
loin des très grandes villes ; aucune extrapolation nationale de ce panel.

Comparer : Gemma2SITE512best1000 et Perplexity0,6B brut révision
8fc2de24534aa3610d85fa59c463313a5f096455. LITTÉRALEMENT les mêmes vuesSITE512
officielles produites par packingGemma, pas un champ appauvri pour Perplexity.
Prompts/tokenizer natifs Perplexity, MaxSim token à token puis MAX par vues du
site, toutes les références complètes scorées, collecteur direct100, aucun
poolGemma reranké. Stop si dépassement limites1024query/4096doc plutôt que
troncature silencieuse. Pas de mélange identité parent, aliasCRM, identifiants
features, expansionBODACC/filiations, positif injecté ou filtrequalifications.

Trois pipelines : Gemma+lexicalnational+BAN/XY ; Perplexity+ces mêmes canaux ;
familleGemma/Perplexity avec MAX de leurs contributionsRRF (pas double vote)
puis lexical/géo identiques. RRF60 fixe, géo1/.5faible, cap100 absolu. Dense
seul, lexical seul et géo seul également publiés. Aucune grille après labels.

Scope propre `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v14/perplexity_paired_20261008_a`
compatible workersV14 immuables. Reprise uniquement caches scellés/bindings
identiques, retries ressources réelles avec owned exits et STOP sans progrès.
Réutiliser Gemma/documents deV15 seulement complets/hashés/ownedexit0. Sinon
préparation complète et référenceGemma dans ce nouveau scope, jamais modifier
le runV15 ou ses artefacts. Si zone Gemma actuellement active, attendre plutôt
que lancer le même calcul simultanément. ConcurrenceGPU peut ralentir les
deux calculs ; pas de promesse « aucun impact débit ».

Perplexity float32/eager/batch4 MPS sans fallbackCPU ; overlayimage séparé,
runtimeGemma inchangé. Déduptextes et shards128 de vecteurs float32 sans pertes,
MaxSim CPU borné par chunks, matrices finalquery/sites intégrales. Admission
pression0–2/free30, yieldfree20/driver6Gio/RSS8Gio ; superviseur ne signale
QUE son enfant Perplexity si free15/pressionanormale/RSS9Gio. Sauvegardes/exit75
coopératifs, pas délai arbitraire pour la totalité du retrieval.

Avant labels : toutes les sorties27 de tous canaux + communes COMPLÈTES,
préparations/générations ownedexit0 et sceau final. Puis labelsvalidation234
hashés et sidecarofficiel opérationnel vérifié. Publier exact/SIREN/opérationnel,
counts/denominator27, Wilson DESCRIPTIF seulement, segmentscommune/fold,
gains/pertes appariés contre pipelineGemma. Petitpanel géocorrélé, population
ayant servi au choix checkpoint, pas OOF/national/SLO/pristine/READY_FOR_FOLD1.
Fold0/1/test4323 restent fermés. Pas de score partiel deV15 présenté comme
sa validation complète. RecettesV15hashées inchangées ; aucun ajustement selon
ces27 résultats dans un run actif. Chaque milestone commité au handover.

Première tentativeA STOP avant chargement du modèle : import de l'utilitaire
géo exigeait DuckDB dans le runtime GPU, alors que celui-ci n'en a pas besoin.
Correction d'import différé, nouvelle exécutionB à
`v14/perplexity_paired_20261008_b`. Même sélection/données/poids/recettes ;
A et son exit1 conservés, aucune dépendance/runtime active modifiée.
