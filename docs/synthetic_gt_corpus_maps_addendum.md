# Avenant préenregistré — branche `MAPS_ASSISTED`

Date de gel : 15 août 2026  
Statut : préparée, désactivée par défaut ; aucun appel exécuté par la
préparation de l'avenant.

## 1. Séparation et autorisation

`MAPS_ASSISTED` est une branche expérimentale séparée de
`SIRENE_SYNTHETIC`. Ses sorties ont une racine, un manifest, un
`source_kind` et une colonne `maps_query_version` propres. Elles ne sont
jamais mélangées aux lignes SIRENE, à une validation, à un dev gate ou au test
final. Seules les lignes `EXACT_HIGH_CONFIDENCE` peuvent devenir des couples
d'entraînement ; `SILVER_AMBIGUOUS` et `REJECTED` restent des audits séparés.

La branche est inactive tant que le plan ne contient pas un budget explicite
strictement positif, un quota explicite, une coupe-circuit active et une
autorisation d'exécution. Le défaut du plan est `enabled=false`,
`max_requests=0` et `max_cost_eur=0.0`. L'absence de budget ne bloque pas la
branche SIRENE.

## 2. Secret et préflight

Le seul nom de variable secret autorisé est
`SIRETO_GOOGLE_MAPS_API_KEY`. Le code lit exclusivement cette variable via
`os.environ.get`; il ne cherche jamais une clé dans un fichier, Git,
l'historique, les logs, un argument, un notebook ou une autre variable.

Le préflight :

- renvoie `NOT_CONFIGURED` si la variable est absente ou vide ;
- vérifie seulement présence, longueur minimale et budget/quota ;
- ne retourne ni n'imprime la valeur, un préfixe, un suffixe ou un hash de la
  clé ;
- masque toute exception HTTP et tout message d'erreur susceptible de
  contenir un header secret ;
- ne fait aucun appel réseau.

Le smoke n'est autorisé qu'avec une option d'exécution explicite, après
préflight `READY`, budget/quota non nuls et vérification des sources train.

## 3. Entrée et requête

Chaque requête part d'un SIRET déjà connu dans SIRENE et de ses champs
officiels nom, enseigne/dénomination et adresse. Aucun seed hors train n'est
admis. La requête est versionnée et son digest ne contient aucun secret.

Le protocole préenregistré utilise Text Search (New), avec le masque minimal
suivant :

```text
places.id,
places.displayName,
places.formattedAddress,
places.addressComponents
```

Aucun wildcard, photo, avis, note, prix, URL, téléphone, site web ou contenu
non nécessaire n'est demandé. La réponse est réduite aux champs autorisés :
`place_id`, nom affiché, adresse formatée, composants d'adresse, provenance,
horodatage UTC et `maps_query_version`. Le JSON brut complet n'est pas
conservé.

L'endpoint et le masque sont des constantes de code/plan ; les retries sont
bornés (au plus deux tentatives par requête), avec backoff déterministe. Le
cache local est séparé, borné et identifié par le digest de requête, sans
conserver le secret.

## 4. Validation indépendante des résultats

Le premier résultat n'est jamais accepté par position. Tous les résultats
conservés dans la limite du smoke sont comparés au seed SIRENE par des gardes
déterministes :

1. commune et CP concordants ;
2. numéro de voie et voie concordants lorsque présents ;
3. nom officiel, dénomination ou enseigne suffisamment concordant ;
4. une seule correspondance locale plausible ;
5. aucun sibling du même SIREN mieux soutenu par ces mêmes signaux ;
6. aucune contradiction d'état ou d'adresse SIRENE non expliquée.

Les résultats sont classés :

- `EXACT_HIGH_CONFIDENCE` : toutes les gardes fortes passent et une seule
  place est admissible ;
- `SILVER_AMBIGUOUS` : plusieurs places ou une garde non décisive ;
- `REJECTED` : garde forte manquante, contradiction ou résultat non
  identifiable.

Les ratios réponse, rejet, ambiguïté, exactité et garde par famille sont
publiés avant toute extension. Aucun résultat Maps ne peut devenir une
validation ni une vérité d'évaluation.

## 5. Quotas, coût et coupe-circuit

Avant chaque appel, le runner vérifie atomiquement : quota quotidien,
`max_requests`, coût estimé cumulé, budget maximal, nombre de retries et
espace cache. Il s'arrête avant tout dépassement. Une estimation de coût
inconnue ou une valeur de coût non épinglée produit `STOP_MAPS_BUDGET`, sans
appel.

Le smoke est très petit et préenregistré ; le rapport publie le nombre de
seeds proposés, appels cache/non-cache, réponses, erreurs, retries, rejet,
ambiguïté, exactité et coût estimé. Aucun dépassement silencieux, retry
illimité, parallélisme non borné ou prolongation opportuniste n'est permis.

## 6. Publication et verdict

Les artefacts vivent sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/maps_assisted/<build_id>/`.
Le manifest ferme le plan, code, masque, version de requête, hashes SIRENE,
assignments, cache, quotas, budget, provenance, réponses réduites et
classification.

`GO_MAPS_SMOKE` signifie uniquement que le smoke est intègre et auditable ;
il ne certifie ni la qualité de production ni une licence au-delà de celle
attestée dans le contrat utilisateur. `PIVOT_MAPS` signifie que le protocole
est sain mais que la réponse, l'ambiguïté ou les gardes sont insuffisantes.
`STOP_MAPS` couvre une fuite de secret, une dépense non bornée, un dépassement
de quota, une source hors train, un résultat non traçable, une garde contournée
ou une erreur d'intégrité.
