# V162 — suffixe communal : apport POI confirmé, recall inchangé

Une seule ablation native pré-déclarée au commit `7efcaaac`, sur les mêmes
526 requêtes que V142. Grammaire V161 figée avant labels : seul libellé
communal officiel entier en suffixe contigu, non vide après retrait ; ancien
matching complet OU matching décomposé, aucun autre nom/ville/prefixe ajouté.

Root : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v162/commune_suffix_native_20261014_night`.
Préparation ALL526 session `38910`, chunk `c8c40b`, exit réel 0 en 56,917 s.
Run session `94209`, chunk final `e377c6`, exit réel 0 ; enfant natif `93211`,
owned exit 0, pic RSS 9 299 509 248 octets, sous son plafond 10 Gio. Le slot
natif précédent EXPLAIN était arrêté et sa fin owned/ps confirmée avant lancement.

Toutes les 526 sorties, au plus 100 SIRET uniques chacune, sont scellées
avant l'ouverture des labels. SHA outputs :
`fd99e98b5e6caa353600dc87f146f46d74f3fa6516a6e04e59a6890b5322690f`.
Reçu de completion tool sous `owned_tool_exit.json`, owned child sous
`commune_suffix_union/generate_exit.json`.

## Résultat mesuré, apparié à V142

| Métrique | V142 | V162 | Gains / pertes |
|---|---:|---:|---:|
| SIRET humain exact | 507/526 | 507/526 | 0 / 0 |
| SIREN humain | 511/526 | 511/526 | 0 / 0 |
| Équivalence opérationnelle | 508/526 | 508/526 | 0 / 0 |

Recall exact brut **96,3878 %**, pondéré **95,6790 %** ; Wilson 95 % brut
**94,4275–97,6755 %**. Les 19 ratés actuels demeurent inchangés, sans label
corrigé, filtre de dénominateur ni ouverture de jeu fermé. Les segments
folds/zones/strates/origines et gains/pertes sont disponibles dans le report.
Les champs génériques `gained/lost` du report hérité comparent une ancienne
baseline ; l'attribution de cette ablation est uniquement `paired_vs_v142`.

La préparation query-only a identifié exactement **11 requêtes avec scores
feuille réellement différents**, issues des 13 nouvelles apparitions de POI
du préflight. Les 515 autres sorties V142 sont réutilisées seulement après
preuve d'égalité exacte des clés ordonnées, scores feuilles complets,
coefficients/modèle, requête complète et contextes RNA/BANATIC. Aucun hit ou
label n'a choisi la réutilisation.

Après seal et évaluation, on constate que les 11 requêtes modifiées étaient
**déjà toutes des hits exacts**, et le restent toutes. L'amélioration du
matching communautaire porte donc ici sur une couverture déjà assurée par
les autres preuves, pas sur un des 19 cas absents. Ce constat ne permet pas
de sélectionner maintenant d'autres villes, règles de suppression ou poids.

## Invariants

- Nom CRM complet inchangé pour BM25, dense, adresses et tous les anciens
  canaux ; aucune nouvelle étape de training, ranker ou decider.
- Même masque public eO/uO, mêmes XY propres canoniques et même univers
  numérique ; aucun nouveau bonus pour les receveurs sans XY compatibles.
- Nouveau bonus nommé faible exactement `w[6] × .5/(1+d/300)` ; MAX avec le
  score feuille complet V142, pas une addition de votes du même POI.
- Aucun nouveau score parent ni coordonnées/adresse propagées ; contextes
  RNA/BANATIC identiques. Les POI/opérateurs OSM ne deviennent pas des alias
  officiels, des probabilités d'identité ou des preuves de numéro de maison.
- Union des preuves avant un seul collecteur national initial100 ; aucun
  pool2000/500, fusion de deux top100 ou réévaluation aval.

## Conclusion

L'hypothèse générique de formulation était réelle, et son essai natif est
maintenant mesuré ; **elle ne réduit pas les ratés de ce panel**. Conserver
les artefacts et ne pas lancer une grille de variantes autour de ces résultats.

Mesure de développement consommée, encodeur non full-OOF, sources actuelles
communautaires et sources héritées post-cutoff : aucune certification
nationale 99,2 %, indépendance ou promotion production. Les timings incluent
une forte réutilisation d'entrées équivalentes et ne valident pas un nouveau SLO.
