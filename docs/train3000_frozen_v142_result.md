# Train3000 V142 — STOP préflight, 17 octobre 2026

**Aucun smoke natif ni run3000 lancé. Aucun nouveau candidat.**

Contrat précommité : `6a3e3128a6627ecfbf0235aeb123ea305cdec55f`. Gel : `36c7a476`.

## Faisabilité entreprise, pas admission du cohort final

- Census projeté du ledger folds2/3/4 : 23571 labels exploitables.
- Pilote : 526 requêtes, 405 entreprises humaines.
- Exclus du census par entreprise pilote : 859, dont 333 hors query_id du pilote.
- Restent 22712 dossiers / 15076 entreprises ; coefficients V32 ET V71 hors entreprise : 22712 OK, 0 échec.
- Les dix modèles et les groupes de fit sont vérifiés ; affectation SHA256 SIREN identique au pilote.
- Ce census n'est PAS un tirage de3000 ni une admission de test4323/control962. Les exclusions spécifiques des jeux fermés et la stratification commune restent à sceller avant toute sélection.
- Aucune preuve full-OOF de l'encodeur n'est déduite de cet audit des seuls coefficients.

## Couverture des entrées scellées

| Entrée nécessaire | Requêtes scellées | Couverture hors entreprises pilote |
|---|---:|---:|
| numeric51 | 526 | 0/22712 |
| activity_shards | 526 | 0/22712 |
| bano_leaf_shards | 526 | 0/22712 |
| osm_leaf_shards | 526 | 0/22712 |

La matrice numérique51 et les shards d'activité, BANO et OSM sont requis par
`embeddinggemma2_v142_bano_native_ablation.generate` et par le replay V173.
Ils sont indexés par query_id et liés au panneau526 ; ils ne constituent pas
un cache national de preuves applicable tel quel à toute nouvelle requête.
Une absence de clé n'est pas une preuve de score nul. Aucun fallback lexical,
rattachement à un dossier voisin ou faux heldout n'a été exécuté.

## Hashes et ressources

- 3681 tailles/statuts de fichiers du gel contrôlés ; 58 SHA réellement recalculés (20783985 octets).
- Contrôle complet des SHA des dix modèles, code scellé et manifestes utilisés. Contrôle intégral149Go du gel NON exécuté : gate manquant et VM occupée. Il resterait obligatoire avant génération.
- Préflight réel : 0.88 s ; pic RSS 197.1 Mio.
- hydraulic : `Running` ; aucune action VM. Calcul lourd non lancé.
- Smoke20 : 0/20 ; s/requête, mémoire du smoke et ETA3000 non mesurés. PID superviseur : aucun.
- Labels, modèles, index, caches et pending préexistants inchangés ; aucune évaluation de recall ni apprentissage.

## Blocage et suite nécessaire

Construire/adapter les préparateurs query-only pour de nouvelles requêtes,
avec exactement les mêmes règles et autorités officielles, dans v174 sans
modifier les caches526. Cette matérialisation de nouvelles preuves n'est pas
un fit production ; sa fidélité doit être établie avant de prétendre réutiliser
la recette figée. La clause « mêmes caches, hashes » a été interprétée strictement,
pas comme une autorisation de leur extension silencieuse.
Puis sceller l'exclusion des jeux fermés, les tailles de communes, le tirage3000,
recalculer tous les SHA du gel, exécuter20 vrais natifs sous supervision,
et seulement alors lancer le complet détaché avec évaluation séparée/commit final.

Statut : `STOP_MISSING_FROZEN_QUERY_INPUTS`. CLI : `.venv/bin/python scripts/train3000_frozen_v142.py --status`.
