# V4.12 — Ranker augmenté par les labels difficiles

Date : 31 juillet 2026  
Périmètre : développement consommé uniquement. Le test final et le holdout V4-Fresh restent fermés. Aucun modèle n'est autorisé en production.

## Méthode

Le corpus historique V4.11 `fit` est conservé. Les 77 nouveaux labels SIRET exacts sont ajoutés avec un poids de groupe testé parmi `1,0`, `0,5`, `0,25` et `0,1`.

Pour chacun des cinq folds gelés V4.11 :

- le modèle voit le `fit` historique et seulement les nouveaux dossiers appartenant aux quatre autres folds ;
- aucun SIREN des dossiers difficiles n'existe dans le `fit` historique ;
- les deux vérités absentes du pool restent des erreurs end-to-end et ne sont jamais injectées ;
- les six dossiers `AMBIGUOUS` sont exclus du ranking et de l'écran de régression ;
- le modèle est évalué sur les dossiers difficiles du fold qu'il n'a pas vus.

Un modèle final est ensuite entraîné sur le `fit` et les 75 nouveaux labels dont le positif est réellement présent. Son seul usage ici est un écran de régression sur 1 197 requêtes `dev` exactes, hors des 83 dossiers adjudiqués et hors de leurs composantes SIREN.

## Ablation de pondération

| Poids des nouveaux cas | Bons top 1 difficiles | Corrections | Régressions difficiles | Bons contrôles | Régressions contrôle | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| baseline gelé | 18/77 | — | — | 1 197/1 197 | — | référence |
| 1,0 | 60/77 | 44 | 2 | 1 194/1 197 | 3 | rejet |
| **0,5** | **59/77** | **43** | **2** | **1 197/1 197** | **0** | candidat retenu |
| 0,25 | 58/77 | 42 | 2 | 1 197/1 197 | 0 | dominé par 0,5 |
| 0,1 | 57/77 | 40 | 1 | 1 197/1 197 | 0 | moins de gain |

Les scores difficiles incluent les deux misses retrieval. Sur les 75 dossiers où la vérité est présente, le candidat `0,5` place donc le bon SIRET en premier dans 59 cas.

## Audit des bascules défavorables

Les deux régressions difficiles à poids `0,5` sont les contre-exemples déjà identifiés :

- `IDEF 86` : le rôle de siège ne doit pas écraser le site explicitement associé à l'adresse ;
- `CCI EMERAINVILLE` : le successeur institutionnel ne doit pas remplacer le SIRET exact publié pour le CFA UTEC.

À poids plein, l'écran historique signalait également deux requêtes dupliquées de l'Observatoire régional de la santé à Saint-Benoît et une requête du CH d'Aunay. L'audit externe confirme :

- pour l'ORS, `82418653000021` reste bien l'établissement de l'adresse ; ce sont deux vraies régressions du même cas métier ;
- pour le CH d'Aunay, le label historique est périmé après fusion. L'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/etablissement/26140092300320) donne désormais `26140092300320` pour le site hospitalier. Ni le baseline ni le candidat à poids plein ne proposaient ce SIRET.

La pondération `0,5` élimine ces trois bascules sur l'écran historique sans réduire fortement le gain difficile.

## Reproductibilité

Artefact principal :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_hard_label_ranker/bba02575366ebe80`

Réplique indépendante :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_hard_label_ranker_replica/bba02575366ebe80`

Les prédictions OOF, l'écran de régression et le modèle final sont bit-à-bit identiques entre les deux exécutions. Hash du modèle candidat :

`45f8735382111ee3dc308926bd4883f2c71601cb9e30be72ebb76eba36fd62cd`

## Décision

Verdict : **`GO_NEW_INDEPENDENT_VALIDATION`**.

Le poids `0,5` est figé comme candidat de développement. Le résultat ne prouve pas la North Star et n'autorise aucun déploiement : les labels ont servi à choisir la variante. La prochaine preuve doit venir d'un nouveau lot de dossiers `REVIEW` non adjudiqués, gelé avant lecture des vérités, en priorité parmi les cas où ce candidat et le ranker V4.12 ne sont pas d'accord.

Reproduction :

```bash
python3 scripts/evaluate_v412_hard_label_ranker.py --hard-weight 0.5
```
