# Contre-audit de quatre labels de contrôle V4.12

Date : 13 août 2026  
Statut : corrections proposées, non appliquées au dataset canonique  
Périmètre : développement consommé uniquement ; aucun test final ouvert

## Verdict

Les quatre erreurs attribuées au reranker neuronal à `alpha=0,75` ne sont pas
des régressions métier. Dans chacun des quatre dossiers, la prédiction
alternative correspond à l'identité décrite par le nom et l'adresse CRM,
tandis que le label historique pointe vers une autre personne morale.

| Requête | CRM | Label historique | Vérité corrigée | Preuve déterminante | Fiabilité |
|---|---|---:|---:|---|---|
| `10420` | LEVAC, 12 avenue Lionel-Terray, Meyzieu | `39187532500014` | `97150121800095` | Les [mentions légales LEVAC](https://www.levac.fr/mentions-legales/) publient ce SIRET et l'adresse CRM exacte. | Haute |
| `12633` | NETWORK HOLDING, rue de Roubaix, Tourcoing | `51942934400020` | `82113394900015` | La [fiche SIRET](https://annuaire-entreprises.data.gouv.fr/etablissement/82113394900015) et les [données RNE publiées par Pappers](https://www.pappers.fr/entreprise/network-holding-821133949) donnent la raison sociale exacte et le 43 rue de Roubaix. L'ancien label désigne K.E. au numéro 38. | Haute |
| `fresh:FR027494` | STOCK J BOUTIQUE JENNYFER, rue Étienne-Dolet, Saint-Ouen | `92264921500014` | `33888018003730` | La [fiche SIRET](https://annuaire-entreprises.data.gouv.fr/etablissement/33888018003730) et l'[avis RNE](https://entreprises.lefigaro.fr/stock-j-boutique-jennyfer-92/entreprise-338880180) portent la raison sociale CRM exacte et l'adresse rue Étienne-Dolet. | Haute |
| `fresh:FR037625` | XAVIER MAITRE ET GUILLAUME LAGUE NOTAIRES, 15 rue Saint-Honoré, Fontainebleau | `39155877200029` | `32671645300021` | L'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/etablissement/32671645300021) donne la raison sociale et le SIRET exacts ; le [site officiel de l'office](https://maitre-lague-fontainebleau.notaires.fr/contact.htm) confirme les deux notaires et l'adresse. | Haute |

Les identités, activités et adresses ont également été recoupées dans les
snapshots locaux `StockEtablissement_utf8.parquet` et
`StockUniteLegale_utf8.parquet`.

## Effet sur l'évaluation

Le recalcul ci-dessous remplace ces quatre vérités uniquement dans une vue
d'analyse en mémoire. Il ne modifie ni `labels.parquet`, ni le jeu canonique.

| Variante | REVIEW fiables | Contrôles corrigés | Ensemble |
|---|---:|---:|---:|
| Ranker actuel | 216 / 254 (85,04 %) | 1 123 / 1 127 (99,65 %) | 1 339 / 1 381 (96,96 %) |
| Business ranker ciblé | 222 / 254 (87,40 %) | 1 126 / 1 127 (99,91 %) | 1 348 / 1 381 (97,61 %) |
| Petit cross-encoder, `alpha=0,75` | 222 / 254 (87,40 %) | 1 127 / 1 127 (100,00 %) | 1 349 / 1 381 (97,68 %) |
| Gate CE `alpha=0,75`, alternative limitée au rang 2 du ranker | 222 / 254 (87,40 %) | 1 127 / 1 127 (100,00 %) | 1 349 / 1 381 (97,68 %) |
| Gate précédent + accord business ciblé/CE `alpha=3` | 224 / 254 (88,19 %) | 1 127 / 1 127 (100,00 %) | 1 351 / 1 381 (97,83 %) |
| Ensemble retenu : gate précédent + accord business/BGE avec marge | 225 / 254 (88,58 %) | 1 127 / 1 127 (100,00 %) | 1 352 / 1 381 (97,90 %) |

L'ensemble retenu effectue treize corrections nettes par rapport à la vérité
corrigée du ranker actuel : neuf dans les dossiers REVIEW fiables et les quatre
contrôles ci-dessus, sans régression observée. Ces résultats servent à choisir
une hypothèse de développement ; ils ne constituent pas une validation
indépendante, car ce dev set a déjà été consommé.

## Règle conditionnelle candidate

1. Conserver le top-1 du ranker actuel par défaut.
2. Le remplacer par le choix CE mélangé à `alpha=0,75` uniquement lorsque ce
   choix est le rang 2 du ranker initial. Sur les données observées, ce filtre
   enlève les deux changements « mauvais vers autre mauvais » de rang 3.
3. Ajouter une seconde voie très étroite : remplacer le top-1 lorsque le
   business ranker ciblé et BGE mélangé à `alpha=10` désignent exactement le
   rang 2, et que le score BGE brut de cette alternative dépasse celui du
   top-1 d'au moins `0,004`. Cette voie ajoute `ALCEANE`, `PHB CREATION` et
   `NEMERA LYON`. Le seuil exclut le changement inutile du dossier
   `CLINIQUE DE TOURNAN`, dont la vérité est hors du top 20.

L'artefact exécutable est
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f`.
Il contient les décisions et les 27 620 candidats du top 20 reclassés pour les
254 requêtes fiables et les 1 127 contrôles. Les scores top-1/top-2 sont
échangés de façon déterministe lorsque le gate change la décision.

Le gate doit rester expérimental jusqu'à une nouvelle validation hors des
requêtes utilisées pour le concevoir.
