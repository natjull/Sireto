# Résultats E2b — score de confiance exact-SIRET

## Verdict

- verdict contractuel : **`STOP_E2B`** ;
- meilleur score : régression logistique standardisée, probabilité brute ;
- couverture à 99,8 % observé : **85/1 280 = 6,641 %** ;
- erreurs parmi ces 85 AUTO : **0** ;
- minimum pré-enregistré : **25 %** ;
- test final lu : **non**.

E2b améliore le point E2 isotonic, qui ne couvrait que 33/1 280 = 2,578 %.
Cette amélioration ne suffit pas à valider l'accepteur.

## Expérience gelée avant calcul

Le contrat `docs/downstream_acceptor_e2b_contract.md` a été commité avant le
run sous `cf91432`. Le code exécuté est `070c123`.

Les candidats, le ranker, les 80 features de scène, les labels, la seed et les
deux moitiés dev sont restés inchangés. Seules trois transformations du score
ont été comparées : brute, sigmoid et isotonic.

## Résultats

| Variante | Scores distincts | AUTO à 99,0 % | Précision | AUTO à 99,8 % | Précision |
|---|---:|---:|---:|---:|---:|
| Logistique + brut | 1 280 | 249 (19,453 %) | 99,197 % | **85 (6,641 %)** | **100 %** |
| Logistique + sigmoid | 1 280 | 249 (19,453 %) | 99,197 % | 85 (6,641 %) | 100 % |
| Logistique + isotonic | 40 | 26 (2,031 %) | 100 % | 26 (2,031 %) | 100 % |
| XGBoost + brut | 1 279 | 204 (15,937 %) | 99,020 % | 77 (6,016 %) | 100 % |
| XGBoost + sigmoid | 1 279 | 204 (15,937 %) | 99,020 % | 77 (6,016 %) | 100 % |
| XGBoost + isotonic | 52 | 33 (2,578 %) | 100 % | 33 (2,578 %) | 100 % |

La transformation sigmoid conserve l'ordre du score brut et produit donc le
même point de couverture. Isotonic regroupe les requêtes en gros paliers et
réduit fortement la couverture possible. Le problème E2 venait donc en partie
de la calibration, mais pas seulement.

## Le blocage principal est maintenant le benchmark

À 25 % de couverture, il faut automatiser 320 des 1 280 requêtes de la moitié
dev réservée au seuil. Avec une cible de 99,8 %, une seule erreur sur 320 fait
déjà tomber la précision à 99,688 %. Le point doit donc avoir zéro erreur sur
ce petit échantillon.

Les 320 premiers scores logistiques contiennent cinq « erreurs » selon les
labels. L'examen du CRM et du snapshot SIRENE montre que ces cinq cas sont
tous des conflits de vérité terrain, pas des erreurs évidentes du modèle :

| Requête | CRM | SIRET prédit | Label actuel | Constat local |
|---|---|---|---|---|
| 14355 | VISSELECT SARL, 9 rue Henri Becquerel | `62820158400024` | `52381510800015` | le prédit est VISSELECT actif à l'adresse exacte ; le label est AVENIS fermé à la même adresse |
| 16826 | IMD OPTIQUE, 20 avenue Hélène Boucher | `82213655200020` | `UNRESOLVED` | le prédit est IMD OPTIQUE actif à l'adresse exacte |
| 2446 | SCI AVOCATS DU PLATEAU, 161 rue André Bisiaux | `51518433100020` | `31714680100053` | le prédit porte le nom CRM exact à l'adresse exacte ; le label est TERTIO AVOCATS |
| 11265 | LMP Santé, 260 rue du Puech Radier | `75394095600018` | `78877503900019` | le prédit est LMP SANTE actif ; le label est une ancienne entité LMP SANTE fermée |
| 10353 | PGDIS Dardilly, 1 chemin de l'Industrie | `90032220700011` | `40225443700369` | le prédit est PGDIS LYON actif à cette adresse ; le label est OFFICE DEPOT FRANCE |

Les champs `reference_date` des requêtes sont vides. Le benchmark ne peut donc
pas dire s'il faut retrouver l'occupant historique ou l'entité active du
snapshot. Il pénalise aussi toute prédiction sur `UNRESOLVED`, même lorsqu'un
SIRET actif reprend exactement le nom et l'adresse CRM.

Si ces cinq conflits étaient tous validés en faveur du SIRET prédit, les
320 premiers scores auraient zéro erreur. C'est un contre-factuel de
diagnostic, pas une métrique revendiquée : aucune vérité terrain n'est
modifiée dans E2b.

## Conclusion d'architecture

Le prochain chantier n'est pas un nouveau modèle :

1. le retrieval fournit déjà 99,572 % de Recall@100 sur le dev V3 exact ;
2. le nouveau ranker gagne 2,804 points sur l'ancien ;
3. le score brut sépare mieux les cas sûrs que la calibration isotonic ;
4. mais la cible d'apprentissage mélange des erreurs exact-SIRET, des dossiers
   sans date et des `UNRESOLVED` traités comme de vrais négatifs.

La trajectoire devient donc **`PIVOT_DATASET_AVAL`** :

- fixer une politique temporelle unique, « SIRET actif au snapshot » ou
  « SIRET à la date du CRM » ;
- si la date du CRM est absente et que plusieurs vérités temporelles sont
  possibles, classer le dossier `AMBIGUOUS`, pas `MATCH_EXACT` ;
- ne plus apprendre que `UNRESOLVED = faux match` : un label inconnu n'est pas
  un négatif ;
- conserver comme négatifs les mauvais top-1 face à un SIRET exact validé,
  ainsi que les vrais `NO_MATCH` et `AMBIGUOUS` validés ;
- construire un nouveau jeu indépendant, sans utiliser les scores du modèle
  pour qualifier les labels, puis seulement réentraîner l'accepteur.

Cette correction ne demande ni GPU, ni dense, ni retour au risk model
historique.

## Artefact

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_e2b_3171ef5020c0f068_070c123`

Le bundle sauvegardé indique explicitement :

- modèle : `logistic_scaled` ;
- calibration : `raw` ;
- seuil : `0.9736660945392409` ;
- bundle : `209c472df8423db8` ;
- dataset : `3171ef5020c0f068`.
