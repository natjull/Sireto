# E2 — Accepteur V4 exact-SIRET

## Verdict

**`GO_HOLDOUT_V4`**

Sur les 189 scènes de la moitié `threshold` du nouveau dev :

- `AUTO_MATCH` : **149/189 = 78,836 %** ;
- bonnes décisions parmi les AUTO : **149/149 = 100 % observé** ;
- erreurs AUTO : **0** ;
- gate préenregistré : au moins 25 % de couverture à 99,8 % observé.

Les 149 AUTO sont tous des `MATCH_EXACT` correctement classés. L’accepteur
rejette :

- les 31 scènes `AMBIGUOUS` ;
- les quatre erreurs exact-SIRET du ranker présentes dans cette moitié ;
- cinq scènes exactes pourtant correctes mais jugées insuffisamment sûres.

Le gate est donc franchi sans exception métier ajoutée après lecture du dev.

## Portée statistique

Zéro erreur sur 149 AUTO ne garantit pas une précision de 99,8 % en
production. L’intervalle Wilson bilatéral donne :

- borne basse 95 % : 97,487 % ;
- borne basse 99 % : 95,737 %.

Le résultat autorise l’ouverture unique du holdout scellé. Il ne constitue pas
encore une certification ni une autorisation de déploiement.

## Variantes

Les six variantes atteignent exactement le même point à 99,8 % observé :

| Variante | AUTO | Couverture | Erreurs | Précision |
|---|---:|---:|---:|---:|
| Logistique + brut | 149/189 | 78,836 % | 0 | 100 % |
| Logistique + sigmoid | 149/189 | 78,836 % | 0 | 100 % |
| Logistique + isotonic | 149/189 | 78,836 % | 0 | 100 % |
| XGBoost + brut | 149/189 | 78,836 % | 0 | 100 % |
| XGBoost + sigmoid | 149/189 | 78,836 % | 0 | 100 % |
| XGBoost + isotonic | 149/189 | 78,836 % | 0 | 100 % |

Le winner `logistic_scaled__isotonic` résulte uniquement du tie-break
déterministe prévu dans le code. Cette expérience ne démontre pas que
l’isotonic est supérieur aux autres variantes.

Bundle gelé :

- modèle : régression logistique standardisée ;
- calibration : isotonic ;
- seuil : `1.0` ;
- dataset : `2b8a9c994e0944be` ;
- contrat : commit `9a22fd8`.

## Dataset

- train : 5 749 `MATCH_EXACT` + 1 108 `AMBIGUOUS` = 6 857 scènes ;
- dev : 305 `MATCH_EXACT` + 53 `AMBIGUOUS` = 358 scènes ;
- candidats : 721 007 paires ;
- 100 candidats maximum ;
- aucun doublon ;
- exactement un positif pour chaque scène exacte ;
- zéro positif pour chaque scène ambiguë ;
- `UNRESOLVED` entièrement exclu ;
- holdout et ancien test non lus.

Les scènes exactes train proviennent exclusivement des prédictions OOF du
ranker. Les ambiguës n’ont jamais participé au fit du ranker et sont scorées
comme `out_of_sample_ambiguous`.

## Artefacts

- Contrat : `docs/v4_acceptor_e2_contract.md`, commit `9a22fd8`.
- Préparation ambiguë : `scripts/prepare_v4_ambiguous_retrieval.py`, commit
  `af5ce0b`.
- Builder et garde OOF : commit `9ec88c8`.
- Dataset :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/acceptor_v4/2b8a9c994e0944be/`
- Bundle :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8/`
- Suite complète : 145 tests passants.

## Étape suivante

Avant toute lecture du holdout :

1. publier les hashes du dataset, du ranker, de l’accepteur, du calibrateur et
   du seuil ;
2. préenregistrer le runner final et ses métriques ;
3. vérifier une dernière fois la disjonction SIREN ;
4. ouvrir `holdout_sealed` une seule fois ;
5. publier couverture, précision SIRET exacte, Hit@1, erreurs et intervalles ;
6. conclure `GO`, `PIVOT` ou `STOP`.
