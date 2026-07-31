# V4.12 — accepteur avec cible nettoyée

Date : 31 juillet 2026

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_clean_target/ac50c1c8c00344b5`

## Correction

Le modèle et ses features ne changent pas. Seule la cible d'apprentissage est
corrigée :

- 4 666 scènes `MATCH_EXACT` du fit historique sont conservées ;
- 881 `AMBIGUOUS` mécaniques, jamais adjudiquées, sont exclues du fit ;
- les 143 dossiers difficiles adjudiqués sont conservés, dont dix ambiguïtés
  réelles ;
- les 30 dossiers indépendants déjà consommés ne servent ni au fit, ni au
  choix de la famille, du poids ou du seuil.

## Sélection hors échantillon

Le meilleur candidat est le même XGBoost monotone, avec poids difficile `10`
et seuil `0.9940522313117981`.

| Population | AUTO | Erreurs AUTO | Ambiguïtés AUTO |
|---|---:|---:|---:|
| 143 difficiles OOF utilisés pour la sélection | **47/143** | 0 | 0 |
| 30 dossiers consommés utilisés après sélection | **8/30** | 0 | 0 |

La couverture difficile OOF passe de 3/143 avec la cible contaminée à 47/143
avec la cible nettoyée. Le gain vient donc principalement des labels et non
d'une nouvelle famille de modèle.

Les intervalles restent larges : 47 succès sans erreur ne suffisent pas à
certifier 99,8 %, et les huit AUTO du lot consommé ont une borne de Wilson à
95 % d'environ 67,6 %. Aucun déploiement n'est autorisé.

## Verdict

**`GO_NEXT_BLIND_DOCKET`**.

Le candidat est figé avant une nouvelle validation sur des REVIEW jamais
adjudiqués. Le futur lot ne pourra modifier ni la cible, ni le poids, ni le
seuil. Le test final reste fermé.
