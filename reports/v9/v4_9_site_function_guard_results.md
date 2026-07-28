# V4.9 — diagnostic du garde-fou de fonction de site

## Verdict

**`STOP_SITE_FUNCTION_GUARD`**

La taxonomie déterministe refuse correctement les trois erreurs structurelles
observées lors de l'ouverture random V4.8, sans refuser aucun des 116 top-1
corrects. Elle ne refuse toutefois que 3 des 34 top-1 faux ou ambigus de la
population rétrospective fiable. Le minimum préenregistré était de cinq.

La V4.9 ne justifie donc pas le coût d'une nouvelle cohorte de 300 dossiers.
Elle n'est ni promue ni étendue après observation.

## Résultats bruts

| Mesure | Résultat | Gate |
|---|---:|---:|
| Dossiers reconstruits | 172 | 172 |
| Labels fiables | 150 | descriptif |
| `TOP1_CORRECT` | 116 | descriptif |
| `TOP1_WRONG` ou `AMBIGUOUS` | 34 | descriptif |
| Mauvais/ambigus refusés | 3/34 | ≥ 5 — **échec** |
| Erreurs random V4.8 refusées | 3/3 | ≥ 1 — passe |
| Bons top-1 refusés | 0/116 = 0 % | ≤ 5 % — passe |
| Règles propres à un dossier | 0 | 0 — passe |

Les trois bascules sont :

1. mairie CRM contre école primaire SIRENE ;
2. maternelle CRM contre école primaire SIRENE ;
3. CRM portant simultanément `FAM` et `MAS`, donc fonction intrinsèquement
   ambiguë.

Les 31 autres erreurs ou ambiguïtés fiables ne sont pas détectées. La piste
est très précise mais trop étroite pour constituer à elle seule la prochaine
architecture de décision.

## Intégrité de l'expérience

- Taxonomie, code et tests gelés avant mesure au commit `a311306`.
- Entrées et hashes épinglés au commit `49832e4`.
- Évaluateur gelé avant exécution au commit `67b2cb5`.
- Aucun modèle réentraîné.
- Aucun seuil modifié.
- Aucune population fraîche ouverte.
- Aucun test final historique rouvert.
- `UNRESOLVED` exclu des métriques de précision.

Artefact immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_9_site_function_retrospective/30e22eae11620538`

Hashes de sortie :

| Artefact | SHA-256 |
|---|---|
| `retrospective_predictions.parquet` | `dc4499c3c71060b992881c09bd194e8479fe9c08f600910676490ce63ce2d5b2` |
| `retrospective_report.json` | `94a7e24064f37749149d4f0f92df7e95d637dd33bd65a6bd8b79f2d17da73801` |

## Conséquence

La prochaine étape n'est pas d'ajouter des mots à cette taxonomie. Il faut
auditer les 31 erreurs ou ambiguïtés non interceptées avec les champs
réellement disponibles avant et après classement, puis rechercher une ou
plusieurs familles générales et mesurables. Toute nouvelle hypothèse devra
être préenregistrée et validée sur une population indépendante.
