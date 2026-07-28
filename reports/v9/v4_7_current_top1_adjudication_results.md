# V4.7 — adjudication exhaustive des top-1 courants

## Verdict

**`GO_ACCEPTOR_FEASIBILITY`**

Ce verdict autorise uniquement la préinscription puis l'exécution d'une
expérience d'accepteur hors test. Il ne certifie ni une précision de 99,8 %,
ni un déploiement. Aucun modèle n'a été entraîné, aucun seuil n'a été choisi
et le test final est resté fermé.

## Résultats du gate préenregistré

| Mesure | Résultat | Seuil | Gate |
|---|---:|---:|---|
| Scènes courantes fiables | 150 / 172 | ≥ 150 | PASS |
| Scènes aléatoires fiables | 52 / 57 | ≥ 50 | PASS |
| Négatifs ciblés fiables | 28 | ≥ 20 | PASS |
| Négatifs aléatoires fiables | 6 | ≥ 3 | PASS |
| Ancien label transporté vers un top-1 différent | 0 | = 0 | PASS |
| Label fiable avec moins de deux groupes de preuves | 0 | = 0 | PASS |

Sur les 37 top-1 ayant dérivé depuis V4.4 :

- 23 sont maintenant fiables : huit `TOP1_CORRECT`, quatorze
  `TOP1_WRONG` et un `AMBIGUOUS` ;
- quatorze restent `UNRESOLVED` ;
- les 37 ont été traités exactement une fois ;
- les anciens verdicts V4.4 n'ont jamais été copiés vers le nouveau SIRET.

## Méthode

Le dossier et le SIRET courant ont d'abord été gelés. Pour chaque décision
fiable, le constructeur exige :

1. une vue du SIRET courant dans le registre officiel archivé ;
2. au moins une preuve publique indépendante, effectivement téléchargée ;
3. la présence déterministe des faits préenregistrés dans le HTML ou le PDF ;
4. au moins deux groupes de preuves indépendants ;
5. une relation unique entre la preuve et le top-1 courant.

Les relations `SUPPORTS_CURRENT_TOP1`, `CONTRADICTS_CURRENT_TOP1` et
`AMBIGUOUS_CURRENT_TOP1` produisent respectivement les labels canoniques. Une
page inaccessible, un terme absent ou des sources contradictoires produisent
`UNRESOLVED`, jamais un verdict forcé.

Une première reconstruction immuable a correctement refusé trois pages dont
les faits étaient présents mais dont les expressions de contrôle étaient trop
strictes (espaces dans un SIRET, article « du », pluriel « étudiants »).
Seules ces expressions littérales ont été corrigées ; ni les relations, ni les
faits, ni les labels visés n'ont changé. La reconstruction canonique ne
contient plus aucune source publique échouée.

## Artefact canonique

Répertoire :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_7_current_adjudications/4cc5420fb5da0683`

Fichiers principaux :

- `evidence.parquet` : registre et preuves publiques avec URLs, dates,
  empreintes, faits et résultats des contrôles textuels ;
- `adjudications.parquet` :
  SHA-256 `c3ceb30e0186c58c6dc9957658935eb7d5a557e75cae4e83c6f6f2cabfb80b74` ;
- `current_labels.parquet` :
  SHA-256 `e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2` ;
- `gate_report.json` : verdict et compteurs du contrat ;
- `public_raw/` : contenus HTML/PDF effectivement utilisés.

Implémentation et spécification des preuves : commit `bdfbadc`.
Suite complète : **324 tests passants**.

## Conséquence

La prochaine étape n'est pas d'ouvrir le test final. Elle consiste à
préenregistrer V4.8, avec le retrieval V4.2-B et le ranker A gelés, puis à
comparer au minimum une règle/logistique simple à l'accepteur historique sur
des prédictions hors pli. Les huit cas aléatoires V4.7 restent réservés à
l'évaluation et ne peuvent servir ni à l'entraînement, ni au choix du seuil.
