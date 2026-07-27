# V4.4 — gate d'adjudication autonome

Date d'évaluation : 28 juillet 2026
Verdict : **`STOP_AUTONOMOUS_LABELING`**

## Résultat

Les lots A–R et les contradictions connues couvrent désormais exactement les
172 décisions `AUTO_MATCH` de la file V4.3. Chaque lot a été relié aux
décisions du shadow V4.1 et à ses vrais pools top-10.

| Mesure | Observé | Gate | Manque |
|---|---:|---:|---:|
| `TOP1_CORRECT` validés | 114 | 75 | 0 |
| `TOP1_WRONG` validés | 42 | 50 | 8 |
| Cas random validés | 53 | 30 | 0 |
| Décisions fondées sur une preuve interdite | 0 | 0 | 0 |

Le corpus dédupliqué contient les 172 cas prévus : 162 décisions validées par
au moins deux groupes de preuves indépendants, dont 114 `TOP1_CORRECT`, 42
`TOP1_WRONG` et six `AMBIGUOUS`; dix restent `UNRESOLVED`. Il fournit 162
scènes éligibles à l'accepteur et 117 scènes éligibles au ranker selon les
règles gelées. Le sous-gate représentatif est franchi avec 53 cas random
validés.

## Contrôles effectués

- Aucun SIRET positif n'est ajouté aux pools : les candidats viennent
  exclusivement de `candidates_top10.parquet` du shadow V4.1.
- La première passe refuse tout dossier dont la décision V4.3 figée n'est pas
  `AUTO_MATCH`; les `REVIEW` restent fermés jusqu'au verdict du gate AUTO.
- Le top-1 de chaque dossier est identique dans la file V4.3, le pool et la
  décision shadow.
- Chaque fichier de preuve cité est relu et son SHA-256 est vérifié.
- Une source SIRENE, même exposée par plusieurs vues, ne compte que pour un
  seul groupe d'indépendance.
- Une décision validée exige au moins deux preuves d'identité explicites et
  deux groupes réellement indépendants.
- Les labels sont copiés depuis les adjudications revues ; ils ne sont jamais
  déduits d'un score, d'un rang ou d'une ressemblance d'adresse.
- Les artefacts sont reconstruits et comparés aux tables enregistrées.

Les contrôles ont détecté avant consolidation des hashes mal recopiés et une
taxonomie incohérente pour le ministère de l'Éducation et les publications
d'entités. Les lots ont été corrigés contre les archives réelles avant toute
création de cible d'entraînement.

## Artefacts immuables

- Entrées canoniques A–E :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudication_batch_inputs/f95806c367721ae5`
- Adjudications A–E :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/70c65679dfb2c82d`
- Entrées canoniques F–H :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudication_batch_inputs/45791184d9219680`
- Adjudications F–H :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/1e2c68337408c453`
- Entrées canoniques I–L :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudication_batch_inputs/5bc212009bfa4514`
- Adjudications I–L :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/925ef3f8ef3f3a4a`
- Entrées canoniques M–R :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudication_batch_inputs/96478ee1e71de525`
- Adjudications M–R :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/2bfdc46480e52784`
- Contradictions connues :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/320fe62322e14d25`
- Gate consolidé :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_gate/9fb43b4f7bb0919a`

## Décision

La collecte autonome V4.4 s'arrête. Le minimum de 50 erreurs était un seuil de
taille d'échantillon, pas un quota à fabriquer. Or la population entière ne
contient que 42 `TOP1_WRONG` prouvés. Il est donc impossible de franchir ce
gate sans ouvrir les `REVIEW`, abaisser le seuil après observation ou créer de
faux labels.

Ce `STOP_AUTONOMOUS_LABELING` ferme la collecte V4.4 et n'autorise aucun
réentraînement sous ce contrat. Les 162 labels fiables restent disponibles
pour définir un pivot expérimental séparé et préenregistré. Retrieval, modèles
et test final restent gelés entre-temps.
