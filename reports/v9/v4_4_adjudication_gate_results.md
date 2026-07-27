# V4.4 — gate d'adjudication autonome

Date d'évaluation : 27 juillet 2026  
Verdict : **`PIVOT_MORE_EVIDENCE`**

## Résultat

Les cinq lots sectoriels A–E ont été reliés à la file V4.3, aux décisions du
shadow V4.1 et à leurs vrais pools top-10. Ils ont ensuite été combinés aux
cinq contradictions déjà canoniques.

| Mesure | Observé | Gate | Manque |
|---|---:|---:|---:|
| `TOP1_CORRECT` validés | 35 | 75 | 40 |
| `TOP1_WRONG` validés | 11 | 50 | 39 |
| Cas random validés | 19 | 30 | 11 |
| Décisions fondées sur une preuve interdite | 0 | 0 | 0 |

Le corpus dédupliqué contient 53 cas : 47 décisions validées par au moins deux
groupes de preuves indépendants, une décision `AMBIGUOUS` validée et six
`UNRESOLVED`. Il fournit 47 scènes éligibles à l'accepteur et 37 scènes
éligibles au ranker selon les règles gelées.

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
- Contradictions connues :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/320fe62322e14d25`
- Gate consolidé :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_gate/6f5972fbdcf10043`

## Décision

Le corpus est encore trop petit et trop pauvre en erreurs prouvées pour
réentraîner honnêtement l'accepteur ou le ranker. Les modèles, le seuil et le
retrieval restent gelés. La collecte gratuite continue en priorité sur :

1. onze dossiers aléatoires supplémentaires ;
2. trente-neuf vrais `TOP1_WRONG`, avec un SIRET alternatif seulement lorsqu'il
   est dans le pool figé et doublement prouvé ;
3. quarante `TOP1_CORRECT` supplémentaires pour préserver l'équilibre du
   corpus.

Le test final reste fermé.
