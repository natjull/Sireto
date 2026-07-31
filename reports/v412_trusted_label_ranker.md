# V4.12 — ranker réentraîné sur les 279 labels fiables

Artefact local :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_trusted_label_ranker/2f57628196fefce0`.

Le retrieval et les features sont inchangés. Quatre poids des nouvelles
requêtes ont été comparés en cinq plis OOF groupés par composante SIREN.

| Poids | Bons top 1 OOF / 254 | Erreurs corrigées | Régressions | Train historique OOF |
|---:|---:|---:|---:|---:|
| 0,25 | 212 | 49 | 5 | 4 657/4 666 |
| **0,50** | **216** | **52** | **4** | **4 655/4 666** |
| 0,75 | 213 | 49 | 4 | 4 651/4 666 |
| 1,00 | 215 | 51 | 4 | 4 653/4 666 |

Le poids `0,5` est retenu. Il fait passer le Hit@1 des 254 cas corrigés de
**168/254 (66,14 %) à 216/254 (85,04 %)**. Sur les 251 cas dont la vérité est
présente dans le pool, cela représente 86,06 %.

Les trois erreurs de retrieval sont `13266`, `13923` et `fresh:AC009634` : leur
SIRET exact reste absent des 100 candidats. Elles sont comptées comme erreurs
end-to-end.

Sur 1 127 contrôles dev non concernés par les adjudications, baseline et
candidat font tous deux 1 127/1 127. Le réentraînement ne produit donc aucune
régression observée sur cet écran.

Verdict : **`GO_BUILD_TRUSTED_OOF_SCENES`**. Le test final reste fermé et le
modèle n'est pas promu en production.
