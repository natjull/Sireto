# Constats après la nuit — 7 octobre 2026

## Mesure de référence, inchangée

V359 terminé à03:39:57 Paris, exit0, 2 000/2 000, 500 producteurs réussis,
une reprise de lot pour coût ; génération/évaluation/report hashes revérifiés.
Exact initial100 : 1 823/2 000 =91,15%, Wilson95[89,825%;92,317%].
Opérationnel officiel scellé :1 845/2 000=92,25%, séparé. Objectif99,2% non
atteint ; au moins161 des177 ratés à récupérer sur CE panel de développement.
Toutes les177 cibles sont dans l'index natif, mais leurs preuves peuvent être
incomplètes ou peu discriminantes. Pas de fullOOF/SLO/READY/fold1.

## Audit post-seal et limites

Code360 `7e07356b`, artefacts
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v360_miss_evidence/run_20261007_a`.
Après nouvelle admission ALL500 et filtres Arrow234/2000 :177 ratés +64 hits
témoins sélectionnés par hash. Les flags ne sont PAS une attribution causale,
ni des catégories exclusives : similarités simples, pas preuve de label faux.

- 58 ratés ont déjà le bon SIREN dans100, mais pas le bon établissement ;
  119 n'ont pas le bon SIREN dans la sortie.
- 150/177 ont une faible similarité nom CRM/noms des champs natifs selon
  le diagnostic ; seulement7/64 dans les hits témoins.
- 49 n'avaient pas le canal géométrique compatible ; sur90 distances GT/BAN
  calculables,34 GT à<=300m et21 au-delà2km. Pas de distance inventée pour
  les coordonnées manquantes. Ces chiffres ne prouvent pas la qualité BAN/GT.

Code361 `446b4609`, artefacts `.../retrieval_v361_name_provenance/run_20261007_a` :
un passage8,56s sur64 buckets/105 564 761 faits et57 530 682 noms officiels V130.
Tous les fichiers consommés hashés puis epochs revérifiés, noms liés au parent
ou explicitement au site, aucun sibling alias emprunté. Cette source factorise
les observations lexicalement éligibles V129, pas les lignes masquées/nonlexicales
ni une nouvelle lecture des stocks bruts27champs.68 ratés ont des variantes
pré-natives non présentes dans les champs stockés Tantivy ;142 restent de faible
similarité même avec ces noms. Aucun cas «raw>=.8 / native<.5» ;3 raw>=.8,
6 gains de similarité>=.2. Une réindexation seule ne semble pas expliquer le gap.

Code363 `062c4dc8`, artefacts `.../retrieval_v363_transfer_coverage/run_20261007_a` :
parquet officiel original scellé692 010 relations BODACC ASSET_TRANSFER SIREN→SIREN,
pas le payload320 rejeté.37 ratés ont un lien ;8 ont une entreprise reliée déjà
naturellement émise, dont6 sans le bon parent. Transfert d'actifs != identité,
pas encore une preuve qu'une expansion admissible retrouve le bon SIRET.
Ancien320 avait STOP_OUTPUT_SIZE sur JSON >256Mio, pas une absence de relations.

## Essai d'un canal indépendant adresse

V362 `446b4609` mots+ngrams : timeout parent270s, pas de sorties complètes ni recall.
V364 `062c4dc8` mots seuls : STOP_RESOURCE à1 701/2 000 et249,14s, après son budget
240s ; pas de recall partiel. Artefacts historiques conservés, pas réutilisés.
V365 `ff1e1a60` conserve EXACTEMENT la recette mots seuls, avec budget explicite
360s/parent390s et nouveau namespace.2 000 sorties générées en239,214s, ownexit0
confirmé, puis labels après seal/admission358 ; zéro adresse contenant nombre
9/14 chiffres dans ce panel (garde généraliste à ajouter avant industrialisation).
Pas de pool puis reclassification : une recherche adresse top100 native par query.

Artefacts `.../retrieval_v365_address_words_budget/run_20261007_a` :
exact adresse@100931/2 000=46,55%.34 ratés antérieurs retrouvés,926 hits antérieurs
perdus. Ce canal ne remplace PAS le retrieval. Aucun recall d'union annoncé :
additionner deux listes100 ne valide pas une sortie100. Filtre commune/CP dur,
cache noncontrôlé, index seulement natif, pas preuve de généralisation/SLO.

## Fine-tuning réellement repris

Code `076108a1`, `docs/embeddinggemma2_cached_contract.md`, job
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/cached_20261007_a`.
Checkpoint initial step1 ;5pas réels préflight→6, puis job détaché5192/child5196.
Trois tests PASS dont loss ET gradients cached/direct FP64 avec faux négatifs
mêmeSIREN masqués. Vrais batches4, micro2, gradient cache déterministe ; pas
une accumulation de petites losses. Même objectif, données, splits et LoRA.
Le pic d'activations est réduit, MPS cap.25 maintenu, guards explicites distincts
du retrieval désormais terminé. Pas de watermark0. Checkpoints25pas/curseur,
validation64 fixe100pas, historique reprises borné, test fermé.

Observé >1 700pas sans reprise, ~1,3Go driver GPU et RSS<1,2Go après warmup.
Meilleure loss validation proche.331 contrebaseline.675 ; checkpoint meilleur
conservé. Cela ne démontre ni recall dense, ni gain marginal initial100.
15 035 TRAIN exploitables /2 212 validation /4 323 test réservé ; original
21 571 dev, pas nouveau holdout de pipeline ; folds0/1 nonlus, panel2000 exclu
des exemples de fit mais pas garanti disjointSIREN avec ce train. Ne pas utiliser
ce panel comme holdout inédit du dense.

## Choix pour la suite

1. Laisser les deuxepochs bornés terminer, puis figer le checkpoint choisi sur
   validation. Aucun réglage sur les4 323 test ni ouverture fold1.
2. Prochaine expérience dense : brut et tuned sur les mêmes requêtes VALIDATION
   et un corpus officiel complet des zones retenues avant labels, distracteurs
   compris, pas les seules cibles positives. Déclarer les erreurs hors partition.
   Mesurer dense initial100, exact/SIREN séparés, coût et complémentarité ; pas
   prétendre que ce test zonal valide le recall national.
3. Développer une recherche initiale à preuves indépendantes : nom/identité
   (lexical+dense+variantes officielles), adresse/XY et résolution du site après
   preuve parent, liens typés en hypothèses plutôt qu'identités. Ne pas continuer
   simplement à modifier des bonus du score conjoint ni copier34 IDs/alias CRM.
   Aucun pool fixe2000/500 ni reranker aval déguisé. L'admission directe100 doit
   être mesurée avec ses pertes, pas déduite du recall d'une union plus large.
4. Auditer le bruit des labels et les absences de preuves, sans corriger ou
   exclure automatiquement les177 du dénominateur. Cross-fitting234 nécessaire
   avant usage des scores par un modèle aval ; aval gelé jusqu'au gate retrieval.

À ce stade : PIVOT_GENERATION, pas gain de recall du pipeline déjà livré.
