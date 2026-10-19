# Étape2-8 — signaux de société co-localisée

## Pré-déclaration avant mesures

Autorités : banc G0 `9a8ecdd6`, familles `c37a1a7b`, tirage3000 `d54cec2c`, optionD `7b0005a8`, V142 `36c7a476`. Aucun retrieval ni label modifié, aucune sélection sur les erreurs ; jeux fermés et v174 non lus.

### Population et comparaisons

- Sous-banc : intersection EXACTE des17427 G0 avec les3000 IDs scellés. Les CV0–4 sont les partitions du banc G0, restreintes à ce sous-banc, pas de nouveau tirage de plis. Quatre recettes seulement :85,85+S1,85+S2,85+S1+S2 ; cible ranking exacte, paramètres R3/M_content inchangés.
- S1 seul également sur17427, comparé à la base OOF existante du même banc. Secondaire, pas de sélection de population par performance.
- Smoke :100 entreprises du sous-banc par SHA256 `20261019|colocated-smoke|SIREN`, toutes leurs lignes du tirage, mêmes partitions héritées. Pas de publication comme résultat complet.
- Si estimation S2>3h : entreprises entières du sous-banc ordonnées SHA256 `20261019|colocated-budget|SIREN`, préfixe maximal tenant dans le budget conservateur. Comptes, estimation et motif publiés ; aucune sélection par label de succès.

### S1 — NAF attendue, supervisée strictement hors entreprise

- NAF rév.2 officielle seulement ; division=deux premiers chiffres, groupe=trois premiers chiffres. Anciennes nomenclatures ou NAF inconnue restent manquantes, aucune requête retirée.
- CountVectorizer unigrams, tokens ASCII alphabétiques≥2, noms CRM normalisés par les helpers existants, sans nombre/identifiant ; lowercase, min_df=2,max_features=20000, aucun alias ajouté. Vocabulaire ajusté seulement sur les noms des lignes autorisées au fit.
- Deux MultinomialNB, alpha=1 ; classes fixées par la nomenclature publique locale, prior de classe lissé add1. Aucune HPO, aucun réglage après les chiffres. Cibles=NAF du SIRET humain officiel, même absent du pool ; utilisée uniquement pour apprendre S1 sur les groupes d'apprentissage.
- Pour CHAQUE fit ranking (outer, proper, meta), features des lignes ranking d'apprentissage obtenues par3 plis entreprise internes, NB ne voyant pas leur entreprise. Features de toutes lignes scorées obtenues par NB fit sur le seul train de ce ranking. Un NB global OOF calculé une fois n'est PAS admissible : il pourrait voir les labels du test externe via les lignes ranking d'apprentissage.
- S1 : `naf_div_logp`, `naf_group_logp`, `coloc_div_rank`, `coloc_group_rank`, `coloc_div_delta_best`, `coloc_group_delta_best`, `naf_missing`.
- Co-localisation calculable sans label : adresse stricte INSEE+numéro positif+suffixe+voie normalisée, sans plage ; aucune équivalence entre SIREN. Rang dense des probabilités distinctes parmi les codes NAF connus de ce groupe ; ex æquo même rang. Delta=logp moins meilleur. Adresse absente : relatifs NaN ; singleton valide : rang1/delta0 ; code inconnu : scores/relatifs NaN et indicateur manquant1.

### S2 — Gemma BASE sans adapter

- Modèle local `H/base`, aucun chargement de checkpoint adapté/LoRA. AutoModel bfloat16 eager/MPS, mean pooling masqué puis L2 comme le code voisin, modeeval/inference sans gradient, threads1, batch4,max512tokens. Prompts publics existants query/document.
- CRM : nom uniquement. Candidat : dénomination UL publique/usuelle + enseignes/usuel de l'établissement publics + LIBELLÉ NAF officiel. Pas d'adresse, code d'identité, statut, nom CRM alias ni vérité dans le texte candidat. Libellés locaux `oubusiness/gold/naf_activity_reference_2026-07-24.parquet`, versionNAF_REV2, provenance INSEE/raw SHA vérifiée. Une référence locale a été retrouvée après le précédent audit ; aucun web.
- Une vue tronquée explicitement à512tokens si nécessaire, compte publié ; textes vides→scores NaN. Déduplication EXACTE des textes avec prompts, cache propre et lié au SHA modèle/textes ; anciens vecteurs adaptés incompatibles, non réutilisés.
- S2 : `gemma_base_cosine`, `coloc_gemma_delta_best` ; delta=cosine moins meilleure au même groupe adresse, adresse absente→NaN. Seul le nécessaire sous-banc est encodé ;526 encodés conditionnellement au gain.
- Microbenchmark réel avant encodage complet, jusqu'à1000 textes propres mélangés, warmup distinct. Projection totale=chargement+1,5×débit×textes nécessaires ; maximum3h réelles d'encodage du sous-banc (arrêt protégé si dépassé). Aucun ajout de label ni extraction supervisée des textes.

### Mesures et décision

- Top1 exact et union contractuelle inchangée ; n/N, Wilson95, global/par CV et folds population2/3/4. Absences du pool conservées. Base sous-banc réentraînée sur le même volume que variantes.
- Delta apparié query-weighted, bootstrap10000 entreprises entières, seed20261017 ; mêmes tirages pour3 variantes×2 vues, IC95 et simultanés6 Bonferroni. Pas de refits bootstrap.
- Famille : masque FIGÉ des1046 erreurs co-adresse de `c37a1a7b`, intersection de population, donc certains cas peuvent être déjà corrigés par la nouvelle base sous-banc. Publier n, base/variante, gains ET régressions, delta net et IC entreprise conditionnel ; ce sous-groupe ne sert jamais à sélectionner ou entraîner. Pour la mesure17427, même masque1046.
- SUR : protocoles du banc G0 inchangés, proper60%/isotonique20%/seuil20% des entreprises train externe, scène logistiqueC1 propre OOF3, deux méthodes marge et scène+isotonique. Cible UNION, seuil empirique interne≥98%, couverture maximale/ex æquo indivisibles, sans minimum d'effectif ni borne Wilson. Test distinct ;0SUR→précision indéfinie. Ranking proper distinct du ranking80% du tableau top1.
- Confirmation si au moins un IC95 UNION ordinaire strictement>0 : meilleure variante sous-banc selon delta UNION, égalité ordreS1/S2/S1+S2 ; à défaut S1 complet si son IC95>0. Multiplicité et sélection publiées ; aucune certification. Ajuster un seul content sur le volume correspondant, puis526 V142 mêmes100 et empilement21 features/mêmes plis entreprise de l'optionD, confiance union ; aucune réoptimisation sur526 déjà consommés. Si aucun gain, pas d'ouverture de confirmation, pas de nouveau fit V142.

### Exécution et preuve

Garde RSS+driver MPS conservatrice5,75Gio (<6), DuckDB512Mio/threads1, workers CPU/Metal strictement séquentiels. Attendre workers V142>4Gio ; hydraulic intouchée : détection des processus VM et attente si hydraulic active (ou VM non identifiable, conservateur) ET mémoire free+inactive<8Gio. Aucun appel SSH/contrôle VM. Attentes dans le superviseur détaché, pas une boucle d'attente interactive.

Namespace neuf `H/step2_colocated_signals_20261019/{smoke,full}` ; sources/modèle/implémentations/protocole scellés, aucun pending/étranger touché. Chaque score ranking et modèle NB/scène/isotonique sérialisé/rejoué ; rôles/groupes/cibles/colonnes et empreintes des features par fit enregistrés. Smokes réels, pas de suite unitaire. Publication automatique isolée des seuls livrables et `handover.md`, arrêt plutôt qu'écraser un journal tiers sale. Rapport≤30 lignes dans les artefacts ; aucune promotion/push/installation/dépense.

## Résultats

**CHECKPOINT RUNNING — résultats OOF, deltas et SUR pas encore disponibles.**

- Implémentation scellée `182d7aa1`, protocole `5ba93ebd`. Deux smokes réels118 dossiers/100 entreprises :100 fits chacun,2832 vecteurs ranking et200 usages NB sérialisés/rejoués sans chevauchement entreprise. Toutes prédictions et décisions des4 recettes strictement identiques entre runs ; `smoke_verified/verification.json` passé.11498 textes BASE,10894 hors probe en69,606s,0troncature, pic RSS+driver2,18Gio. Seul changement post-smoke : télémétrie RSS immédiate ;31 fonctions/classes scientifiques AST inchangées, garde réelle vérifiée.
- Préparation complète terminée en30,75s, pic RSS1,35Gio :17427 dossiers/11483 entreprises. NAF rév.2 disponible pour17375 cibles supervisées et1322009/1742700 paires candidat ; tous les dossiers conservés.
- Sous-banc retenu intégralement :**3000 dossiers/2454 entreprises**, partitions G0 héritées, aucune réduction. Famille co-adresse FIGÉE :196 cas du sous-banc,1046 sur l'ensemble ; pas de sélection sur ces cas.
- Probe BASE réel du nouveau texte du brief :**1000 textes (500CRM/500SITE) en5,822s**, chargement1,069s/warmup séparés,0troncature ; pic RSS+driver2,24Gio. Ce débit concerne les textes courts du brief, pas les anciens textes nom+adresse/adaptés.
- **268697 textes uniques nécessaires**, projection conservatrice1,5×débit :2347,45s (**39,12min**, estimation et non temps final), sous le budget3h ; garde3h réelle active. Encodage complet en cours, puis quatre recettes et S1 complet17427, bootstrap/SUR, confirmation526 conditionnelle seulement.
- Complet détaché **PID91732** : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/step2_colocated_signals.py run --commit`.
- Suivi : `tail -f /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/step2_colocated_signals_20261019/full.log` ; progression `full/encode/progress.json`, puis `full/<recette>/worker.log`.
- À succès : résultats remplacent ce checkpoint, `full/final_report.md`≤30 lignes et `full/publication.json` attestent la publication/son commit. Arrêt plutôt qu'écraser un journal tiers sale ; en cas d'arrêt lire les logs et `resource_stop.json`, ne pas interpréter le checkpoint comme statut vivant.
- Aucun résultat top1/delta/SUR ni gain V142 complet annoncé. Aucune modification de labels/pools/jeux fermés/v174/VM/étrangers/pending, aucun push. Une seule attente bloquante utilisée pour vérifier la fin du smoke, pas d'attente sur le long run.
