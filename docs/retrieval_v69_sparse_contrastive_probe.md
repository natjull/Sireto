# V69 — Impacts sparse appris par contraste, contrôle avant index national

Préenregistré avant mesure. Ce n'est ni un pool2000 reclassé, ni une preuve
de recall100, ni une reproduction de SPLADE. Aucun modèle aval modifié.

Hypothèse : apprendre directement les poids des termes officiels émis par
la requête, face aux négatifs naturels, plutôt que traduire uniquement vers
les mots du positif puis ajuster un poids global. Représentation NOM/ADRESSE
query-only gelée de V52, bases SVD16 par champ, biais, vocabulaire officiel
typé N:/A:/I: plafonné1024. I: est compilé depuis UN alias courant à la fois,
langage V67/V68. Pas d'identifiant, alias CRM ou clé administrative dans ce
vocabulaire. DF3..15% des documents train sûrs seulement, pos ET neg des
parents protégés exclus. Normalisation documentaire par tous les termes de
chaque vue (indépendante du vocabulaire sélectionné).

Activation identique train/serve : Top16(log1p(ReLU(zW))), poids non négatifs,
pas de renormalisation tardive. Les23 preuves natives sont les19 V56
(lexical/exacts + XY25/100/500/2000m + fiabilité BAN) et quatre preuves de sigle
V67. Loss égale par query log(1+sum exp(score_neg-score_pos)), tous négatifs
naturels sûrs, pénalité native0,005 vers prior V56, decoder ridge0,01/0,1
divisée par dimension latente vers initialisation ridge10, FLOPS0,1 sur
somme des poids moyens au carré. Gradient sparse, pas matrice pair×vocab dense.

Comparateurs : NATIVE23 seul ; RIDGE_SPARSE decoder fixé et coefficient
appris HORS log/activation ; CONTRASTIVE_SPARSE poids et decoder conjoints.
Choix famille/ridge sur contrôle interne entity-disjoint V53 uniquement ;
non convergés exclus, refit choisi sur train+contrôle et OOF externe rapporté,
jamais utilisé dans sélection. Une sélection native est un rejet recevable
de cette hypothèse, pas une raison de changer les critères après résultat.

512 queries, folds2/3/4, caches V52/V56 hashés immuables et validation des
alias V67. Les positifs de cette source TRAIN ne sont pas des injections
dans une évaluation retrieval : aucune génération/évaluation de candidats
nationaux ici. Chaque query reste présente, aucun négatif sûr impose STOP.
Vérification exhaustive des alias courants exacts versus proof native V56.
Champs vides query masqués malgré le préfixe encoder.

Budget80s,4GioRSS,zéro nouveau swapout ; supervision90s,2threads. Aucun
nouvel encodage/download/build national, folds0/1 non ouverts. Sortie scellée
models/reports/manifest, aucun candidat écrit. STOP immuable si garde dépassée.

Limites : négatifs source restreints, équivalents opérationnels exclus du
teacher (PAS du dénominateur retrieval), alias regroupés dans certaines vues,
pas de cohérence temporelle/source, pas de relations/filiations ou activité
juridique intégrées, pas d'encodeur de DOCUMENTS appris. Les preuves XY/BAN
font partie du fit mais leur exécuteur national conjoint n'est pas réalisé.
Le coût V68 Python ne valide pas un index national efficace. Seul un futur
run full-corpus direct100 OOF pourra établir le recall/SLO et le gate99,2%.

Appui limité : [SPLADE](https://arxiv.org/abs/2107.05720) étudie une loss
retrieval avec représentation sparse, log-saturation et régularisation FLOPS.
Ici, le decoder et le vocabulaire sont différents, la représentation
documentaire fixe et aucun résultat du papier n'est une garantie pour SIRETO.
