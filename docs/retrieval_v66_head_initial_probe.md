# V66 — vrai initial100 national, recette de tête V65

Préenregistré avant génération/labels. V65 choisit HEAD_WORD sur les3 folds
internes, preuve de sigle rejetée et non forcée. Tester sa recette exacte :
11 preuves natives + <=8 mots officiels appris, vocabulaire/décodeur par fold
issus seulement des entités autorisées, poids de tête contrastifs. Un seul
collecteur natif national100, pas d'ancien pool/candidat lu, pas d'aval.
Les4 poids de sigle V65 doivent être EXACTEMENT0 pour réduire algébriquement
le noyau16 au noyau11+alpha : sinon STOP, ne pas supprimer une preuve active.

128 mêmes queries préexistantes, folds2/3/4, disjointes512 train, all-human
inchangé. Réutiliser UNIQUEMENT le tableau128x384 de vecteurs de NOM, étape
query-only complète/pass-resource de V61 (son pilote de candidats a STOP).
Vérifier hash/query input/order/folds, encoder/prefix/max-length/model refs
identiques à V52. Ne lire aucun batch de candidats partiels V60/V61 ni leurs
labels ; ne pas reprendre leurs sorties comme si leur protocole avait passé.
Recalculer entièrement les expansions avec les nouveaux vocabulaires et
projections V65 ; anciens termes ignorés. Nouveau pilote entier indépendant,
pas une reprise élargie de V61 ou réutilisation de ses métriques inexistantes.

Native V61 flat, score exact mathématique des features sélectionnées V65.
IEEE float32/ties natifs possibles ; canonical seulement parmi hits retournés,
pas de parité bit-exacte globale revendiquée. Corpus43896818 sites complet,
inventaire/meta hashes ; gros segments size/mtime non entièrement rehashés.
Visites internes de postings non instrumentées ;100 matérialisés ne veut pas
dire100 postings parcourus. Pas de nouvel index, encoder/GPU/cloud/dépense.

Même garde80s/6Gio/zero nouveau swap par worker8, superviseur90s, contrôleur360s,
query20s contrôle après retour natif et superviseur externe. Deux threads.
Premières16 sans labels, moyenne primaire<=4s expérimental PAS SLO ; STOP
conservé immuable, aucun label si128 non scellées, aucune relance élargie.
Les diagnostics numériques RSS/swap/temps des STOP sont conservés désormais.

Évaluation séparée après128 sorties intégralement scellées avec sidecar/ledger
officiels, exact/opérationnel séparés, V2/V3/historique/état/origine/folds,
Wilson, gains/pertes et latences. Comparateur fixé : dernier vrai pilote V57
113/128 exact, PAS baseline ancienne moins bonne choisie pour annoncer gain.
Code d'évaluateur V49 réutilisé avec ce manifeste baseline explicitement fixé,
pas changé sur disque. Pas de timing comparateur rejoué ni cache contrôlé.

Limites : pas XY/BAN, filiations ou temporalité ajoutés à cette ablation.
Panel déjà réutilisé, pas holdout neuf. Même100% ne donne pas READY_FOR_FOLD1 ;
objectif final all-human>=99,2%, Wilson>=99,0%, parfold>=98,8%, SLO et attribution
exhaustive toujours nécessaires. Fold0 non lu, fold1 fermé, aval gelé.
