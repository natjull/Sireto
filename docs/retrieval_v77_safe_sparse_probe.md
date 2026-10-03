# V77 — Vocabulaire sémantique réellement isolé par entreprise

Préenregistrement avant mesure. L'objectif national initial100 exact≥99,2%
reste inchangé. Contrastes TRAIN512 ne prouvent PAS le recall ni le SLO.

Audit du code V59 : `vocab=vocabulary(documents,...)` précède la boucle des
folds. Documents issus de la sélection TRAIN positive/négative des512 scènes.
La protection des poids ne protège donc pas la SÉLECTION du vocabulaire
contre les documents associés aux entreprises outer/inner. Ceci n'est pas
preuve d'un gain frauduleux mesuré (aucun recall complet V60 n'existe), mais
interdit de présenter ce chemin comme prétraitement OOF totalement isolé.
Conserver V59/V60 et leurs contrôles historiques immuables ; pas de rerun
national coûteux du bundle partagé avant correction.

V77 conserve les familles ridge1/10/100, alpha0/0,5/1/2, max8 mots et les11
preuves natives. Chaque outer fold a SON vocabulaire officiel : positifs ET
négatifs des seules requêtes de fit, parents protégés exclus y compris quand
présents comme négatifs. Même exclusion du vocabulaire lors de la sélection
inner, puis reconstruire sur inner-train+validation pour le refit outer.
Ne construire aucune cible lexicale des requêtes hors fit. Les mots viennent
des5 champs officiels de noms, jamais des adresses ni d'un dictionnaire CRM.
Documents du fold testé uniquement dans le contrôle extérieur APRÈS fit.
Les contrôles inner déterminent hyperparamètres, jamais les métriques outer.

Source512 V52/V56 et validation de sigles V67, registre COMPLET234, protégeant
entreprises positives ET négatives. Aucun fold0/1, alias CRM, synthétique,
identifiant comme signal, positif injecté. Ces scènes historiques et leurs
exclusions opérationnelles ne deviennent pas le dénominateur all-human final.
Tests contrefactuels : changer texte des documents/targets/queries du fold
extérieur ne change ni vocabulaire inner/final, sélection ou projection.

Worker80s/4Gio/zéro swapout, supervision90s, threads2 ; pas de budget élargi,
GPU, cloud, dépense, re-embedding documents ou changement aval. Exporter
projections ET vocabulaire distincts par fold pour un futur serving query-only.
La mesure nationale128 à sorties100 natives, indépendante et scellée avant
labels, reste indispensable après ce contrôle : ne pas substituer une erreur
de tête ou un nombre de concurrents au recall@100, ni appeler ceci SOTA.
