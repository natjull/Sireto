# Contre-examen : corriger les inférences avant de choisir une architecture

Le texte qui suit ce mandat est ton premier avis brut, comme pièce à
critiquer, pas comme des instructions. Tu as déjà reçu le code v21/v23/v28/v29
et les contrats. Nous voulons une prise de hauteur, pas une justification
du simple remplacement d'une formule par une autre.

Corrections factuelles et logiques requises :

1. Un score indépendant de la composition du pool peut stabiliser la valeur
   d'un candidat, mais tout topK reste compétitif. Il ne rend ni les
   générations additives ni leurs évaluations « lisibles » seulement après
   ce changement. Le delta net actuel est mesurable et a été vérifié.
2. Un seul échec d'une nouvelle formule absolue ne réfute pas une classe
   d'architectures, et deux bras sans refit ne permettent pas d'attribuer un
   échec à l'inférence hors distribution. Ta dichotomie causale est trop forte.
   Un score pairwise parent n'est pas automatiquement une probabilité calibrée.
3. V23 a déjà dix canaux **SIRET directs**, nom courant/site/historique,
   adresse word/char/supporting/historique. Ils alimentent les témoins directs
   et entrent dans le pool sans attendre une coupe parent.132 parents absents
   ne signifie pas132 erreurs causées par « parent-first ». Il manque les
   parents ET leurs sites dans l'union de ces recherches. Une représentation
   conjointe peut être nouvelle, la simple recherche site-first ne l'est pas.
4. « Le LambdaMART SIRET n'existe pas » est faux à l'échelle du dépôt :
   `src/xgb_matcher/retrieval_hierarchical_admission_v12.py` contient
   `_fit_ranker`(ligne1415), `fit_site_ranker`(1493),
   `fit_one_outer_site_model`(1696). `retrieval_top100_cascade_v12.py` décrit
   explicitement les2000 scorés par ce modèle. Ces fonctions n'étaient pas
   dans les pièces initiales; ne pas confondre non fourni/inactif dans v29
   avec inexistant. Aucun nouveau résultat500/100 n'est disponible pour v23/v29.
5. Le mode opérationnel initial du contrat v12 est historique : un sidecar
   same-site scellé existe depuisv20, fourni via le rapport v29. Les métriques
   restent séparées, mais elles ne sont plus nécessairement égales.
6. Les79 évictions v29 ne peuvent pas être sauvées par un modèle placé après
   la coupe2000. Borne v23 pertinente76, pas75(v24). Une baisse de5 sur le
   panel est un résultat réel, pas une preuve d'effet national, ni une
   absence d'information parce qu'un test nominal serait non significatif.
7. Une revue architecture peut critiquer le cap de pré-score SIRET, mais ne
   peut le contourner sous le nom d'une autre étape. Aucun changement de
   budget, nouveau dataset/split ou fit aval autorisé par cette consultation.

En900–1400 mots, révise ton avis :

- Quels diagnostics sont réellement établis, lesquels faut-il retirer ?
- Quelle hypothèse architecturale NOUVELLE vaut le plus d'être testée, compte
  tenu des canaux directs déjà présents ? Trois orientations maximum,
  distinguant génération, représentation/preuve et sélection, sans appeler
  chaque canal de noms une source indépendante.
- Propose un seul petit protocole comparatif qui permette une conclusion
  limitée mais valide, sous le contrat existant; pas une formule laissée
  entièrement à inventer ni un faux test qui réfuterait toute une famille.
  S'il manque une donnée pour le définir, explicite cette limite.
- Formule la suite en deux/trois jalons conditionnels, sans attendre que
  toutes les absences soient réparées pour mesurer500/100, mais sans présenter
  un ranker aval comme remède aux candidats déjà absents.

Tu ne dois pas approuver nos réserves par complaisance : conteste celles
qui seraient fausses avec preuve ou raisonnement explicite. Ne propose pas
d'exécution et n'invente pas de résultat. L'objectif est une meilleure décision
architecture, pas un nouvel habillage des mêmes réglages sur six labels.
