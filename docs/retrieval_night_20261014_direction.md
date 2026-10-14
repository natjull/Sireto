# Nuit du 14 octobre : résultat et direction, sans surapprentissage revendiqué

## Ce que le retrieval a réellement gagné

Sur les **526 mêmes requêtes de développement**, la recherche initiale
limitée à100 SIRET passe de **504 à507 résultats humains exacts** :
trois gains distincts, zéro perte. Recall brut96,3878%, pondéré95,6790%.
La vue SIREN est511/526 et l'équivalence opérationnelle508/526 ; elles ne
remplacent pas le score exact. Il reste19 ratés.

Les trois gains viennent respectivement de lieux nommés OSM, de noms RNA et
de mentions propres d'un fournisseur. La combinaison des preuves est faite
avant un unique collecteur initial100, pas par reranking d'un pool2000/500.
V134, V142 et V162 mesurent les mêmes507hits, pas trois confirmations
indépendantes. BANO et la dernière normalisation générique du suffixe
communal n'apportent aucun gain exact supplémentaire sur ce panel.

**L'objectif99,2% n'est pas atteint.** Il faudrait déjà15 gains nets
supplémentaires pour dépasser ce seuil brut ici, sans que cela suffise à une
certification. Cette population a été consommée pour la conception ; son
encodeur n'est pas full-OOF et certaines sources sont actuelles/post-cutoff.
Le score n'est donc ni un résultat national indépendant ni une garantie
sur un autre CRM. Voir [audit de validité](retrieval_scientific_validity_all526_20261014.json).

## Ce qui ressort de l'audit, et ce qui reste une hypothèse

**Une limite d'accès aux preuves historiques est établie.** Les historiques
INSEE existent sur disque et dans un dictionnaire de features, mais cela
ne rend pas toutes leurs variantes accessibles aux postings de recherche
lexicale nationale. L'index principal possède déjà des mentions historiques
BODACC/RNE : dire qu'il n'utilise aucun historique serait faux. L'écart
précis de paires propriétaire/nom et son gain de recall ne sont pas encore
mesurés. Voir [audit temporel](retrieval_temporal_lineage_direction_20261014.md).

**La calibration du score n'a pas suivi l'enrichissement.** Les poids
principaux sont appris, pas tous arbitraires. Mais le même modèle ancien
alimente aujourd'hui de nouveaux canaux dont certains bonus réutilisent
des poids et règles de MAX transférés. Ses négatifs étaient déjà variés
(géographie, lexical, uniforme, contexte corporate), mais ne proviennent
pas de la compétition du score joint enrichi actuel. Cela justifie une
expérience d'apprentissage fidèle, pas la conclusion qu'un coefficient
particulier est responsable d'un raté. Voir
[audit calibrateur/serving](retrieval_native_calibration_serving_audit_20261014.json).

**La géographie doit distinguer les observations et leur qualité.** Un
autre fichier officiel fournit des positions absentes de notre table XY
actuelle. Il vient d'une autre méthode et d'un autre mois : on ne doit ni
écraser les XY actuels ni leur transférer ses codes qualité. Même un code
identifiant un numéro ne garantit pas universellement une position actuelle
précise. Son diagnostic source est séparé du retrieval ; une bonne proximité
isolée ne prouve pas une place dans le top100 composite. Voir
[source géographique complète](retrieval_v171_separate_study_geo_source_stageA_20261014.json).

**L'attribution causale fine reste incomplète.** Deux ratés étaient
géographiquement plausibles dans un diagnostic antérieur, mais l'essai
d'explication du score complet a échoué avant validation de fidélité, malgré
une dernière admission mémoire explicitement autorisée. Aucun score cible,
rang complet ou coefficient fautif n'en est déduit. Voir
[STOP instrumentation](retrieval_v160_explain_final_c_stop_20261014.json).

Enfin les scans complets BODACC activité, sigles et administration de personnes
morales n'ont pas identifié de preuve contextuelle positive supplémentaire
sur les19 ratés actuels sous les règles auditées. Cela ne condamne pas tout
texte BODACC ni la piste des dirigeants ; les personnes physiques n'ont pas
été testées ici. Ce n'est pas une raison de poursuivre une grille de recettes
sur ces mêmes19 exemples.

## Suite recommandée : une architecture, pas dix rustines

1. **Rendre le score utilisable à l'identique en production.** Le choix du
   vecteur ne doit jamais dépendre du SIREN humain inconnu. Choisir une seule
   politique indépendante de la cible et recalculer sous elle toutes les
   interactions et tous les bonus. Centraliser le score complet, son domaine,
   ses masks, ses quantifications et ses ties. Vérifier sa fidélité sur des
   entrées figées sans labels avant tout apprentissage ou comparaison.

2. **Fermer le vrai écart de couverture historique.** Mesurer globalement,
   sans CRM ni GT, les variantes publiques propres absentes de l'index.
   Construire une annexe compacte de postings à partir des observations déjà
   disponibles ; conserver propriétaire, rôle, période/publication et source.
   Un ancien nom d'un site reste celui de ce site ; un nom de parent ou de
   successeur n'est pas copié comme enseigne de chaque établissement. Les
   postings s'unissent avant le même collecteur100, sans pool intermédiaire.

3. **Apprendre la compétition des preuves ensemble sur TRAIN seulement.**
   Miner les distracteurs du score complet, dans le corpus réel, et comparer
   une seule évolution de minage à une baseline fidèle. Splits entiers par
   SIREN, choix en inner-folds ; géographie/densité/date comme axes de
   généralisation. Inclure une éventuelle observation géographique séparée
   uniquement après décision d'admissibilité/as-of et recette pré-déclarée,
   pas en choisissant a posteriori la position qui aide les ratés. Les scores
   hors-échantillon restent requis ; pas refit sur une matrice simplifiée
   censée représenter un moteur dont les interactions dépendent des poids.

4. **Geler une version, puis la confronter à une vraie cohorte indépendante.**
   Même corpus national, sortie initiale<=100, tous les humains exploitables,
   labels annotés indépendamment et en aveugle. Cohorte non consommée,
   groupes SIREN séparés, tailles de communes représentées et dates connues.
   Pour promettre la généralisation inter-CRM, il faut aussi un autre CRM.
   Une seule comparaison confirmatoire ; aucune retouche après ouverture.
   Publier exact, opérationnel, pondération/incertitude adaptée aux clusters,
   gains/pertes et latence production réelle. Les jeux fermés ne sont pas
   ouverts implicitement par cette proposition.

## Protections et décision de cette nuit

Zéro label humain changé, zéro alias CRM ou règle par client, aucun positif
injecté, aucun filtre de dénominateur. Résultats complets et vrais exits
scellés avant labels ; gains et pertes publiés. Fold0/fold1/test4323/control962
fermés, modèles aval et serving gelés, pas de gradient encodeur cette nuit.
Les modifications étrangères de l'utilisateur sont préservées.

Verdict retrieval : **PIVOT_GENERATION, objectif non atteint**. Cela ne désigne
pas une cause unique des pertes : couverture et compétition restent à
séparer. Aucun READY_FOR_FOLD1, promotion, garantie SOTA ou SLO neuf.
Fin de mandat prévue07:06:08Paris/05:06:08UTC ; aucune nouvelle nuit implicite.
