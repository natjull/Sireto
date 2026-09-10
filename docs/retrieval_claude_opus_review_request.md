# Revue externe demandée avant un nouveau test de retrieval

Demande utilisateur du 10 septembre 2026 : solliciter Claude Opus 5 via
`claude -p`, effort extra high, et reprendre explicitement le sujet
« anciens / nouveaux candidats ». Aucun run de retrieval lancé dans cette
consultation. Le texte ci-dessous contient des chiffres agrégés, pas de
données CRM brutes ni d'identifiants d'entreprises/personnes.

## Mandat au relecteur

Tu es sollicité comme expert indépendant en entity resolution, retrieval,
learning-to-rank et évaluation expérimentale. Réponds en français. Critique
franchement les hypothèses et la stratégie, sans confirmer par complaisance
le plan proposé. Aucun outil, exécution, mutation du projet ou recherche web
n'est demandé. Ne prétends pas avoir inspecté des artefacts non fournis.
Ton avis n'est pas une mesure de qualité. Distingue les faits fournis,
inférences, hypothèses et expériences à faire. Ne choisis pas une recette
parce qu'elle sauverait quelques labels connus.

## Objectif et limites actives

CRM nom/adresse → entreprise SIREN → établissements SIRET. Objectif final :
conserver le SIRET humain exact pour ≥99,2% de toutes les requêtes ayant un
label exploitable, au plus100 candidats (Wilson95% ≥99,0%; PreRank500≥99,4%).
Étape de travail actuelle : améliorer réellement Master@2000, pas seulement
un canal ou le recall SIREN. Les2000 sont des références légères,500 maximum
en préclassement/hydratation riche,100 en sortie. Aucun LambdaMART SIRET sur
un grand réservoir >2000; les grands réservoirs sont exploratoires de
génération. Budget global et cascade parent/site. Modèles aval ranker,
decider, risk/accepteur gelés jusqu'au gate retrieval.

Développement uniquement en OOF folds2/3/4; pas de fuite des labels tenus
à l'écart vers générateur/features/selection. Folds humains groupés par
SIREN, disjoints entre train/test. Fold0 contaminé interdit au développement;
fold1 fermé jusqu'à autorisation one-shot explicite, non disponible pour
choisir une recette. Aucun positif injecté, alias CRM, identifiant comme
feature, nom personnel ou donnée synthétique. Labels exacts jamais corrigés
ou exclus en fonction de statut, disponibilité ou plausibilité.

Vue opérationnelle distincte : autre SIRET du même SIREN prouvé au même site
physique, selon adresse forte (numéro/voie/CP-INSEE), priorité au site actif.
Une distance seule ne justifie ni équivalence ni réécriture de label.
Absences officielles restent des erreurs end-to-end avec sidecar scellé.
Calcul Mac M4 Pro + disque externe, pas de GPU loué ni achat de données.
L'utilisateur perd patience : plusieurs diagnostics, pas encore de gain
livré dans le pipeline produit. Il veut une expérience décisive et bornée.

## Population : limite fondamentale de toute interprétation

Panel1034 =522 échecs de la recherche initiale top1750 +512 succès choisis
comme contrôles. Il est réutilisé depuis plusieurs cycles. Ses proportions
de succès ne sont PAS le recall national. Même l'OOF ne supprime pas le
biais de choix adaptatif de stratégies sur ce panel. Pas de nouveau CRM
indépendant actuellement évalué. Aucun pourcentage national extrapolé,
aucune promesse99,2%. Les segments historiques/V2/V3 restent au dénominateur.

## Anciens / nouveaux : ce qui a vraiment été fait

« Ancien » = candidat issu du raw de recherche initiale, pas une entreprise
ancienne, inactive ou confirmée. « Nouveau » = référence ajoutée par un
canal complémentaire ou une expansion SIREN→sites. Ce sont des provenances,
pas des jugements de qualité.

- V21 : protéger jusqu'à1750 anciens SIRET, remplir le reste des2000 en
  rescue. Réservoir naturel3900144 paires requête/site,898 SIRET exacts
  accessibles sur1034. Master :728 exacts,730 opérationnels.
- V22 : même réservoir et scores, supprimer la protection1750+250 et
  sélectionner le top2000 global. **773 exacts**,773 opérationnels;
  **46 gains/1 perte**, net+45 contrev21. Contrôles511/512. Aucun modèle
  ni poids changé. Ni score ancien ni quota ancien/nouveau garantis.
- V23 : toujours même réservoir; parent appris OOF avec noms officiels et
  preuve conjointe site. Master **822 exacts/824 opérationnels**;
  **78 gains/29 pertes**, net+49 contrev22; contrôles512/512. Formule globale
  schématique : preuve directe RRF +1/(60+rang_parent_OOF+rang_site_conjoint).
  Aucun modèle SIRET sur le grand réservoir; max2000 retenus. V23 reste
  référence de recherche, pas produit validé.
- V24 : mélange de contributions v22/v23, poids scalaire calibré en
  inner-OOF (pas sur le fold cible), mêmes références et modèles. **823
  exacts/825 opérationnels** :6 gains/5 pertes contrev23, net+1; non promu.
  La calibration optimisait un objectif parent puis transférait le poids
  aux scores site : hypothèse de substitution, pas optimisation directe
  démontrée du SIRET exact@2000.

V24 attribue211 pertes :132 parents absents des preuves,3 sites non développés,
75 évictions versMaster2000,1 absence catalogue. Le réservoir contient898
exacts :136 absences naturelles ne peuvent pas être corrigées par un meilleur
tri seul. Ne pas réinventer un mélange global ou un quota déjà essayé.

## Audits récents sur les132 parents manquants

V25 : tous132 ont le parent et le SIRET exact indexés localement.88 sont
atteignables par au moins un canal textuel si l'audit filtre sur le parent
humain (ce n'est PAS une génération naturelle);44 ne matchent aucun des dix
canaux SIRET testés. Numéro exact AND adresse-caractères :0 nouveau parent,
0 nouvel exact. Parmi119 requêtes avec numéro/adresse, seulement2 parents
ont un site local au numéro demandé,1 le SIRET exact. Pas de durcissement
du numéro après ce résultat.

V26/27 noms : pas d'omission entre portfolio officiel et documents globaux;
ensembles locaux concordants avec10018 documents effectivement lus dans
Tantivy. Pas de perte des champs SIRENE courants autorisés bruts→dossier→
portfolio, ni RNE canonique→dossier→portfolio. L'extraction du brut RNE reste
non auditée exhaustivement :177 Go décompressés, pas d'index fichier/SIREN
retrouvé.8 égalités CRM avec des anciens noms absentes du portfolio sont
déjà présentes dans les sources du canal temporel existant; aucune parmi
les132 parents manquants. Chez ces132, zéro égalité nominale CRM dans les
champs SIRENE courants/historiques examinés, seulement3 avec token commun
dans le portfolio. Noms historiques, sites, relations/successions existent
déjà comme canaux; leur simple ajout n'est pas une proposition nouvelle.

## Géographie et pilote V28

BAN résout le CRM à partir des seuls champs CRM, mêmeINSEE ouCP, numéro
exact puis meilleure voie Jaro-Winkler.866/1034 résolues,550 non déclarées
faibles,316 faibles;168 non résolues. « Non faible » n'est pas précision
garantie; proxy d'incertitude non calibré. Aucun X/Y brut réexporté.

V16 géo génère les4096 sites SIRENE valides les plus proches dans la commune
CRM, puis512 parents; distance prioritaire, numéro/voie surtout ex aequo.
Audit de tous les sites du parent humain, sans restriction de commune :
12 des132 parents manquants ont un site à≤500m, zéro à≤100m;61 sans distance
exploitable. Ces12 ne sont pas bloqués par une frontière :10 après le
top512 parents (rangs618–3465),2 exclus avant4096sites. Distances154–493m.

V28 : même pool4096, même génération, aucune injection.21 features légères
SIRENE nom/adresse, distance/rang/nombre de sites/état et qualité BAN. Champs
de personne exclus; pas de noms RNE au rôle ambigu. Nom×voie calculé sur
le même site, pas de transfert d'enseigne entre frères, pas de numéro dur.
Un XGBRanker parent pairwise,240 arbres profondeur6, hyperparamètres v23,
trois fits OOF. Positif déjà dans le pool+jusqu'à512 négatifs naturels pour
entraîner; les requêtes sans positif restent au dénominateur d'évaluation.
Même top512 parents par score OOF; aucun nouveau Master ou top100 produit.

Résultats V28 sur1034 :
- Parents retenus457→509, **74 gains/22 pertes**, net+52; plafond pool553.
- Difficultés131→170 (56 gains/17 pertes). Contrôles326→339 (18 gains/5 pertes).
- Gains nets parfold :+6,+12,+34.
- **7 des132 parents absents multicanal** nouvellement retenus (17 accessibles).
  **3 des12 absents proches** retenus (10 accessibles).
- Présence de SIRET exact derrière les parents sélectionnés389→424;
  opérationnel410→445. Ce sont des PLAFONDS D'EXPANSION, pas une sortie
  SIRET plafonnée, pas des gainsMaster ouRecall100. Sur les7 nouveaux
  parents,5 cas ont déjà le SIRET exact dans le pool spatial.
- Recette rejetée selon le critère PRÉENREGISTRÉ « zéro perte de succès
  géographique parmi les contrôles ». Pas de retuning ni d'intégration.
- Trois modèles rejoués : scores exactement identiques;82 tests passent.
  Les neuf artefacts modèles/scores/splits ont été conservés. Un bilan a
  été finalisé sans refit après contrôle de l'unique absence catalogue,
  conservée comme erreur exacte/opérationnelle. Aucun résultat changé.

Vérification ultérieure read-only : les22 parents/SIRET perdus par le canal
géo sont tous dans l'ancien réservoir naturel. Sur les5 contrôles perdus,
les5 SIRET exacts ont aussi un témoin DIRECT venant d'un canal textuel.
Sur les17 autres pertes géo,5 parents et2 SIRET exacts ont un tel témoin
textuel direct. Leur survie au nouveau Master2000 n'est PAS mesurée.

## Proposition actuelle de l'assistant, à remettre en question

Juger désormais le résultat end-to-end : remplacement de la sélection géo
par les scores V28 figés, autres canaux inchangés, expansion naturelle des
parents remontés, sélection globale max2000 à recette fixée AVANT run,
comparaison appariée SIRET humain exact et opérationnel avec gains/pertes.
Pas d'empilement illimité des deux canaux, pas de gate top100 annoncé.
Les5 régressions géo des contrôles pourraient être couvertes par le texte,
mais il faut mesurer les évictions sous budget, pas le présumer.

## Questions demandées

1. Diagnostic priorisé : où sont les erreurs de méthode ou de priorité,
   notamment rejet au niveau canal, vieux/nouveaux candidats, objectif parent
   versus exact SIRET, biais du panel et contraintes2000/500/100 ?
2. Recommander UNE prochaine expérience courte, falsifiable et exécutable
   sur Mac : hypothèse, baseline exacte à utiliser (v23,v24 ou autre),
   construction/expansion/sélection sans labels, apprentissage/calibration
   OOF si requis, budget, métriques et critères d'arrêt. Ne pas proposer
   de scorer tous les SIRET du grand réservoir avec LambdaMART malgré le cap.
3. Que garder précisément de la piste ancien/nouveau ? Qu'est-ce qui est
   déjà acquis, qu'est-ce qui a échoué, quel piège ne pas reproduire ?
4. Le plafond de génération doit-il passer avant les75 évictions ? Comment
   arbitrer sans un nouveau mois d'audits et sans inventer une promesse99,2% ?
5. Trois risques majeurs qui invalideraient un gain apparent, et la preuve
   minimale nécessaire avant un calcul national ou une revendication de
   généralisation. Fold1 reste fermé : ne pas demander de l'ouvrir pour tuner.

Réponse attendue : avis direct, concret, environ1000–1600 mots maximum.
Si les données ne permettent pas une décision, dis exactement quelle mesure
manque; n'invente ni performances futures ni faits issus du projet.
