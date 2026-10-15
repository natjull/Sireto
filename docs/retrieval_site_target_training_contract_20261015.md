# Cible d’apprentissage « site raccordé » — 15 octobre 2026

Contrat préenregistré avant toute mesure. Décision Nathan du 15/10/2026 :
apprendre à viser un établissement de l’entreprise humaine au site CRM,
sans corriger, remplacer ni réécrire aucun label. Périmètre : 526 requêtes
de développement V142, folds 2/3/4 exclusivement. Aucun jeu fermé, web,
modèle aval, promotion produit ou recalcul complet autorisé ici.

## Règle unique de cible, réservée à l’apprentissage

Réutiliser les fonctions `address_match`, `crm_point`, `candidate_point`,
`distance`, `threshold` de `scripts/retrieval_site_operational_metric.py`
et son contrat du 14/10/2026, sans réécriture ou assouplissement. Sources
figées identiques : CRM/BAN v17, stock SIRENE août 2026, géolocalisation
SIRENE juillet 2026 avec précision typée. Adresse concordante OU distance
≤200 m si les deux points sont numéro, ≤500 m sinon si admissibles ;
commune/inconnue : adresse seule. Absence de preuve n’est pas erreur humaine.

Pour chaque requête :
1. si le SIRET humain vérifie la règle site, conserver ce SIRET, même fermé ;
2. sinon scanner tous les établissements du même SIREN dans le stock officiel,
   indépendamment des sorties V142 ; choisir parmi ceux vérifiant la règle :
   actif avant fermé (état inconnu après fermé), puis adresse concordante
   avant distance seule, puis distance croissante (absente après finie), puis
   plus petit SIRET lexicographique ;
3. sinon conserver le SIRET humain, y compris s’il est absent du stock.

Le mapping exporte séparément label original et cible d’apprentissage,
branche, preuves, état et siège du label et de la cible. « Dont sièges »
compte les labels originaux sièges parmi les cibles changées ; le nombre de
nouvelles cibles sièges est aussi publié. Aucun nom/adresse brute nécessaire.
Identifiants : jointures, supervision et groupes uniquement, jamais features.

## Apprentissage inchangé hors cible

Partitions internes par SIREN et graine V32 inchangées :
`sireto-native-calibration-20261012-v1`, SHA256(seed+'|'+SIREN), premiers
8 octets big-endian modulo 5. Réutiliser assignments et listes train/heldout
scellés des modèles V71 ; mêmes exclusions des entreprises de contrôle Gemma.
Ces partitions internes 0…4 ne sont pas les folds fermés 0/1.

**Concurrents re-minés avec la même procédure** : jusqu’à 12 forts par
signal, 32 uniformes (graine `training-negatives|query_id`), lexical100
historique et contexte parent/administration101. Le positif devient la cible
ci-dessus et seul ce positif est retiré des négatifs. V32 utilise ses 22
signaux numériques ; V71 re-mine sur ses 51 signaux, comme avant. Les valeurs
publiques préexistantes sont réutilisées, la cible absente de l’univers a
ses signaux numériques nuls comme dans le code historique, jamais injectée
dans un univers de recherche. Même pondération commune, standardisation RMS
pondérée train-only, logistic pairwise non négative L-BFGS-B, L2=0,01,
maxiter500, ftol1e-10, gtol1e-7. V71 gèle les 28 nouveaux poids V32 et ajuste
seulement ses 30 coefficients résiduels avec la marge de base fixe.
Aucun réentraînement Gemma et aucune nouvelle hyperparamétrisation.

## Mesures autorisées et évaluation future

D’abord mapping exhaustif lecture seule, nombres branches/changements/sièges
et folds. Ensuite un seul modèle interne (partition 0), V32 puis résiduel
V71 ; chronométrer préparation des paires séparément des optimiseurs.
Génération native nationale ≤100 avec ces poids pour 20 requêtes : tirage
`random.Random(20261015)` depuis les query_id triés ; réserver d’abord trois
requêtes de grandes communes (les dix communes ayant le plus de sites dans
le cache officiel préparé), puis tirer 17 autres sans remise. Pas de choix
selon labels, succès ou cibles. Modèle interne 0 fixe pour ce chronométrage,
pas de sélection label-side du modèle, ni prétention OOF sur ces 20.
Aucun label ni mapping dans le worker de génération, aucun recall calculé.

Après accord Nathan seulement : cinq modèles et 526 générations, puis vues
exact humain (principale inchangée), site opérationnel et union exact OU site,
côte à côte, gains/pertes appariés contre V142, Wilson bilatéral 95 % sur
comptes bruts, détail folds 2/3/4. Tous labels et statuts au dénominateur,
sidecar officiel scellé ; pas de certification nationale ni OOF complète.
Le défaut historique de sélection de modèle par SIREN humain reste explicite :
ce brief ne définit pas une règle de routage production sans label.

## Ressources et immutabilité

Nouveau namespace exclusif `H/v173/site_target_timing_20261015/` ; ne rien
écraser. Un calcul lourd à la fois, un thread si Lima hydraulic tourne,
aucun acte sur la VM. Plafond RSS 8 Gio par processus supervisé, arrêt si
dépassé, journaux et reçus conservés. Sources ouvertes en lecture seule,
SHA256 des petites autorités, taille/mtime/inode avant/après des gros stocks.
Calcul long détaché. Extrapolation basse/haute conditionnelle distinguant
caches Gemma/univers réutilisables, paires/modèles/recherche à recalculer.
Si blocage ressource/intégrité : STOP documenté, aucune estimation inventée.
Deux commits isolés : ce contrat seul avant mesure, puis scripts/mapping/
chronométrage et une ligne handover auto-référencée, sans push.
