# V210 — pivot vers une hypothèse de retrieval mesurable

Protocole avant mesure. Les travaux V208/V209 sont des expériences de coût,
rejetées, pas une amélioration du recall. Ne pas poursuivre automatiquement
le tampon parent RAM suggéré dans leur handover. V210 change la recette de
recherche initiale, pas la représentation mémoire de V206. Objectif final
inchangé : 99,2 % exact all-human dans au plus100, avant tout modèle aval.

## Hypothèse

Un score unique peut laisser un type de preuve dominer les100 places. Tester
des canaux indépendants et complémentaires, alimentant directement la
sortie par tours successifs avec déduplication, sans pool fixe puis re-score :

1. texte national nom/adresse, vues natives V38 et bonus lieu souple ;
2. variante entière exacte OU tous mots du nom dans la MEME variante V134,
   parmi57530682 noms officiels, vers TOUS ses propriétaires V132 ; le nom
   de personne légale donne accès à ses sites, une enseigne de site reste
   strictement site-local. Les adresses natives classent les sites du canal ;
3. tous prédécesseurs/successeurs1hop de ces propriétaires via V133,
   sans cap de fanout ni filtre de continuité, auto-liens non expansés.
   Hypothèses officielles seulement, jamais preuve d'identité fusionnée ;
4. TOUS établissements géocodés dans300m du point BAN compatible, rayon
   élargi à max(300m,3×incertitude) si BAN faible ; noms natifs classent les
   sites. Pas de nearest100 avant texte, de rue obligatoire ou de filtre
   géographique sur les autres canaux. BAN absente => canal absent, requête
   conservée dans les autres canaux et dans le dénominateur.

Rotation fixe texte/nom_officiel/succession/géo ; doublons ne consomment pas
de place, les sources épuisées rendent leur tour. Pages physiques16, aucun
rang limite par canal, aucun callback de reranking. Arrêt à100 SIRET uniques
ou épuisement. Scores natifs conservés comme diagnostic, jamais comparés
entre canaux ni réappris. Cette allocation est une hypothèse, pas RRF ou un
top100 certifié sous un score global. Elle peut perdre des hits : les pertes
doivent être publiées et la recette rejetée si elle n'aide pas.

## Limites explicites, pas redéfinition de la cible

Le collecteur natif utilise le corpus V38 de43896818 sites, pas le catalogue
append44403034. Les noms officiels supplémentaires sont utilisés pour des
prédicats de propriétaires, mais ne rendent pas les sites supplémentaires
magiquement indexés. Toute vérité absente reste une erreur end-to-end.
L'essai de ce canal n'est PAS la solution finale tous-sites/toutes-sources.
Pas nouveau bi-encodeur, BODACC/TUP exhaustif, normalisation apprise, activité
V114 ou préférence actifs ; ne pas en revendiquer l'utilisation. Historique
et enseignes sont ceux effectivement disponibles dans les autorités utilisées.
Ce test vise d'abord à retrouver une boucle de mesure qualité utile, puis à
décider si intégrer ce canal au nouveau corpus complet vaut son coût.

## Déroulement et autorité

ALL128 CRM d'origine validées/folds2/3/4 avant sélection. Première vérification
coût sur q24, choisie car STOP de coût V206, PAS à partir d'un label/raté.
Nouvelle racine, nouvelles sorties, aucun fragment STOP réutilisé. Ce probe
ne permet AUCUNE évaluation : même s'il émet100, GT fermées. Code/config/
contrat et handover commités avant mesure, puis immuables. La génération128
et son évaluateur séparé doivent être figés dans une nouvelle version après
ce probe ; ils ne sont pas implémentés par le script210 de coût seul.

Chaque fichier de source utilisé reçoit SHA COMPLET avant et après lecture,
avec epoch original capturé avant hashing. Autorités officielles immuables
vérifiées, readonly, pas sampled/stat-only substitué au hash. Inventaire et
metadata natifs vérifiés, mais segments natifs non rehashés intégralement :
limite historique explicitement publiée, pas certificat fullhash national.
Pas de ré-audit complet des artefacts inutilisés des compilateurs sources.

Budgets153 inchangés : worker80s/1Gio, quiet30/p1/free>=50, swap global
nonattribué/nonbloquant ; query20 pour TOUT lookup+géo+graph+recherche.
Mac/CATNAT un thread, pas cloud/spend/purge ni arrêt d'une autre application.
Un dépassement impose STOP sans troncature ; pas élargissement ni retry.
Le mean16<=4s demeure prévu pour l'essai complet, pas SLO production.

GT uniquement après sorties128 scellées dans un évaluateur séparé, all-human
exact/op/V2/V3/absences, sidecar officiel et folds234 ; fold0 interdit, fold1
fermé, aval gelé. Pilote réutilisé jamais READY même100%. FullOOF/Wilson95/
généralisation/attribution exhaustive et SLO restent obligatoires. Dernier
recall vérifié reste V101114/128=89,0625% ; tests/probe ne sont pas du recall.

Appuis de recherche, sans transfert de performances :
[Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) pour tester
attributs/tokenisations lexicales ;
[UniBlocker](https://arxiv.org/abs/2404.14831) pour la complémentarité sparse/
dense. V210 n'est une reproduction d'aucun de ces systèmes, ni une preuve
SOTA. Le dense demeure une piste, pas un canal déjà exécuté ici.
