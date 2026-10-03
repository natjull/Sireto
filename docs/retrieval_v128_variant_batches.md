# V128 — Variantes officielles complètes par lots

Contrat avant mesure,3 octobre2026. Le prochain compiler doit projeter
l'historique SIRENE et les variantes EI réellement, pas seulement les
SIRENE_CURRENT/RNE/BODACC antérieurs. V128 accélère exactement la projection
V118 en lots Arrow, sans changement des règles de source ou de score.

ASCII natif Arrow, Unicode non-ASCII passé au normaliseur Python V118
inchangé après déduplication locale des valeurs. Aucune substitution
Unicode approchée, translittération nouvelle ni dictionnaire CRM appris.
Masques et identifiants gardent les mêmes protections. Le texte de cette
projection n'est pas directement un embedding/feature ; l'encodeur/scorer
doit toujours appliquer ses protections avant de scorer.

Chaque champ UL/site disponible, tous doublons de variantes/périodes,
contextes sans nom et propriétaires invalides conservés. EI : noms/prénoms
de la même ligne actuelle explicitement diffusible O seulement, tous
prénoms disponibles, concat de tous1–4 si plusieurs, deux ordres, avec
source_fields exacts. Pas de prénom courant joint à un nom ancien, aucune
adresse/XY historique inventée ni enseigne propagée aux sites frères.

Toutes27 colonnes OBS_SCHEMA sont conservées, dans le même ordre complet
que V118, y compris source/période/diffusion/raw/original/field/derivation/
scope/éligibilité/current. Aucun latest-period, cap ou filtrage de contexte.
Les originaux complets restent dans les autorités de samples SIRENE et
familles RNE/BODACC/SIRENE, non modifiés par la projection.

Probe sur TOUS les323597 observations du sample V118, données officielles
seules. Comparaison intégrale typée et ordonnée des27 colonnes vs autorité
V118 et vs exécution scalaire fraîche. Comparaison scalaire séparée aussi
des12288 noms de samples de familles V125 (inclutBODACC absent de V118).
Temps projection neuf/scalar appariés sur la même table en mémoire ; ordre
alterné par rôle. Temps de lecture/export séparés, cache non contrôlé. Ne
pas comparer ces temps à l'ancien worker V118 comme tâches équivalentes.

Budget80s/90s,3GioRSS,zéro nouveau swapout,32Mio sorties ; STOP si dérive
ou dépassement, pas de relance augmentée. Petites autorités entièrement
hashées. Aucun nouveau scan de gros stock ni hash complet national dans
ce probe, aucune requête CRM/GT/fold0/fold1/fit/modèle aval/cloud/GPU/coût.

Ce probe valide une primitive pour compiler en phases streaming, PAS un
corpus national complet ni recall/SLO. Dictionnaire global, owners/proofs,
postings/relations/adresses et recherche nationale restent à construire.
Dernier pilote exact@100 V10189,0625% inchangé, gates99,2% NON franchis.
