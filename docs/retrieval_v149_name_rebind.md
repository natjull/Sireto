# V149 — Réattacher les noms officiels à la namespace append

Contrat avant mesure. V137 conserve158575 faits nominaux parent non liés.
V141 a ajouté2743 parents depuis des identités d'adresses officielles. Leur
intersection doit être MESURÉE, pas supposée nulle ni supposée positive.

Auditer chacun des105564761 faits V132/V137, même déjà lié : identité UL/site,
scope, reconstruction de toutes anciennes positions. Seuls les UL MISSING
réellement absents de V136 sont cherchés dans V141. Jointure exacte de clés
officielles, pas alias CRM/GT/ID comme feature. Toute position liée reste
inchangée, toutes preuves site restent site, aucun produit alias×membres.
Les absences restantes et toutes observations d'origine sont conservées.

Un patch sparse par chacun des64 buckets, y compris vide : ordinal ORIGINAL
de fait et nouvelle position parent. Chaque export est relu entièrement,
sources originales/hash/offsets/flags conservés avant/après. Les patchs ne
changent ni les noms, ni leurs scores, ni leur état historique/courant. La
namespace contient TOUS les79745 sites ajoutés, pas ceux du CRM seulement.

`patched_positions` sert au scatter V138 inchangé sur une copie d'un bucket.
Les vecteurs parent utilisent alors toute la namespace V141. L'adaptateur
`ReboundNameSnapshot` garde V146 inchangé pour anciennes preuves/site/one-hop
et rend accessibles les noms parent DIRECTS réattachés aux nouveaux sites.
Exact courant reste le flag explicite du fait, jamais inféré d'un lien.
Cet adaptateur seul ne certifie pas la complétude d'un appelant ou son index.

Run source-only court : quiet30s, attente55s maximum, même garde80s/3GioRSS/
ZÉRO swapout GLOBAL ; superviseur90s, threads1,16MiB sortie maximum. Aucun
budget relâché ni auto-retry root STOP. Toutes autorités, artefacts utilisés
et implémentations hashés avant/après. Manifest seulement APRÈS Budget PASS.
Une intersection zéro est un résultat utile, pas un échec à transformer.

Pas de CRM/labels/folds0/1/candidats/ranker/fit/dense/cloud/dépense. Ce run
mesure la couverture de preuves et non recall, latence retrieval ou99,2%.
La projection d'adresses reste une dépendance distincte, pas contournée.
