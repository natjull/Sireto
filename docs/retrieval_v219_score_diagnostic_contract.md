# V219 — audit ponctuel des dix pertes V218, avant mesure

Recherche initiale100 et objectif exact99,2% inchangés. Cette étape n'est
PAS un run de retrieval ni une nouvelle mesure de recall. Elle examine TOUTES
les dix pertes du pilote V218 scellé, y compris les régressions12/62 face à
V215. Aucun identifiant diagnostique ne devient une source de génération.

Le programme refuse d'ouvrir labels et catalogue nominal V103 avant
`evaluate_retrieval_v218_postseal.admission` : tous128, toutes sorties,
hashes, epochs et sorties effectives des producteurs doivent être valides.
Le ledger humain original et le sidecar restent immuables, folds234 seulement.

Pour chaque perte, recompiler EXACTEMENT V217 avec ses sources officielles
et modèle OOF gelé. Interroger ponctuellement le GT et les deux références
déjà émises aux positions1/100. Reproduire leurs scores originaux avec
tolérance float32 2e-6, sinon STOP. Publier déficit GT/frontière,11 preuves
lexicales, score natif217, incréments owners/successions et radial. Ce sont
des différences conditionnelles à score natif fixé, pas une ablation de
population ni une attribution causale complète. Les noms, XY et champs GT
servent exclusivement au diagnostic post-scellé.

Trois workers propres4/4/2 sous politique214 autorisée : RSS2GiB,80s worker,
20s/requête ouverture native incluse, quiet30s/free>=50/pression1, un thread,
swap système non attribué non éliminatoire. Sources utilisées hashées
entièrement avant/après, epochs reconciliées. Native inventaire/métadonnées
seulement, gros segments NON entièrement rehashés. Aucun ancien fragment
LIVE/STOP n'est repris. Aucun changement aval, fold0/fold1 fermés.

Les cas diagnostiqués sont sélectionnés par label APRES V218 : ils ne valent
ni recall national, ni généralisation. Aucun dictionnaire de leurs noms ni
correction de labels. Une amélioration ultérieure nécessitera nouveau contrat,
nouvelle génération128 labels fermés et évaluation séparée post-scellé.
