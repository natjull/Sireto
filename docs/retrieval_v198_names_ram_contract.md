# V198 — compteur lexical exact en RAM, limite de query inchangée

Protocole avant mesure, suite au STOP réel de V197 sur la position4. Le
runtime V187/V197 impose **20 s par query de preuve**, pas30. Cette limite,
le worker80 s/1 Gio et quiet30 restent inchangés. Aucun patch, retry ou
réemploi de fragments du scope STOP V197. Les 128 sorties restent dues.

V170 ouvre un fichier mmappé de petits compteurs entiers pour chaque mise
à jour de postings puis chaque scan de channel. V198 conserve exactement
ces compteurs mais en RAM lorsqu'ils occupent <=64 Mio : dtype déterminé
par le nombre MAX de tokens de la query, pas de saturation, même totalité
des tokens et postings uniques/sortés, division float64 et maximum des trois
canaux par variante entière. Couverture float64 et flags bool restent des
fichiers privés. Les compteurs plus gros utilisent V170 tel quel : fallback
physique, jamais troncature de tokens, aliases, propriétaires ou candidats.

RamCountersNames conserve les constructors/certificats/namespace V150/V170,
patches parent V149, distinctions parent/site, toutes les références de
propriétaires et les mêmes liens de succession1hop à partir du snapshot
figé, mêmes grades .5/.25, sans nouvel exact ni propagation2hops. Sa méthode
query ne change que l'appel de coverage, et conserve le même guard global
20 s pour lexical+scatter+relations+stats. Il ne réinitialise pas ce timer
entre étapes. Les checkpoints natifs publient leurs coûts, sans texte CRM.

Tests : comparaison bit-exacte indépendante de chaque alias aux kernels
V138/V170 et du snapshot parent/site/relations au scalaire V150, sur tous les
chunks, trous, doublons de liens, aliases courts/vides et plus de255tokens.
Source/header/type/posting incorrect, compteur insuffisant, timeout et
source dérivée modifiée doivent toujours être rejetés. Fallback disque
testé, aucune feature business synthétique dans la production.

Probe de coût fixé à la position4 parce qu'elle a échoué dans V197, PAS pour
choisir une variante selon le recall. Valide d'abord la population128, son
ordre/vecteurs/heads/folds2/3/4. Source nationale COMPLETE :57530682aliases et
44403034sites, mêmes catalogues officiels. Pas de GT, ancien candidat ou
fragment STOP, ni score/reranker/admission100. Une phase produit les preuves
complètes de cette query ; une phase séparée hash FULL tous les fichiers
source+preuves après lecture, avec vrais parents0 et epochs avant/après.
La comparaison scalaire est un test, pas une garantie du recall national.

Temps source/query/flatten/audit, RSS et cache non contrôlé publiés. Mac et
CATNAT uniquement, aucune dépense/purge/autre application arrêtée. Si ce
probe STOP, garder son scope intact, pas hausse20/80/1Gio ou réduction corpus.
Il ne valide ni recall, SLO, OOF complet ou READY. Suite : nouvelle génération
entière128 utilisant la même formule, preuves et source, avant labels séparés.
