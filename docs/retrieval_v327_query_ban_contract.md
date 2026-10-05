# V327 — correction BAN préparée, pas exécutée

V322 a STOP avant sélection/scellement/encodage/GT : l'ancien cache BAN V34
n'est pas équivalent au serving V43 repris dans V287. La surface V33 utilisait
le numéro brut, alors que V43 passe crm_number normalisé. Sur les mêmes128
requêtes scellées, deux numéros normalisés sont vides et le pilote n'a pas de
coordonnées alors que l'ancien cache en avait. Aucun gain/perte de recall ne
peut être déduit de cette différence.

V327 prépare un helper ONLY query-side : même normalize_street, même
_resolve_ban local, même source BAN officielle SHA48f9e3d4..., même indicateur
Lambert93 V43 et flags de confiance. Aucun ajout de coordonnées/positif/alias
du label-side. Ne jamais remplacer un numéro normalisé vide par le numéro
brut ni accepter silencieusement le vieux cache. Les requêtes non résolues
restent au dénominateur et atteignables par les autres preuves natives.

Pas d'exécution nationale, pas de nouveau panel ni recherche/score/recall
produits par ce helper. Fixtures seulement. Le futur producteur doit reparcourir
les sources originales, sceller la sélection hashée inchangée, et résoudre
le nouveau panel avec ce prétraitement EXACT, puis prouver parité ALL128 avec
la géométrie V43 gelée, avant tous candidats. Encodage/moteur286/heads77/228
et limites214 restent inchangés. SHA complets pre/post, epochs, exit0/PASS
réels de chaque producteur, pas de reprise de STOP322 et pas de données GT.
Une nouvelle autorisation humaine est requise avant nouvelle préparation.
Les contrats/artefacts mesurés V322 sont immuables ; ni source317/320/321
relancée ni budgets modifiés. Tous les autres gates de V322 restent dus.

Correction de documentation de population : le V38 original portait sur les
quatre premiers lots, MAIS les128 effectivement utilisés depuis V42/V78 et
par V288/V307 sont les lots d'indices4..7 (positions128..255). Même ordre
préexistant indépendant des labels, folds234, DEV déjà consommé. Le futur
panel exclut ces contrôles réellement utilisés, pas ceux du premier V38.
