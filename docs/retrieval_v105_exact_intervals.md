# V105 — sigle factorisé et top100 exact par intervalles

Avant mesure. V104 terminal STOP20s seconde requête, native triée11,864s,
points8,629s ; aucun128/recall. Tous guards V104 inchangés. Même coefficients16
V67, sources nationales complètes, champs/relations/BAN et path_weight64.
Aucun pool fixe ni suppression de requête/site/preuve, aucun tuning labels.

Les4 composantes sigle V67 partagent le même prédicat courant officiel.
Factoriser A*(w11+w12*place+w13*number_at_place+w14*address_at_place) :
UNE sous-arborescence de regex, pas quatre. Même fonction réelle/sources
et pondérations apprises, MAIS regroupement de sommes float32 Tantivy
peut changer quelques ulps. NE PAS promettre identité bitwise aux scores
V104. Publier cette limite ; oracles composants mathématiques à tolérance,
top100 exact entier contre le nouveau scorer natif, accès trié/ponctuel
du même scorer doivent être strictement identiques. Pas de poids nouveaux.

Après page native triée, tout site hors cache a native<=frontière. Son
score reste exactement max(native,64*nom*site). Si branche>=borne native,
score final exact=branche sans point natif. Si max(branche,borne) ne peut
battre le100e, y compris sa clé canonique, aucun point natif nécessaire.
Sinon accès ponctuel réel obligatoire. Valeur native inconnue reste None
avec borne explicite, jamais remplacée par0 inventé. Chaque score admis
dans heap100 est exact. Les preuves nom/site sont calculées exactement
sur les mêmes scopes ; liens/fanouts entiers avant avance de frontière.

Le heap ne baisse jamais : dominance prouvée reste valable. Un target
revu dans un autre flux doit respecter composantes/borne précédentes,
sinon STOP. Les égalités de score/clé supérieures ne sont pas coupées.
Les blocs/buffers ne constituent pas un réservoir reclassé. Trace sépare
targets bound-checked, score final exact et accès natif effectivement évité.

Apprentissage/expansions/query-only V104 inchangés, all-human234/sidecar,
fold0/1 fermés, aval gelé.80s/6Gio/zéro swap,20s requête,90s supervision,
900s contrôleur,16mean<=4s.128 scellées avant ledger/évaluation distincte
comparateur V101 fixé. Pilote non fullOOF/SLO/généralisation ni99,2% prouvé.

Appui API limité : [Tantivy ConstScoreQuery](https://docs.rs/tantivy/latest/tantivy/query/struct.ConstScoreQuery.html)
préserve l'ensemble des documents du prédicat. La factorisation et la
borne max ci-dessus sont notre implémentation, pas une garantie performance
du moteur ni une revendication SOTA. Oracles full-corpus indispensables.
