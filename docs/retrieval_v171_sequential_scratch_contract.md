# V171 — initialiser séquentiellement les mêmes tableaux zéro

Contrat avant mesure,4octobre2026. Métier/top100 initialV38/protections/V153
inchangés, aucun modèle aval, GT, ancien candidat ou fold0/1 ouvert.

V170 passe sur UNE première requête nationale complète : RSS956678144,
query17,197419s dontscatter10,818386s, gather44403034sites intégral.
Le coût reste incompatible avec SLO1s/2s et ne prouve aucun recall.

Hypothèse limitée : le premier scatter vers les fichiers zéro sparse frais
peut payer de nombreux défauts de pages/allocation lors des writes random.
V171 matérialise les MÊMES zéros dans l'ordre séquentiel, par fenêtres524288,
avant ces updates. Parent-coverage/parent-exact/site-coverage/site-exact,
ainsi que la relation, restent float64/bool complets, sans compression ou
troncature. Ni algorithm138/149/137, score, namespace, flags, sources,
nombre de postings/faits/relations ni preuve manquante ne changent.

Pas de preload global/caches d'autres processus, pas de purge, ni augmentation
du budget : worker+parent1Gio/80s, query20s, normal pressure/free>=50,
swap global uniquement télémétrie, scratch3Gio. Temps de mise à zéro inclus
DANS queryguard et timing total ; aucun déplacement de coût hors mesure.
Une première même requêteV115/folds234 sur corpus national complet, fullhash
before/after, gather de chaque site une seule fois, vrai childexit0 nécessaire.
Temps LOAD/LEXICAL/zero/chaque bucket/SCATTER/ONE_HOP instrumentés.

Fixtures bitexact au natif et fenêtres1/2/32/524288. Puis root171 frais,
arrêt au premier échec sans retry ni changement des codes170 mesurés.
Comparer le digest national intégral de gather à170 :
3927c45d5d912c01ab3e9967aa9c4f104b4fd1544f0796f27137a635edbe4f73.
Même quantum/ordre/site-namespace, égalité attendue AVANT toute lectureGT.
Pas d'inférence causale/garantie de gain avant mesure, aucune conclusion
99,2% : dernière mesure exacte V101114/128 inchangée, noREADY.
