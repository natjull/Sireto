# V70 — Sparse continu, plafond16 identique train/serve

Préenregistré APRÈS le STOP V69, AVANT cette mesure. V69a a stoppé sur erreur
de vérification de manifeste alias ; correction d8552303 puis V69b STOP sur
refit non convergé du deuxième fold. Pas de manifeste complet V69 ni recall.
L'unique fold achevé n'est pas une validation globale ou un critère de choix.

Le hard Top16 V69 est discontinu : à l'égalité des16e/17e activations, échanger
les termes change brutalement le score d'un document portant UN des termes.
Ce mécanisme rend l'optimisation quasi-Newton difficile ; cela ne prouve pas
à lui seul la cause du STOP. Test de contre-exemple de continuité explicite.

Une modification seulement : v=log1p(ReLU(zW)), tau=17e plus grande valeur
de v (zéro si vocab<=16), q=max(v-tau,0). Au plus16 termes strictement positifs,
poids continus au changement de support. Gradient de tau inclus, pas de
straight-through estimé ni soft train/hard serve. Tests finite differences
vérifient loss, FLOPS, decoder, preuves natives et gradient du terme frontière.
C'est une formulation expérimentale locale, PAS SPLADE ou entmax.

Reprendre corpus, registre, nested entities, vocabulaire officiel train sûr,
bases16/16, initialisation ridge10, familles, sélection, refit et ressources
EXACTEMENT du [protocole V69](retrieval_v69_sparse_contrastive_probe.md).
Ridge0,01/0,1, FLOPS0,1, maxiter180, budget80s/4Gio/zéro swapout,2threads.
Aucun élargissement de garde ou de population. V69 artefacts et code gelés,
aucune reprise de leurs sorties partielles. Tous folds2/3/4 doivent terminer,
refits sélectionnés convergés, sinon STOP non déployable et pas de recall.

Les23 preuves gardent exact nom/adresse, numéro/commune, sigles et XY/BAN.
Limites inchangées : alias en bags dans certaines vues, pas graphes typés,
cohérence source/période ni juridique/activité ; documents à impacts fixes,
pas d'exécuteur national conjoint/SLO. Aucun candidat produit ni vérité
ouverte pour une génération, contrôle naturel512 PAS recall100. Le gate
national direct10099,2%/Wilson/OOF et tous labels restent à prouver.
