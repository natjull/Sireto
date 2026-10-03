# V108 — score lexical et score conditionnel sigle, toujours initial100

Avant mesure. V107 STOP à la deuxième requête22,87s : bornes somme trop
larges, natif trié14,53s/points8,07s. Aucun128/recall. Ne pas élargir les
guards ni recycler ses sorties. La séparation naïve V107 n'est pas promue.

Décomposition exacte en nombres réels des poids V67 non négatifs :
score=max(L, C,64*N*S), où C=L+A sur le scope du prédicat pondéré sigle,
et C=0 sinon. Quand A=0 et le prédicat ne matche pas, L reste servi dans
TOUT le corpus. Quand il matche, C>=L. Un flux natif lexical global et un
flux conditionnel MUST(sigle pondéré)/SHOULD(lexical) remplacent le flux
sum V107. Aucun filtre global de commune, identifiant ou statut, aucun
pool préalable, heap100 unique. Les sources nom/site/relations/BAN restent
complètes et inchangées. Tous les sigles appris restent servis.

C est le score float32 de cette requête conditionnelle, L celui du flux
lexical. La métrique exacte V108 est max(L,C,64*N*S), pas une identité
bitwise à V104/V105/V107. Fonction mathématique pondérée inchangée,
arrondis au plus quelques ulps à vérifier par les oracles composants.
Exiger oracle top100 de cette arithmétique contre TOUTE fixture, ordre et
scores stricts, composantes V67 à tolérance, trié/points identiques.

Frontière native=max(frontièreL,frontièreC), non leur somme. Si C connu,
borne L<=C justifiée par non-négativité, à vérifier contre oracle réel.
Sinon bornes conservatrices de flux. Un composant connu dominant toutes
les bornes inconnues donne un score final exact sans point ; graph path
dominance et dominance cutoff/clé canonique restent possibles. Unknown=None,
zero seulement prouvé par épuisement. Source conditionnelle lue avant
lexicale : son absence réelle évite des points inutiles. Pas choix lié GT.

Mêmes modèles/expansions sourceV67 OOF234, poids64, données nationales,
query-only128/hash antérieur, all-human/sidecar, folds0/1/aval gelés.
16mean<=4s,20s/requête,80s/6Gio/zéro swap,90s supervision/900s contrôleur
inchangés. Évaluation seulement après128 scellées, comparateurV101. Pilote
non fullOOF/Wilson/SLO/généralisation, aucun READY_FOR_FOLD1 automatique.
