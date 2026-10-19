# Étape 2 — Gemma adapté CRM ↔ établissement

## Pré-déclaration avant tout apprentissage

Autorisation : brief Nathan `sireto-finetune`, 19 octobre 2026. Recherche V142
et candidats G0 immuables, aucun label corrigé, aucun jeu fermé ouvert, aucun
web, poids antérieurs adaptés interdits. Développement, aucune promotion.

### Population et découpage unique

Banc G0 scellé : 17427 dossiers / 11483 entreprises, folds2/3/4, exclusions
héritées optionD `7b0005a8`/G0 `9a8ecdd6` (pilote405 entreprises, test4323,
control962, folds0/1). Lecture des seuls artefacts admissibles et de leurs reçus.
Le cache BASE268697 couvre seulement3000 dossiers : compléter les textes pour
TOUT le banc, jamais réduire l'évaluation selon succès ou budget.

Permutation NumPy des SIREN triés, graine20261019 : ceil(20%) entreprises en
évaluation, reste80% apprentissage. Dans les80%, réserver ceil(20%) entreprises
à la calibration de SUR ; reste = proper. Le gradient de Gemma ne voit que
proper (environ64% du total), jamais calibration/évaluation/pilote. La séparation
interne est nécessaire pour ne pas régler SUR sur des entreprises vues par
l'encodeur. Le ranker top1 utilise les80% ; le ranker de confiance utilise
proper et son seuil de marge est choisi sur calibration, puis gelé. Même
partition et mêmes volumes pour BASE/adapté. Les526 servent exclusivement à
une confirmation descriptive finale, pas au choix des hyperparamètres.

### Textes et entraînement

Textes EXACTEMENT construits par `step2_colocated_signals.metadata` : nom CRM,
raison sociale/usuel/enseignes + libellé NAF officiel côté candidat ; mêmes
prompts, pooling masqué/L2, max512tokens et troncatures explicites comptées.
Identifiants uniquement jointures/groupes/exclusions, aucune feature d'identité.
Source BASE locale BF16 gelée ; LoRA neuf r8/alpha16 Q/V/projection, A Kaiming
seed fixe/B zéro, aucune lecture d'ancien checkpoint adapté. Mécanique V2
`cached_objective`/InfoNCE gradient cache réutilisée directement ; micro2,
batch16, température.05, AdamW3e-5, decay.01, clip1, warmup3%, cosine vers10%,
2epochs maximum, pas de recherche ni sélection de checkpoint sur évaluation.
Dernier checkpoint au budget, aucune early-stop sur les chiffres externes.

Un positif = candidat humain exact déjà dans G0. Si absent, texte manquant ou
sans négatif sûr : pas de gradient pour cette ligne, compte explicite ; toutes
ces lignes restent dans les métriques. Quatre négatifs propres à la requête,
priorité même adresse puis même SIREN puis rang G0, rotation déterministe par
epoch. Masquer les positifs union, textes identiques au positif, et candidats
d'entreprises calibration/évaluation/pilote. Aucun négatif inter-requête : le
masque de la perte ne laisse que le positif et les négatifs propres à l'ancre.
Les sites même SIREN non union ne sont utilisables que si texte distinct ; pas
de faux objectif de séparation de sites indiscernables par ces textes nom-only.
Aucun positif ajouté au pool ; aucune vérité de l'évaluation pour miner.

### Budget, reprise et smoke

Un seul gros calcul local, workers CPU/Metal séquentiels. Total Metal <=4h
(chargements/smoke/encodages BASE complémentaires/adaptés compris, pauses du
superviseur hors budget) ; fit <=1h, réservant3h aux encodages complets. Si le
budget est insuffisant, publier STOP/incomplet, pas une métrique partielle.
Garde RSS+driver7.5Gio, sous8Gio ; MPS fraction.2. VM hydraulic intouchée,
aucune commande de contrôle : admission/arrêt propre du seul worker détenu si
VM active ou non identifiable et free+inactive<10Gio. Attendre aussi un autre
worker V142>4Gio. Attentes uniquement dans superviseur détaché.

Checkpoint atomique tous25pas et aux premières étapes, optimiseur/cursor/LR/
liaison SHA conservés ; shards1024 encodés atomiquement et hashés, reçus par
étape, reprise vérifiée sans réentraîner les étapes achevées. Au plus12 reprises
resource-only ; erreurs de code/données/numérique fatales. Smoke réel avant
run complet :60 entreprises admissibles par hash, mêmes proportions ; deux
pas de gradient, encodages et ranking/SUR/confirmation descriptifs sur données
réelles ; ses poids ne sont jamais repris par le run complet. Pas de tests
unitaires, aucune installation. Une seule attente bloquante du smoke, calcul
complet détaché. Namespace `H/step2_gemma_finetune_20261019`.

### Évaluation et décision figées

Recette85+S1+S2, mêmes paramètres ranking R3 que le banc ; seule substitution
des deux features S2 (cosine, delta au meilleur de la même adresse). S1 NB
crossfit3 par entreprise DANS CHAQUE fit ranking comme le code voisin ; les
scores externes utilisent un NB ajusté uniquement sur le train correspondant.
BASE et adapté réentraînés sur les mêmes80%, pas comparaison au vieux OOF3000.

Top1 exact/union, n/N et Wilson95, global/par fold2/3/4, disponibilité@100,
absences comptées erreurs. Delta apparié adapté−BASE, IC95 bootstrap10000
entreprises entières/graine fixe, gains ET pertes. Famille même adresse = masque
figé `c37a1a7b` (1046 sur17427), intersection de l'évaluation, jamais sélection
pour fit ; publier effectif, delta et IC. SUR : marge, cible empirique interne
union>=98%, couverture maximale/ex æquo indivisibles via helper existant,
aucun minimum ni garantie Wilson ; seuil sur calibration seulement. Précision
exacte/union, couverture et IC externes ; zéro SUR => précision indéfinie.

Anti-par-cœur : top1 cosine seul proper vs évaluation, plus ranking train
(in-sample, S1 crossfit) vs évaluation ; écart descriptif/optimiste, pas preuve
OOF du train. L'encodeur est explicitement non-OOF à l'intérieur des80%.

Confirmation systématique526 : content entraîné sur les80% seulement, mêmes
100 V142, puis empilement21 features et confiance groupée du code voisin,
BASE et adapté. Les526 sont déjà consommés, stack utilise leur train de pli ;
aucun chiffre indépendant ni réoptimisation. Les poids des526 ne remontent pas
au content/encodeur. Recommandation unique : poursuivre l'audit hors production
uniquement si borne basse IC95 delta UNION externe>0 ; sinon conserver BASE.

Publication automatique `docs/step2_gemma_finetune.{json,md}`, rapport français
<=30lignes (durée réelle, BASE/adapté, SUR,526, recommandation), commit isolé
avec `handover.md`. Sur STOP, publier raison/étapes mesurées sans inventer les
résultats manquants. Sources/implémentations/protocole/config scellés ; journal
ou livrable tiers sale => arrêt de publication, jamais écrasement. Aucun push.
