# V144 — Référence canonique au lancement, moteur de projection V143 inchangé

V143 init48526 TERMINAL1. Le gate Mac a PASS30,646712s,74% free,
zéro swap dans fenêtre. Aucun worker nouveau lancé, aucun plan national
scellé. Cause démontrée : manifest V142 contient référence launch à trois
champs (path,sha256,size_bytes), config V143 deux champs. Comparaison de
dicts refuse ces autorités malgré MÊME fichier/hash et schéma export exact.
Conserver V143 code/config/doc et root mesurés inchangés.

Nouveau bootstrap V144 : verify le fichier/hash, puis record l'autorité
canonique complète. Si taille explicite fournie, elle doit correspondre
exactement. Un hash/path invalide reste STOP. Références stockées dans
launch/plan sont complètes avant toute comparaison backend. Pas de bypass
de validate, modification des preuves ni réduction du périmètre.

Moteur V143 réutilisé INCHANGÉ (load/worker/execute/seal/remaining), ses
schémas status V143 restent le format de compatibilité. Launch précise
bootstrap_revision=144, root V144 et hashes de tous sources/scripts/docs/
configs V142/V143/V144 réellement utilisés. Worker projet/audit V142
inchangé. Les75 succès sont importés intégralement, STOP74 non importé.
Sources full hash à l'init et au seal. Même317683879 observations et312parts.

Toutes protections V143 restent : batch8192, quiet30s init/1s par phase,
minimum50% memory_pressure -Q, gate55s max, counter inchangé jusqu'au Budget,
80s/3Gio/ZÉRO swapout global par worker,90s superviseur, mêmes limites
d'output. Pas de retry même root, garde levée, app fermée ni artefact ancien
réécrit. Pas de CRM/GT/fold0/fold1/fit/aval/recall/SLO/READY/push/dépense.

Tests : config minimale deux champs ET référence record complète, taille
mensongère/hash dérivé refusés ; import/sorties mêlés et dénominateur complet
passent les verifiers V143 inchangés. Aucun succès national revendiqué avant
seal complet des312 parts. Après projection, index de variantes d'adresses
et collecteur initial100 restent nécessaires.
