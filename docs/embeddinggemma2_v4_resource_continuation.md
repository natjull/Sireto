# 8 octobre01h50 — continuation d'exécution, pas nouveau fit scientifique

V3 : minage complet117 158textes/15 035queries, ownedexit0. Préflight réel
deuxbatches512dual backward PASS23,96s, driver1,48Go/RSS1,96Go. Training a
avancé10pas mais subi huitSIGTERM de supervision liés ressources, sans preuve
d'OOM du modèle. Le garde `level!=1` rejette aussi les alertes intermédiaires.

V4 conserve EXACTEMENT textes/paires/minage/split/objectif/LR/curriculum/seed,
checkpoint optimizer/step/epoch/cursor/exposure et deadline originales. Nouveau
namespace, aucun code/config/hash ou état scientifiqueV3 écrasé. Migration
checkpoint CPU vérifiée contre ancienlaunch puis hash nouveaulaunch tracé.
MeilleuradapterV3 également conservé, pas recyclage des pas du préflight.

Seule politique d'exécution changée : admettre30%libre, yield20%, supervisor15%,
pression0/1/2 tolérée si mémoire disponible/RSS sous8Gio ;3/4/inconnu bloquent.
MPS.4 inchangé. Checkpoint10pas. SIGTERM coopératif : sauvegarde à la frontière
du batch, supervisor45s de grâce, puis kill si nécessaire. Les raisons exactes
free/level/RSS/deadline sont enregistrées au lieu d'un seul libellé ambigu.
Budget nocturne reste08:43, entraînement06:43, pas huitheures supplémentaires.

[Apple distingue avertissement et pression critique](https://developer.apple.com/documentation/dispatch/dispatch_source_memorypressure_flags_t).
[XNU publie plusieurs niveaux internes](https://github.com/apple-oss-distributions/xnu/blob/main/doc/vm/memorystatus_notify.md).
La tolérance0/1/2 est notre politique encadrée, pas affirmation que toutes les
versions de macOS donnent les mêmes valeurs. Les RSS/free/MPS bornent l'usage.

Comparaison V4 reprend exactement la préparationV3 :80queries/119064sites,
troisablations/unionRRFinitial100, même cohort/lexical/noms officiels scellés.
Seuls chemins nouveladapter/config/night et code d'exécution sont liés àV4.
Labels toujours après trois générations complètes/ownedexits0 ; aucune nouvelle
sélection des zones ni aucun label ouvert pendant reprise. Serving/aval/test
et folds0/1 inchangés. Disque interne très plein, aucun tmpGit d'autre travail
supprimé ; données/checkpoints restent surCATNAT.
