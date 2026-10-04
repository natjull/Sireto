# V197 — génération complète des 128 recherches initiales

Protocole avant mesure. Ce run produit TOUTES les requêtes 0..127 du panel
figé V192, dans leur ordre original et avec leur vecteur et modèle held-out
propres (folds 2/3/4). Aucun label, fold 0/1, ancien candidat ou sortie
antérieure n'est lu. Le panel reste un pilote de développement déjà consommé,
jamais un nouveau holdout, même si le recall futur est de 100 %.

Sources et calcul sont ceux du test V196 : 44 403 034 SIRET, 98 804 872 faits,
variantes officielles courantes/historiques, portée parent/site/succession
V149, numéros et adresses, XY V139, fiabilité BAN et activité couplée aux
preuves de site. Ni nouvelle relation BODACC/TUP inventée, ni attribut hérité
d'un autre site. Les établissements fermés restent accessibles. Modèles V114,
30 features, noyaux V182/V183 et formules sont figés. La sortie initiale a
un plafond absolu de 100 ; les quatre partitions sont physiques uniquement,
et la réduction exacte ne re-score jamais un réservoir sélectionné.

64 paires fixes couvrent exactement les 128 positions. Chaque paire a deux
workers de preuves (noms et adresses) et quatre workers de scoring physique.
Les sources sont ouvertes en lecture seule. Les preuves propres à chaque
requête sont conservées, contrôlées avant/après lecture, jamais réutilisées
comme les preuves d'un autre client. Les workers contrôlent réellement les
certificats, types CSR, namespaces, compteurs, parents et epochs d'origine.

Le premier scorer construit une capsule de TOUTES les références externes
réellement utilisées, avec leurs hashes et epochs pré-lecture. Elle comprend
également les sources des deux premiers workers de preuves. Les scorers
suivants exigent l'égalité de leur ensemble externe COMPLET à cette capsule ;
les workers de preuves exigent l'inclusion exacte de leurs références et
epochs. Les fichiers produits dans cette nouvelle racine sont contrôlés
séparément : exclure un fichier de la capsule externe ne l'exclut PAS de
l'audit final. Aucune capsule seule n'admet une sortie initiale.

Après TOUS les scorers, un plan scellé rassemble les sources externes, tous
les fichiers de preuves propres aux 128 requêtes, artefacts de partitions,
contrôles et certificats des producteurs. Chaque référence est conservée une
fois, sans conflit de hash ou d'epoch. Les fichiers sont entièrement hashés
en chunks physiques de <=5 Gio et <=4096 fichiers, pas des échantillons ou
des stats. Un fichier isolé trop gros impose STOP, pas suppression ou
élargissement automatique. Chaque receipt est liée au plan, donc à TOUS
les producteurs réels et à leurs contrôles de requête/source.

TOUTES les receipts FULL sont nécessaires avant les réductions à 100.
Après toutes les réductions, un ledger indépendant revérifie les 93507
empreintes V177, sources et artefacts. Admission finale : mêmes 128 IDs,
folds et recettes, chaque site/fait couvert, cap 100, tous les vrais parents
0 et receipts valides. Les fragments et sorties intermédiaires restent en
quarantaine. Un STOP est immuable, sans reprise partielle ni retry élargi.

Budget inchangé par worker : 1 Gio RSS, 80 s, score par partition <=30 s,
preuves/flatten <=30 s, pause de quiet de 30 s, pression 1, libre >=50 %.
Un seul worker mono-thread à la fois, Mac/CATNAT uniquement ; pas de dépense,
purge système, arrêt d'applications ou suppression de sources. Les contrôles
répétés font prévoir plusieurs heures pour le pilote entier : publier cette
limite, le temps total, la somme des workers, les pauses et le cache non
contrôlé. Ce n'est pas un SLO de production. Aucun nouveau plafond métier ou
de ressource n'est déplacé pour obtenir un PASS.

Le code/contrat/config/tests sont commités et cités dans handover AVANT run.
Tests avec véritables petits fichiers/hashes et oracle du noyau historique ;
contrôles de processus et pression simulés explicitement, pas des preuves
nationales. Exiger aussi absence de requête manquante/dupliquée, contamination,
source supprimée, changement de hash/epoch, receipt partielle, cap 101 ou
rescore. Les résultats historiques ne sont jamais patchés.

Aucun recall n'est mesuré ici. Les labels ne pourront être ouverts que dans
un évaluateur séparé APRÈS scellement des 128 sorties. Il devra publier exact
et opérationnel, toutes les vérités exploitables, absences, V2/V3 et
qualifications sans filtrer le dénominateur, et l'attribution officielle
exhaustive avec le catalogue courant. L'objectif 99,2 %, Wilson, OOF complet,
généralisation et SLO restent à prouver ; aucun READY ou claim SOTA. Aval gelé.
