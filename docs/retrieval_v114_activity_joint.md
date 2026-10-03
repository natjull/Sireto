# V114 — activité conditionnée aux preuves de site, TRAIN doublement exclu

Probe de score INITIAL, pas reranker, recherche nationale ou recall.512 scènes
TRAIN234 V52 et23749 négatifs naturels, preuves native19 réelles V56 et
descriptions2140 officielles V113. Aucun candidat émis ou positif injecté dans
une génération d'évaluation. Les targets TRAIN connus restent supervision.

Formule30 : native11 + géographie4 échelles(25/100/500/2000m) et leurs copies
fiabilité BAN, plus activité sémantique bornée0..1 x preuve lieu, numéro au
lieu, adresse au lieu et les8 preuves géographiques. PAS d'activité pure
globale/filtre de classes ou actifs. Unknown activité0 mais native/geometry
restent ; query sans BAN non supprimée. Même code/site ou libellé ne prouve
aucune identité légale. Dates/état/siège officiels restent conservés mais
pas utilisés comme heuristique de version ou préférence actif-vérité.

Décodeur choisi par le INNER du fold externe tenu dans V113, sans emprunter
le modèle d'un autre fold. Pour features du head TRAIN : CROSS-FIT par fold
source. Chaque décodeur exclut outer familles, inner familles et toutes les
familles de son fold évalué, même si elles apparaissent dans un autre fold.
Minimum16 TRAIN et présence de chaque requête exigés ; pas skip silencieux.
Pour inner contrôle, décodeur appris sur TRAIN hors outer/inner seulement.
Pour refit head final, cross-fit hors outer et fold source évalué ; pour
outer contrôle, décodeur fit final hors outer (parité bytes V113 exigée).
IDs servent à joins/exclusions/audits, JAMAIS aux features ou embeddings.

Sélection INNER uniquement : NATIVE11, GEOMETRY19, ACTIVITY30, tous même
log(1+sumexp(négatif-positif)) égal par requête et poids nonnegatifs0..64.
Regularization.005 native/.05 extras fixées avant mesure,200 itérations.
Prior19 historique V56, extras0 ; modèle choisi min(head error,mean outrank,
parameters), converge obligatoire. Refits de contrôles figés publiés aussi
sur outer, pas reselection. Erreurs/couverture/cross-fit/geometry corrompue
STOP, jamais retrait de dénominateur ni positif réparé.

Tous les champs sont monotones nonnegatifs, interaction activité x preuves
du MÊME site. Cette algèbre permet des bornes de génération, mais elles ne
sont PAS exécutées/certifiées par ce probe. Native float32 conserve ses
petits excès1e-6 du cache ; pas clip caché ou claim bitwise avec native futur.
Bornes/heap100/scope/fanouts et SLO restent à implémenter et éprouver.
La réduction exacte vers19 coefficients natifs conditionnés à l'activité
est testée, ainsi que sa borne monotone sur200 fixtures indépendantes.
Cela ne certifie pas encore des bornes de blocs du corpus réel. Segments
UNKNOWN activité (jointure absente) et preuve activité zéro (inclut connus
non pertinents) sont publiés séparément, sans changement du dénominateur.
Les exclusions cross-fit portent sur le FIT du décodeur ; sa recette
family/ridge demeure celle sélectionnée dans le INNER V113 du fold externe.

Guard80s/4Gio/zéro swap, superviseur90s, deux threads CPU ; pas encoder,
modèle/source téléchargés, cloud ou budget élargi. Sources/implémentations
rehashées avant/après, modèle final décodeur byte-parity avant seal.
Fold0 jamais lu, fold1 fermé, aval gelé. Ces contrôles sur négatifs TRAIN
ne sont PAS Recall@100 national/OOF complet. La barrière99,2% exacte
all-human/Wilson/généralisation/attribution/SLO demeure ; dernier recall
V101114/12889,0625% inchangé. Ne promouvoir aucune branche de scoring
avant vraie génération initial100 et évaluation séparée scellée.
