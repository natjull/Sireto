# V250/V251 : nouvelles preuves croisées et conditionnelles, TRAIN uniquement

Hypothèse avant observation5 octobre2026, ni garantie ni choix oracle.
Un nom CRM peut contenir une rue/lieu ; ce n'est pas nécessairement le nom
légal. Donner une preuve faible de texte CRM nominal dans les adresses
officielles du MEME établissement à commune OU CP cohérent. Couverture
bornée v56 sur addresses/historical_addresses/supporting_addresses/ngrams,
MAX rôles/vues, aucune invention de nom légal ou d'alias historique.
La propagation native de supporting addresses existante n'est pas corrigée
par ce volet ; ce n'est pas une nouvelle garantie du rôle des sources.

Trois composants supplémentaires250, pas réécriture217 :
1.nomCRM→adresse officielle, place obligatoire pour ce nouveau signal ;
2.preuve exacte actuelle nom217 à place cohérent ;
3.couverture adresse à numéro ET place cohérents, confiance fractionnelle.
Tous autres chemins mondiaux217/228 restent ouverts, fermés compris. IDs
uniquement joins, aucune table de correspondances CRM/labels/features.

251 observe ces trois preuves dans Tantivy sur TOUS les points supervisés
déjà scellés232 :64 TRAIN strictement disjoints du pilote128 ; références
naturelles nationales227 + locales ancienTRAIN + positif TRAIN (ce n'est
jamais génération produit ni rappel). Pas injection de positif en serving.
Même ordre query-only227, mêmes entités/labels TRAINGT, pas128GT/fold0/fold1.
Chaque référence doit exister et conserver son vrai parent, sinonSTOP.
Composants2/3 comparés numériquement aux features actual217232 dérivées
x2×(place>0), x1×(number_at_place>0), atol2e-5/rtol2e-6 sur TOUS les points.
Composant1 réellement interrogé, pas dérivé d'une cible ou d'un embedding.
Sortie contient anciens11+fixedscores scellés inchangés, trois nouvelles
colonnes, erreurs de parité. 64/8 vrais exits0/PASS214 avant fit éventuel.
Sources FULL hash utilisées pré/post+epochs ; natif inventaire/méta/cardinal,
segments non FULLhash, absence de changement garantie par garde existante.

Gardes214 inchangées2GiB/20squery ouverture comprise/80sworker, quiet30,
pression1/free50/unthread ; pas retry STOP/capfanout/relance élargie.
Prochain fit252 seulement après251 terminal intégral. Poids des11 originaux
fixés228, apprendre seulement trois coefficients nonnégatifs hors fold/entités
held et équivalents officiels, objectif/prior figés avant fit. Aucun modèle
aval, prix externe, ou rappel/généralisation/SLO/READY revendiqué sur TRAIN.
