# V234 : score natif initial100 appris sur ses vraies composantes

Préenregistrement4 octobre2026, avant mesure. Kernel217 inchangé,
onze poids native remplacés par233, appris sur TRAIN64 disjoint/OOF,
avec loss listwise et paramètres figés AVANT contrôles. Toutes les autres
métadonnées77 (projections/vocab/alpha/query_ids/folds) restent identiques.
23264 complets/parité/8 exits214, puis233fit exit0/PASS requis.

L'autorité de serving233 est MODELE-SEUL : références, succès réel du fit,
epochs des inputs, code et poids. Elle ne charge pas les observations
232 contenant des positifs TRAIN ni leurs anciennes listes de candidats.
Ni ledger ni cibles128 ni noms label-side en génération. IDs seulement
joins officiels/sortie/contrôles, jamais feature de similarité.

Une seule recherche native globale : onze composantes EFFECTIVES217,
bonus fixes sparse/propriétaires/successions/XY identiques ; score dans
postings AVANT collecteur100. Pas de pool fixe, score aval, quotas,
cap de fanout ou filtre géographique dur. <=100 uniques pour chacune des
128 mêmes requêtes CRM, folds234 uniquement. Pas de fit à chaque requête.

Sceller toutes128 et16 vrais exits PASS214 avant évaluateur235 distinct.
Paramètres/recette/code/config/hash figés avant émission ; aucune adaptation
aux sorties/labels en cours, ni sélection du meilleur essai par requête.
Baseline développement229/230 exacte120/128, paired gains ET losses sur
mêmes cibles128, sans remplacer les métriques historiques.

Gardes214 inchangées2Gio/80s/20s ouverture comprise/un thread/calme30s,
pression normale50% libres. STOP/partiel non recyclé/évalué. Hashes complets
des sources effectivement utilisées pré/post, epochs et inventaire natif ;
segments natifs pas entièrement rehashés. Modèles aval gelés, aucun achat.

Panel128 adaptativement consommé, aucune preuve de généralisation ni READY
même100%. OOF complet99,2%, Wilson99%, folds98,8%, attribution officielle,
généralisation et SLO toujours obligatoires,fold0 interdit/fold1 fermé.
