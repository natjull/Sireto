# V72 — Observation officielle cohérente et collecteur100 SIRET distincts

Préenregistré avant mesure. Corriger la représentation, pas annoncer un
recall sur des contrôles ou modifier les résultats historiques.

Une observation conserve evidence_id, source_record_id, source, sujet,
dates, noms et adresses nested. Identifiants utilisés seulement provenance,
joins/groupes/ties ; aucun dans les termes de similarité. Noms exacts de
sources officielles uniquement, mots avec masque9/14 chiffres ; adresse
conserve numéro/commune/CP pour preuve précise.

Kernel FIXE de coût, PAS recette apprise : meilleure couverture des mots de
requête dans UN alias de nom, couverture dans UNE adresse, interaction de
ces deux couvertures dans LA MÊME observation, bonus place et numéro+place.
Nom+adresse ne peuvent se croiser entre périodes/sources, ni alias/adresses
concaténés. Pas de filtre géographique, état ou date qui exclut une vérité
fermée ou hors commune. Les dates restent métadonnées, pas encore ajustement
temporel du score. Les preuves manquantes restent absentes/UNKNOWN.

SIRET possède plusieurs observations : score=max des scores d'observation,
heap au plus100 groupes SIRET VIVANTS, ties clé lexicale. Une observation
parent SIREN n'est pas transformée arbitrairement en site, même avec adresse.
Sa non-projection est comptée explicitement ; cette ablation ne réalise pas
encore la jointure sûre légal→établissements ou BODACC→site. Pas nom propagé
entre siblings ou relation ASSET_TRANSFER prise pour identité.

Index inversé des mots de nom/adresse et clés géographiques, traversal de
TOUTES observations correspondantes, score FINAL avant collecteur de groupes.
Pas de top2000/500/pool préclassé, pas de plafond sur observations qui ferait
perdre des sites distincts. Heap compacté, état live<=100. Oracle indépendant
scanne TOUTES observations du cohort puis groupe pour comparer score/ordre100.
Ce pooling exhaustif est l'oracle de test, pas une phase de produit.

Population indépendante des requêtes/labels : premier préfixe physique32768
observations de RNE bulk et chacun des BODACC2008/2017/2026,131072 total.
Pas un corpus aléatoire/représentatif national ou sélectionné autour de GT.
Manifestes source hashés, taille/mtime/inode/footer Parquet vérifiés avant/après,
digest intégral du préfixe réellement lu ; gros fichiers NON rehashés en entier.
RNE/BODACC officiels bruts, pas portfolio déjà aplati/dédupliqué. Source/type,
noms/adresses/liaisons disponibles, dates connues/inconnues publiés par source.

8 premières queries CRM whitelist/folds234 v34, aucune lecture de labels,
teacher ou anciens candidats.8 sentinelles de coût de noms fréquents dans
les OBSERVATIONS OFFICIELLES SIRET, PAS données CRM synthétiques/train/qualité.
2 passes, ordre indexed/oracle alterné,32 recherches, parité exacte exigée.
Publier scans, groupes max, cardinalités/sites distincts, latences, build,
RSS/swap. Résultats clés candidats non écrits. Corpus RAM, cache non contrôlé,
Python référence sans bound-pruning : PAS SLO national ni nouveau recall.

80s/4Gio/zéro nouveau swapout,CPU2, superviseur90s, STOP immuable. Aucun
build/encodage national, modèles aval inchangés, folds0/1 non lus, all-human
et sidecar inchangés. Limits : kernel non appris, pas XY/BAN ni sigles,
relations/filiation/légal→site non servies, parent-only non projeté,
absence d'entraînement source/time coherent et de preuve sur corpus complet.
