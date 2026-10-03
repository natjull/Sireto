# V142 — Toutes adresses officielles, sans héritage fictif de site

Avant index query-side, projeter TOUS142096607 records canoniques et
TOUS43896818 sites raw courants. Un record par adresse canonique, QUATRE
variantes par site courant : ligne principale, principale complète, ligne
secondaire, secondaire complète. Total317683879 observations, même vides,
masquées ou de grain parent/non vérifié. Pas de sélection famille/date/état/
diffusion/CRM/label/hit, ni sites autour d'une vérité. Sources historiques
SIRENE sans adresse : aucune adresse ancienne fabriquée.

Chaque preuve garde ID ordinal global, ordinal de la ligne raw et rôle de
projection. Les sources originales ENTIÈREMENT hashées conservent dates,
masques/valeurs originales, flags, evidence_id et source_record_id. Le
manifeste décrit les champs exacts de chaque variante. Pas de copie de tous
ces champs volumineux ni perte de leur référence ; ils ne deviennent pas
features d'identifiant. Relecture complète source vérifie tous champs projetés
après export, avec reconstruction indépendante des sites via clés raw.

Adresses SIRET ASCII14 + SIREN ASCII9 cohérents = preuve de ce site seulement.
TOUS ces sites et parents doivent exister en V141 sinon STOP. Parent SIREN
explicite = contexte parent, lexical_eligible FALSE, site_position UNKNOWN.
Autres grains publiés contexte non vérifié, pas supprimés. Parent sans
membres conservé UNKNOWN. Jamais propagation d'une adresse parent aux sites.

Adresse canonique : raw_value prioritaire, fallback normalized_value seulement
si raw NULL/vide/whitespace. Masque formel dans raw interdit le fallback.
Normalisation V118/V128 UNE fois ; ce texte n'est pas encore un token index.
Pour raw multi-champs, composant [ND] = UNKNOWN, pas disparition des autres
champs publics de la même adresse. On joint les valeurs visibles RAW puis
normalise UNE fois, pas les valeurs déjà normalisées. Compléments, bornes de
numéros primaire, distribution spéciale, CP/communes/étranger/cedex/pays sont
dans variante complète ; ligne = numéro/indice/type/voie. Aucun identifiant
d'adresse/SIREN/SIRET n'est inclus dans texte/features.

CP/INSEE réversibles5 caractères V139, UNKNOWN distinct de00000 ; numéro
normalisé une fois. Ces champs appartiennent à CETTE adresse, jamais un CP/
numéro déduit du texte ou copié d'un autre site. Ils ne remplacent pas les
attributs primary-current V139. Pas de géocodage/XY, état, fermeture, activité
ou validité physique inventée. Dates et bool source sont dans raw ; bool
source_is_current conservé exactement pour canonique. Pour raw courant il
vaut TRUE comme source-stock, mais current_exact secondaire FALSE est une
hypothèse conservatrice de score, PAS une affirmation que l'adresse est
périmée. Hypothèse primaire TRUE inchangée V121, canonique TRUE seulement
si flag original explicitement TRUE. Champs masqués restent auditables.

Plan physique de tous row groups,8 groupes/part, jamais cap corpus ou canal.
Workers séquentiels80s internes/90s superviseur,3GioRSS,zéro swapout,
512Mio/part et32Gio total. Sources entièrement hashées init/seal ; stat et
footer complets avant/après chaque part ; dictionnaires V141 utilisés hashés
à chaque ouverture. Relecture raw complète de CHAQUE part et comparaison
de chaque champ exporté. Normalisation native vérifiée par tests et oracle
Python indépendant sur64 premières lignes de chaque source seulement, pas
revendiquée comme oracle scalaire indépendant sur317M observations.

Part STOP reste immuable, pas retry au même root ni garde relevée. Tous parts
doivent passer avant seal national. Impl/sorties/part autorités hashées.
Aucun CRM/GT/fold0/fold1/fit/aval/dépense/push/recall/SLO/READY. Ceci prépare
des preuves complètes, PAS encore l'index lexical d'adresses ni la recherche
initial100. Prochaine étape : dédup whole-variants+owners/scopes, accès
query-side, puis collecteur natif100 sans pool fixe ni reranker aval.
