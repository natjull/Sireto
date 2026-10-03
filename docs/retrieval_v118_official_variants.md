# V118 — Prototype borné de variantes officielles et de provenance

Contrat avant mesure, 3 octobre 2026. Étape non conflictuelle de V117 :
améliorer l'objet indexable sans changer initial100 et sans nouveau modèle.
**Aucune évaluation de recall**, corpus réduit exclusivement pour mesurer
le coût de projection, jamais pour rechercher des candidats ou publier un
score. Fold0 non lu/fold1 fermé, aucun CRM/GT/ancienne sortie consulté.

## Preuves et représentation

Six sources locales : stocks UL et établissements courants/historiques,
observations complètes de noms SIRENE/RNE/BODACC du dossier V3, liens officiels
de succession. Projection de noms indépendants : champ d'origine avec rôle,
owner SIREN/site et période réellement fournie. EI diffusibles O : nom ou
nomUsage + prénomUsuel / chacun des prénoms déclarés / suite des prénoms
déclarés, dans les deux ordres. Origines exactes des champs conservées ;
pas de dictionnaire CRM ou d'association prénom actuel + nom historique.
Pseudonyme courant, sigle et usuelles officiels conservés s'ils disponibles.

Les originaux et leur ordinal physique sont conservés dans les samples
Parquet **avec tous les champs source**, y compris adresses, coordonnées,
activité, dates, état et diffusion lorsqu'ils existent. La projection de
noms ne prétend ni scorer ni exploiter déjà tous ces champs. Le nom courant
ne reçoit pas une date de fermeture ni un début de validité inventé depuis
dateDebut. Les sources historiques apportent seulement leurs propres
champs ; aucun scaffold d'adresse actuelle ajouté à leur preuve.

Masques [ND] sans feature nominale ; numéros SIREN/SIRET supprimés uniquement
du signal lexical, jamais de la jointure/provenance. Les champs absents
restent absents. Les noms/coordonnées masqués des EI en diffusion partielle
ne sont pas reconstruits : voir [documentation INSEE sur la diffusion](https://www.insee.fr/fr/information/7456564).
Les nouvelles combinaisons personnelles sont limitées aux lignes O ; les
autres lignes/observations originales restent conservées, pas de filtre
du dénominateur ni d'inférence qu'un statut P serait entièrement vide.

Déduplication de chaînes → dictionnaire partagé ; faits distincts
(subject_kind,SIREN,SIRET,scope,name_id) → toutes les observations.
Aucun cap de noms, d'observations, de périodes ou de fanout. Aucune dernière
période sélectionnée, aucun nom courant exclu de l'historique, aucune
obligation de géographie/membre actuel/graphes. Pas de produit cartésien
aliasUL × sites, pas de propagation d'enseigne aux frères. Original des
liens de succession conservé séparément ; **expansion graphe non construite**.
Les observations invalides/vides/masquées restent contextuelles, non admises
comme preuve lexicale. Les variantes ne sont pas un sac de mots mélangé.

Contrôle de factorisation : observations complètes reconstruites puis
EXCEPT ALL dans les deux sens, égalité exacte et dénominateurs. Chaque
source row du sample a au moins une observation/context et demeure dans
son sample original. Ce contrôle porte sur la projection, pas la qualité
des données officielles ou la couverture sémantique du rapprochement CRM.

## Coût et limites avant exécution

Au plus32768 lignes par source réparties entre le début des rowgroups
premier/médian/dernier, sélection physique indépendante de CRM/labels.
Ce n'est pas une distribution représentative, ni un benchmark national.
Toutes les périodes des lignes sélectionnées sont gardées ; les autres
lignes ne sont pas projetées dans cette expérience de coût.

Budgets :80s worker,90s superviseur,3GioRSS,zéro nouveau swapout,
DuckDB mono-thread512Mo/spill512Mo, sortie totale≤384Mio. STOP sans relance
élargie. Pas d'embeddings, pas d'entraînement, pas de cloud/dépense externe.

Intégrité : hashes implémentation/config/docs et sorties complets ; source
stat + footer intégral avant/après, nombres de lignes et schémas contrôlés.
Les SHA complets des grosses sources sont **déclarés depuis leurs autorités
antérieures mais pas revérifiés ici** ; limite explicite, aucune publication
nationale ou validation goal permise à partir de ce seul prototype.

Prochaine décision fondée sur cette mesure : volume/temps observés, qualité
du roundtrip, besoin de représentation compacte/chunking. Une extrapolation
linéaire depuis trois rowgroups ne garantit pas le temps de compilation
nationale. L'index de recherche couplé lexical/géo/activité/graphes et ses
bornes restent à construire ; ne pas répéter les scans/pages V115/V116.
La question union→LightGBM reste en attente : cette étape ne l'autorise pas.
