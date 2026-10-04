# V195 — scoring groupé national, deux recherches initiales à 100

Protocole avant mesure, corpus intégral et population128 intacts. Test physique
positions0/1, fixes sans GT, folds3/2. Deux propres flatnames193 et snapshots
adresses194 scellés par leurs vrais parents/audits, pas d'ancien candidat.
Toutes128queries/vecteurs vérifiées V192 avant sélection. Géographie/BAN,
nom, numéro/suffixe, voie, CP/INSEE, vecteur et head OWN par requête ; aucune
réutilisation de vectors[:1] pourquery1. Identifiants de contrôle, pas features.

Même formule30features182/146 et kernel183, mêmes weights/heads114 gelés,
preuves nom parent/site/historiques/onehop149/170, adresses177/181, XY et
attributs139/180. Aucun reranker aval, réservoir2000/500, fit ou positif injecté.
TOUS44403034sites/98804872facts sont scorés pour CHACUNE des deux queries.

Quatre spans PHYSIQUES contigus de tous678intervalles, plutôt que trois :
25049175/25956701/25357414/22441582faits. Fixés par array_split des intervalles
officiels, pas par nom/geo/label/candidat ; pas de changement du score/corpus.
Raison : le scoring187 de33-34Mfaits dure25-26s. Deux tels scores +20s d'ouverture
laissent peu de marge au plafond80s. Quatre spans devraient abaisser le coût
par worker, sans garantie avant mesure ni augmentation de budget. Les preuves
194 contiennent les anciens spans3 : leurs vecteurs complets ne dépendent pas
de ce découpage physique. L'oracle183 couvre déjà partitionnements multiples.

Chaque worker score ouvre catalogue AVANT les grosses métadonnées comme191,
ouvre les sources officielles UNEfois, puis crée pour chaque query un propre
PartitionFacts/cursor/grades192. Conservation float64, ordre de réduction,
égalité lexicographique et top100 exact locaux inchangés. Fermer seulement
mapsOWN, garder les artefacts. Chaque query écrit son part avant la suivante.
Le top100 global est la réduction identique des quatre top100 physiques,
sans rescore/admission apprise. Ces top100 physiques ne sont pas un pool fixe.

Un paquet commun par worker contient toutes les références/epochs une fois,
deux recettes propres et leur empreinte de contrôle commune. Capturer TOUS
facts.used, CSR/autorités/impl/plan/ledger177, catalogue/attributs, les sources
et preuves193/194 intégrales, contrôles128, descriptions, modèles et3décodeurs.
Conserver les epochs originales capturées aux étapes193/194 ; tout conflit
de hash/epoch bloque. Les targets sont regroupées sans doublons physiques,
mais aucun fichier ni contrôle supprimé. Pas trois copies de40Mo en mémoire.

Audits séparés après TOUS les quatre scores : FULLhash de toutes les cibles
fact/posting/context, files/bytes/digest/epochs, vrais parents0 et rapports153.
Ils sont liés à tous les manifest/packets et aux deux recettes exactes. Pas
de candidats admis avant ces audits, les deux réductions complètes, puis un
worker indépendant ledger93507 aprèsmerge. L'admission exige les deux vrais
parents merge/ledger0, traces intégrales et plafond100 absolu pourchaquequery.

Limites153 inchangées :1Gio/80s parworker, chaque score30s, quiet30/pression1/
free>=50%. Swap global seulement télémétrique non attribué. STOP immuable,
pasretry dansscope, paspurge/autresapps/cloud/dépense. Mac/CATNAT seulement.
Publier coût TOUT compris, y compris les preuves193/194 ; temps du run actuel
et préflight du contrôleur séparés. Cache froid/chaud inconnu. Aucun SLO validé.

V195 émet au plus deux recherches nationales initial100, PAS les128 outputs
ni une nouvelle mesure recall. Aucun label ouvert avant toutes128 sorties
scellées et évaluateur séparé. Dernier89,0625%exact resteinchangé ici. Cible
99,2%exactinitial/Wilson/fullOOF/généralisation/sidecar/SLO toujours entière,
fold0/1fermés, avalgelé. Aucun READY/SOTA ni objectif revendiqué atteint.
