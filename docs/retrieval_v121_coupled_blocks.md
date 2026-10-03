# V121 — Recherche initiale exacte par blocs couplés de sites

Contrat avant mesure, 3 octobre 2026. Après projection V118–V120, construire
le kernel de recherche, pas une nouvelle optimisation de projection. La
sortie est directement≤100 SIRET, aucun réservoir2000/500/reranker/admission
aval. Prototype de coût/intégrité sur sample officiel, **aucun recall**.

## Score et certificat

Pour chaque site : maximum cohérent parmi ses variantes nominales officielles,
maximum parmi ses variantes d'adresse, proximité XY réelle du site,
commune/CP/numéro/siege souples. Aucun mot pris de deux noms différents pour
fabriquer une variante. Couverture=max(words,char3,char4) sur chacun des
noms/adresses séparés. Égalité courante ne peut venir d'une hypothèse de
succession ; les noms anciens restent recherchables sans être dits courants.
Masques et IDs exclus du signal textuel, pas des références/provenances.

Base19 proche V56, interactions30 V114 activité×lieu/XY prises en charge.
Mais char4/cohérence/toutes variantes changent la fonction : le transfert
des poids V114 est une hypothèse FIXÉE de ce probe, pas une équivalence de
score aux anciens moteurs. Pas nouveau fit/réglage depuis les misses.
L'activité est testée dans les oracles mathématiques, mais les grades ne
sont PAS appliqués à ce probe officiel ; pas réinterprétation des anciennes
nomenclatures. Les couples bruts nom/code sont gardés en metadata.

Blocs physiques adaptatifs XY, inconnus conservés dans d'autres blocs ;
aucun rayon, statut ou territoire ne filtre le corpus. Posting de tokens
de bloc uniquement pour MAJORER la meilleure variante de chaque bloc ;
union des tokens jamais utilisée comme preuve du site. Les bornes de nom,
adresse, lieu, géométrie et activité portent sur les MÊMES blocs. Heap100
des scores feuilles complets ; arrêt seulement si toute borne restante est
strictement sous le cutoff (ou zéro sans signal). Frontière égale explorée,
ties globaux par SIRET (IDs jamais features). Aucun offset/source relu pour
reclasser un pool. Gardes d'arrondi f64 et vérification de chaque borne visitée.

Oracles exhaustifs : mêmes ordre/score/cap, vainqueur au-delà100 dans les
deux canaux pris séparément,325 ties, permutations/tailles de blocs/BAN
fiable-faible-absente/UNKNOWN activité, formules30 indépendantes et fixtures
aléatoires. Ces oracles prouvent l'algorithme pour le corpus fourni, PAS la
couverture nationale ni le recall métier.

## Adaptateur officiel et probe de coût

Utiliser toutes les lignes du sample_current_site V118, sans sélection par
CRM/positif ; joindre les faits V120 au propriétaire officiel. Noms parent
sur ses seuls membres, enseigne de site non propagée aux frères. Tous les
faits/observations sources derrière les références restent conservés.
Ajouter4096 observations par famille SIRENE_CURRENT/BODACC/RNE pour noms
et adresses, dans un rowgroup pur médian identifié par footer. Sélection
source-only indépendante de query/GT ; originales complètes gardées.
Adresse SIREN non propagée à tous les sites, adresses SIRET explicites
courantes/historiques admissibles avec leur propre statut de courant.

Successions du sample, deux directions une seule fois, grade.5 si continuité
économique explicitement true, .25 sinon, preuve hypothétique. Pas de chaîne
juridique déclarée équivalence, ni alias exact fabriqué, ni expansion vers
des membres hors du sample dans ce probe. La couverture des liens reste
PARTIELLE ; la compilation nationale devra utiliser les autorités V84/V96
complètes. Activité/état/diffusion/adresses/champs originaux restent dans
les samples officiels ; on ne prétend pas avoir ajouté toutes les features.

4 premières requêtes CRM query-only dans l'ordre128 V115 préexistant,
folds234 exclusivement. BAN query-only existante ; aucun lookup de label.
Poids initiaux propres au fold, aucune query partagée avec TRAIN. Comparer
chaque sortie initiale au tri EXHAUSTIF de TOUT le sample ; **ne jamais
ouvrir l'évaluateur GT pour calculer un recall sur ce corpus réduit**.

Budgets80s/3Gio/zéro swapout, query20s, supervision90s, sortie≤32Mio.
STOP sans hausse de budget ni évaluation partielle. Hashes complets petits
artefacts/implémentations ; grosses sources supplémentaires contrôlées par
stat/footer avant/après, SHA nationaux déclarés mais non revérifiés ici.
Pas de compilation nationale, fit, embeddings, cloud ou dépense externe.
Pas de modèle aval modifié, fold0 non lu/fold1 fermé.

Le coût32k sites ne valide pas le SLO national ; aliases/DF/blocs/relations
et fanout à43,9M changent la difficulté. Étape suivante seulement si preuve
et coût passent : format compact persisted/mmap, dimensionnement complet
sans perte et nouvelle projection nationale, puis génération128 aveugle et
OOF complet sous le contrat actif. Pas de READY_FOR_FOLD1 à partir d'ici.
