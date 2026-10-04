# V217 — une recherche initiale native, preuves corroborées

Hypothèse nouvelle après audit216, pas modification214/215. Remplacer les
quotas et BM25 de214 par une métrique compilée dans les postings natifs,
collecteur unique100 sur TOUT le corpus43896818 établissements. Pas pool fixe
puis reranking, pas modèles aval. OOF234 uniquement,128 contrôles inchangés,
GTfermée jusqu'au scellementALL128 puis évaluateur218 séparé.

Réutiliser SANS FIT les11 poids nonnégatifs OOF et les8 termes sémantiques
query-only de78/77 (entités du fold tenu exclues du vocabulaire, mêmes128CRM,
aucun des128IDs de requête dans le training). Poids/vocabulaire/projections
figés historiques, jamais entraînés sur les21 ratés. Leurs paramètres
calibrent une recherche initiale sur postings, pas un ranker de candidats.
Nouvelle sémantique nonvalidée : aucune garantie99,2% ou généralisation.

Preuves nom/adresse : moyenne de présences de mots/ngrammes, maximum entre
vues officielles ; scores bornés, pas fréquenceBM25/IDF des adresses rares
dominant les noms. Bonus de coïncidence nom/adresse/numéro/commune selon le
head OOF11 existant. Un nom officiel courant COMPLET inclus dans un segment
contigu du CRM compte comme preuve de nom (exactfields natifs3). Pas dictionnaire
d'alias CRM, tous spans admissibles testés, pas top-k de spans. Au moins deux
mots hors particules/formes juridiques, ou un mot unique long>=4 ; le libellé
complet reste essayé. Pas initialismes arbitraires ou synonymes appris des21.
Cette preuve remplace le composant exact borné, pas un bonus doublé.
Lorsque tous les mots CRM informatifs sont corroborés par les champs courants
parent ET site du même document SIRET, ajouter le coefficient OOF name_at_place
si la place concorde. Preuve de contexte parent/site distincte, PAS un alias
exact inventé, pas mélange d'enseignes de frères ou assertion de covalidité
historique. Les identifiants ne servent jamais à cette similarité.

Noms officiels/historiques packed210 + propriétaires exacts et toutes
successions1331hop restent accessibles. Leur scope est corroboré par maximum
des preuves adresse ou place, grades fixes1 et0,5, jamais identité fusionnée.
L'ancrage packed continue la conjonction210 stricte, limite publiée : les
fragments ajoutés concernent les exactfields natifs, pas tout le packed corpus.

Géo : parcourir TOUS43896818 XY natifs officiels. Anneaux disjoints25/100/300/
500/1000/2000m et dernier rayon>=3sigmaBAN ; score par anneau
1/(1+borne_sup/500), ×0,5 siBANfaible. Approximation en marches DECLAREE,
pas distance continue ni nearest100. Bonus souple dans la requête globale :
hors2000m/XYabsent/BANincompatible reste retrouvable par les preuves natives.
Pas adresse physique substituée au GT, ni équivalence opérationnelle inférée.

Un seul collecteur natif100, ties canonisées seulement parmi les retournés.
Tous les canaux servent la même métrique avant collecte, aucune allocation
de places ni liste4×100 réordonnée. Fonction commune score réellement nouvelle,
pas équivalence avec101/214 revendiquée. Pas BODACC/TUP exhaustif, dense ANN,
append absent du natif ou nouveau modèle téléchargé ; limites honnêtes.

16 workers8 requêtes, moteurs propres fermés entre requêtes, quiet30,
RSS2Gio/pression1/free>=50/query20(workeropen inclus)/worker80/un thread.
Batch8 est une organisation physique bornée par les MEMES guards, pas gate
qualité contourné ; tout dépassement STOP, sans recyclage/retry/partiel mesuré.
Moyenne16 descriptive. FULLhash/epochs utilisés avant/après et globaux,
indexmanifest/meta/inventory seulement, PAS FULLhashsegments.

Objectif final inchangé99,2%exact/Wilsonlower99%/folds98,8%/SLO1s/2s,
allhuman sans filtre ; exact/op séparés et sidecar scellé. Ce128 réutilisé ne
peut jamais READY. Pas fold0, fold1 fermé, downstream gelé, Mac/CATNAT
seulement sans dépense. Code/contrat/config/évaluateur commités avant mesure.
