# V93 — supprimer la restriction aux endpoints de succession

L'audit V92 scellé sur128 requêtes trouve75 vérités hors endpoints V84,
dont9 des14 manquées. Cela ne signifie pas qu'elles sont absentes du moteur
national : seule la NOUVELLE branche noms-chemins leur donnait zéro. V91
reste114/128 exact et116/128 opérationnel, sans gain, SLO échoués.

Remède structurel, non spécifique CRM : conserver TOUTES162537492
observations de nom du dossier officiel, et les43896818 établissements de
son registre, fermés compris. Aucun filtre de graphe, commune, CP, label,
requête, hit ou owner sélectionné. Aucun alias CRM ni modèle aval modifié.
Identifiants uniquement grain, membership, provenance et futur accès.

Les rôles sont conservés : SIREN légal/usuel/commercial vers personne morale,
SIRET vers son propre site, SIREN enseigne contexte seulement. Les valeurs
vides, rôles inconnus et grains non vérifiables sont préservés avec contexte
et éligibilité explicites ; ils ne deviennent pas des preuves de similarité.
Pas transfert d'enseigne entre siblings, pas alias de succession automatique,
pas co-validité temporelle inventée. Dates, source, priorité, IDs de preuve
et ordinal d'origine restent dans chaque observation. Aucun index lexical
ni candidat produit pendant ce build, aucun gain recall revendiqué.

Budget80s/4Gio/zéro swapout, supervision90s, DuckDB1GB/2GBspill, deux threads,
sur CATNAT. Le plafond physique8Gio est préfixé pour le volume complet
162,5m contre50,1m précédemment, pas agrandi après échec. Sources et code
hashés avant/après. Cardinalités et grain du registre vérifiés ; export de
toutes observations sans jointure ni filtre, cardinalité de sortie égale
source. Vérification indépendante exhaustive d'unicité des ordinals non
encore répétée : publier cette limite.

Suite requise : facteur/postings et membership complets, puis vrais flux
initiaux100 exhaustifs/scopés, avec même corpus pour accès triés/ponctuels,
tests d'oracle, préflight aveugle et évaluation séparée128. Aucun ajustement
du poids64 selon les14 labels ni seed de vérité ni pool2000/500. Le scope
complet n'est pas une garantie de99,2% ; fullOOF/généralisation/SLO restent
indispensables. Folds0/1 fermés et modèles aval gelés.
