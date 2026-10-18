# Coût train3000 V142 — analyse en lecture seule, 18/10/2026 17:47 UTC

## Conclusion
Le gros coût est la fabrication répétée de banques documentaires **par commune**, pas l'obtention de seulement100 candidats.
Les scores finaux restent **par dossier** : réutiliser une banque ne permet pas de réutiliser les scores d'un autre dossier.
Recommandation unique : conserver les3000 scellés, partager les banques nouvelles entre lots avec reprise vérifiée ; budget préparation **160–250h utiles** après adaptation, plus génération native100 non mesurée et attentes VM sans borne. Ne pas lancer le full sur cette estimation.

## Ce qui est réellement calculé
Sources : `scripts/train3000_frozen_v142_{numeric,pipeline,supervisor}.py`, `embeddinggemma2_v{14_full_pipeline,19_proof_retrieval,21_relation_views,23_native_features}.py`.
Une zone est la valeur exacte `crm_insee` : commune, arrondissement municipal le cas échéant, ou chaîne vide (`UNKNOWN`), jamais un rayon autour du dossier.
- Geometry : résolution BAN de chaque adresse, repli empreinte de rue ; V21 parcourt43 896 818 coordonnées nationales pour tous les dossiers du lot, puis score nom/distance des sites dans leur empreinte. Géométrie et scores dépendent de l'adresse/du nom, pas de la seule commune.
- V14 : extrait **tous** les établissements de la zone depuis Tantivy (82 772 691 documents nationaux), ajoute noms officiels/entreprise/adresses, trie les SIRET ; réemploie les corpus historiques si disponibles.
- V14 `dense_zone` : construit les textes SITE officiels512 tokens, déduplique, encode Gemma768 float32 (batch8, shards1024 textes), puis fait aussi un premier scoring/top100. Les vecteurs SITE sont indépendants des requêtes ; ce premier scoring ne l'est pas.
- V19 : prépare/réemploie les textes et vecteurs PARENT officiels, encode la requête complète et son nom seul ; compare chaque dossier à chaque site/identité de sa commune, calcule nom flou, adresse cohérente, rue et distance.
- `FULL_CORPUS_NATIVE_PROOF_SCORE` désigne la boucle par128 sites qui remplit six matrices (`site,identity,name,address,street,geo`) et entretient trois top100. L'adaptateur reprend ces matrices, pas le top100 historique, pour la suite numérique.
- **`total=2554848` est une constante du pilote526**, pas une taille par zone ni le total train3000. Dans l'adaptateur `count=0` reste inchangé : `completed_units` repart de0 dans chaque zone, notification tous8192 sites. Une phase longue d'encodage PARENT peut précéder la première notification.
- V23 étend les liens de familles/successions et les preuves géographiques ; V32→V57→V68→V71→V75 compose les51 preuves, rôles de noms, géographie remplie et fournisseur. Ces matrices dépendent des dossiers et ne sont pas transférables entre dossiers.
Les sources/lookup nationaux, corpus, textes SITE/PARENT et leurs vecteurs peuvent être préparés une fois pour des entrées équivalentes ; les similarités et empreintes restent à refaire. Préencoder toute la France n'est ni déjà fait ni nécessaire pour ces3000.

## Couverture réellement comptée, sans labels
H=`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` ; E=`H/v174/train3000_frozen_v142_20261017/extension_20261017`.
Lecture de `E/{queries,sample,commune_counts}.json` et des autorités V19 `_a`/`_b` : mêmes295 corpus/banques SITE sous `H/v14/representative_20261008_b/zones/` ; identités sous `H/v18` via les références V19. Caches des526 en lecture seule.
-3000 dossiers : **1233 zones** (1232 codes + vide), dont2 zones à0 site ;5 dossiers sans code. Aucun retrait.
-**104 zones déjà couvertes /353 dossiers** (vide inclus),1129 zones non couvertes /2647 dossiers. Banques SITE et corpus ont ici la même couverture ; couverture PARENT partielle, dépendant des entreprises.
-Union des sites communaux : **20 730 848**, dont **18 611 240 nouveaux** et2 119 608 déjà couverts ; **159 235 465 couples dossier×site** à scorer avant les extensions.
-Smoke20 :19 zones,1 121 454 sites uniques,814 990 nouveaux ;2 zones couvertes/3 dossiers ;1 375 877 couples dossier×site.
-Code actuel : smoke puis149 lots20, déduplication uniquement DANS chaque lot : **2893 expositions de zones**,147 854 714 sites exposés, dont **124 853 375 à extraire/encoder**. Les banques nouvelles ne sont pas recherchées dans les lots précédents.

## Mesures et estimation, pas une promesse de délai
Références sous `E/brief10/prepare_ad3920cee85b2a761e5e3302ff95ac58f7fd48f3209b48a63a692decf14d1324_{34ffbab8eb,8fce752d2e}/` : `memory.jsonl`, `receipt.json`, `numeric/{geometry,v14,v19}/worker_state_*.json` ; aucun document de corpus volumineux chargé pour cette analyse.
-Tentative34ff : durée utile interrompue21 394,872s=5,943h, attente admission7200s séparée. Géométrie nationale :42 notifications en1,433s (scan seul, résolution BAN exclue).
-V14 :17 000,356s=4,722h entre première extraction et dernier top100 ; extraction des17 zones nouvelles313,764s, packing+encodage+premier scoring16 668,363s pour814 990 sites/731 551 textes. Même étape répétée en17 034,231s sur8fce : ce n'est pas du travail nouveau.
-Exemples zone33063 :322 756 sites, extraction102,690s, dense6601,624s ;67482 :211 137 sites,63,554s et4192,071s ;87085 :82 235 sites,33,255s et1745,046s. Coût dense typique **0,020–0,025s/site** (textes/padding variables), environ20–25s/1000 sites nouveaux.
-V19/33063 :3463,639s entre première notification de31358 et celle de33063 (transition/préparation incluses) ;1647 nouveaux shards PARENT,647 619 456octets. Encodage entre premier/dernier shard3104,578s ; boucle score33063 seulement23,732s sur319 488 sites. Proxy préparation PARENT nouvelle **0,009–0,012s/site**, très sensible au réemploi interentreprises, pas une mesure nationale.
-Contrôles historiques complets20 : numérique fidèle598,380s, activité195,198s, OSM863,022s ; ils ne représentent pas les nouvelles grandes communes. Multiplier seulement598s par150 serait trompeur.
-Projection froide, code actuel :124 853 375×[0,020;0,025]/3600=**694–867h SITE**, plus proxy PARENT312–416h : **1006–1283h** de banques seules. Scénario de budget préparation **1050–1400h (~44–58j utiles)** avec provision non mesurée pour les autres étapes, extrapolation descriptive, non borne supérieure garantie.
-Avec partage :18 611 240 sites nouveaux →103–129h SITE +47–62h PARENT (proxy conservateur ignorant les doublons d'entreprises entre communes), soit150–191h de banques ; budget préparation160–250h. Comparaisons des159M couples et autres preuves restent dues.
-Génération native100 nouvelle : pas de smoke20 achevé consulté, temps total **non mesuré** ; l'ajouter aux budgets ci-dessus. Reprises VM, I/O, disque et temps d'adaptation exclus. Le gate historique≤30h n'est pas démontré ; les banques seules le dépassent largement dans cette projection.

## Options à recette constante (non implémentées ici)
| Option | Gain estimé | Validité / précautions |
|---|---|---|
| Partager corpus+SITE+PARENT entre lots |6,71× moins d'expositions nouvelles :124,85M→18,61M ;≈85% de ce coût, pas85% de tous les scores |Pas de biais : garder les3000, sources/index/modèle/textes/ordre/dtype/runtime et bindings d'encodage/padding identiques ; ne jamais remplacer un score par celui d'une autre requête. |
| Traiter par commune / zones déjà chaudes d'abord |Même plafond de gain que le partage, non additionnel ; sans changer les tâches/caches, gain scientifique nul garanti |Conserver les lots d'encodage des requêtes : regroupement naïf change le padding Gemma et peut modifier les sorties. Garder toutes les lignes et l'ordre final ; arrêter après les seules communes faciles biaiserait le jeu. |
| Reprise par points de contrôle |Évite notamment les4,72h V14 répétées ; quatre tentatives arrêtées VM cumulent11,50h utiles sans préparation complète |Le superviseur réemploie seulement les tâches complètes exit0 et recrée un UUID après arrêt. V14 sait reprendre un shard1024 scellé, mais pas si on change de racine ; V19 adapté ne scelle/reprend pas ses matrices. Ajouter sceaux/curseurs/écriture atomique par phase et shard : perte SITE≈20–30s/shard, PARENT≈2s/chunk128, hors I/O ; jamais accepter des `.pending` incomplets. |
| Retirage privilégiant les communes gelées |Élimine le coût SITE froid pour la fraction ainsi remplacée, jusqu'à694–867h dans la projection actuelle ; scores et autres preuves subsistent |Ce n'est plus le tirage scellé :353/3000 actuels seulement couverts. Nouveau mandat/sceau et quotas/inclusion probabiliste requis ; géographie déjà vue, tailles et difficultés biaisées, capacité3000 admissible couverte non établie ici. Pondération ne répare pas une couverture nulle. |
| Réduire à1000 sans favoriser les caches |En lots actuels ≈⅓ du coût répété (ordre350–470h préparation) ; avec partage, pas÷3 : sous-échantillon uniforme1000 des3000 expose en espérance605 zones/13,68M sites nouveaux,73,5% du coût froid,≈110–141h banques |Nouveau sous-tirage proportionnel fold×taille et sceau avant résultats ; moins de diversité/précision, erreur d'échantillonnage≈√3 plus grande. Ne pas prendre le préfixe1000, ordonné par strates. |
Sans changer la recette : ne pas réduire les communes/pools, remplacer les preuves absentes par0, changer précision/modèle ou utiliser un label comme clé de cache. La fidélité des sorties après toute adaptation d'I/O reste à vérifier réellement.

## Reproduction légère des comptes centraux (lecture seule)
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json, pathlib, resource
h=pathlib.Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2'); e=h/'v174/train3000_frozen_v142_20261017/extension_20261017'
r=lambda p:json.loads(p.read_text()); q=r(e/'queries.json'); s=r(e/'sample.json'); c=r(e/'commune_counts.json'); f=set(r(h/'v19/proof_engine_20261011_b/launch.json')['corpora']); z={a['crm_insee'] for a in q}; b={a['query_id']:a for a in q}
ids=s['smoke_query_ids']+[i for i in s['selected_query_ids'] if i not in set(s['smoke_query_ids'])]
ex=sum(sum(c[t] for t in {b[i]['crm_insee'] for i in ids[j:j+20]}-f) for j in range(0,3000,20))
v=(len(q),len(z),len(z&f),sum(a['crm_insee'] in f for a in q),sum(c[t] for t in z-f),ex)
assert v==(3000,1233,104,353,18611240,124853375); print(v, 'RSS_bytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
PY
```
Vérification réelle des JSON/journaux et SQLite identité en mode RO : pic analyse≤47Mio ; aucun import moteur, calcul lourd, web, lecture de labels ou action processus/VM. Seuls ce document et la ligne de suivi sont modifiés ; aucun fichier du namespace de l'autre agent modifié.
