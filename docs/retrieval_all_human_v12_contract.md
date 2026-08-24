# Contrat v12 — retrieval all-human top100-first

Statut : **préenregistré, non exécuté**.

Ce contrat décrit le prochain cycle scientifique. Il ne constitue ni un
résultat, ni une autorisation d'ouvrir le fold 1. Il remplace, pour ce cycle
uniquement, l'enchaînement v11 qui arrêtait toute admission tant que
l'Oracle@2000 n'atteignait pas 99,5 %. Les artefacts et résultats v11 restent
immuables et doivent continuer à être cités comme historiques.

La métrique produit est le Recall@100 SIRET humain exact. Le pool de 2 000
reste un plafond de couverture et de diagnostic : il n'est ni une sortie
produit, ni une preuve que ses candidats sont classables dans les 100.

## 1. Population et vérités

La cible principale reste `human_ground_truth_siret` :

1. `ground_truth_siret` s'il contient exactement quatorze chiffres ;
2. sinon `historical_ground_truth_siret` s'il contient exactement quatorze
   chiffres ;
3. sinon la ligne est exclue faute de SIRET humain exploitable, et publiée
   dans la liste des absences du fold concerné.

C'est la seule exclusion autorisée. `MATCH_EXACT`, `AMBIGUOUS`, `UNRESOLVED`,
V2, V3, l'état administratif, la présence dans les sources officielles et un
résultat de retrieval sont uniquement des segments d'audit. Ils ne peuvent
pas réduire le dénominateur principal.

Le SIRET humain exact n'est jamais remplacé. La politique métier autorise à
terme des SIRET du même SIREN prouvés au même site, dans un sidecar label-side
séparé. Toutefois aucune preuve same-site scellée n'est disponible pour ce
premier cycle v12 : son mode est donc
`EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE`. Le builder ignore les siblings
upstream pour la cible opérationnelle et publie seulement leur nombre comme
audit non vérifié, globalement et par fold. Ils ne sont ni comptés corrects ni
neutralisés parmi les négatifs. Jusqu'à un addendum préenregistré et une
preuve same-site scellée, la métrique opérationnelle est volontairement égale
à la métrique exacte. Les deux vues restent publiées côte à côte.

## 2. Cloisonnement scientifique

- folds 2/3/4 : développement en outer OOF uniquement ;
- fold 0 : contaminé par les cycles antérieurs, diagnostic historique
  uniquement, jamais lu par un builder, un fit, un choix de cap, un seuil ou
  un gate v12 ;
- fold 1 : fermé jusqu'à une autorisation explicite one-shot après gel de
  tous les artefacts v12.

Pour chaque outer fold, tous les composants appris sont ajustés uniquement
sur les deux autres folds : ranker parent, ranker SIRET unique, routeur de
rescue et éventuel adaptateur contrastif. Le ranker SIRET score l'ensemble du
Master naturel (au plus 2 000 références), définit le PreRank@500 appris, puis
sert à l'allocation finale du top100 : il n'existe pas un second modèle SIRET
ajusté après le préclassement. Les scores parent fournis au ranker SIRET et les
cibles du routeur proviennent d'un inner OOF.
L'adaptateur textuel contrastif est en outre validé avec séparation par SIREN.

Chaque bundle parent nested-OOF scelle le hash du pool candidat v11, de son
manifeste, du manifeste des sidecars et du ledger de labels avec son manifeste.
La cascade et l'admission refusent un bundle ajusté sur un autre snapshot, même
si ses folds et son schéma sont par ailleurs compatibles.

Les seuils de routage et de rescue sont préenregistrés ou produits uniquement
par l'inner OOF. Aucun label, hit, rang de vérité ou statut de qualification du
fold externe ne peut influencer la génération, l'arrêt du chemin normal ou le
déclenchement du rescue.

## 3. Architecture top100-first

Le chemin public est hiérarchique et label-blind :

1. recherche compacte des preuves SIRET directes, des dossiers
   `(INSEE, SIREN)` et des cellules d'adresse ;
2. agrégation et classement des dossiers SIREN ;
3. expansion des meilleurs parents vers leurs établissements locaux ;
4. score LambdaMART SIRET de toutes les références du Master, puis sélection
   apprise d'au plus 500 SIRET à hydrater ;
5. allocation déterministe d'au plus 100 SIRET à partir du même score
   parent/site.

Le shortlist normal est plafonné à 500 SIRET. Le rescue peut consulter au
plus 2 000 références SIRET légères, mais le ranker SIRET doit toutes les
scorer avant l'hydratation : le nombre de SIRET hydratés reste plafonné à 500.
L'ancien ordre heuristique d'hydratation peut être conservé comme feature et
comme diagnostic, jamais comme condition d'éligibilité au modèle. La sortie
publique reste plafonnée à 100.

La sélection finale utilise un budget global. Elle ne comporte ni quota fixe
« ancres/consensus », ni plafond rigide de trois établissements par SIREN. Le
nombre de sites alloué à un parent dépend de son soutien et de l'ambiguïté
locale. La diversité entre SIREN peut être une pénalité douce ; elle ne peut
pas évincer une preuve conjointe fortement soutenue par simple règle de quota.

Le premier run OOF v12 exécute uniquement la stack officielle. Le rescue
contrastif reste une branche conditionnelle différée : il ne peut être activé
que si ce run est gelé et publie un headroom Master insuffisant, puis après un
addendum préenregistrant son bundle nested-OOF et son budget non déplaçant.
Jusque-là, une entrée `--rescue-hits` est diagnostique et non admissible à un
gate. Une fois autorisé, le rescue est exécuté au plus une fois par requête,
uniquement à partir de signaux disponibles en production. Il ne peut ni
injecter le positif, ni relire une vérité humaine, ni déclencher une
succession d'essais automatiques. Son taux et sa latence ne sont publiés que
pour ce run séparément préenregistré.

## 4. Sources et features autorisées

SIRENE reste l'autorité d'existence, d'état et de matérialisation des sites.
RNE et BODACC apportent des preuves de nom, d'adresse, de temporalité et des
relations structurées. Une relation BODACC/RNE est limitée à un saut, doit
porter un SIREN structuré et être compatible avec la géographie ou le site.

Le ranker peut exploiter :

- rangs et scores normalisés séparément par canal ;
- accords nom, adresse, numéro, voie, CP et INSEE ;
- consensus de sources réellement distinctes ;
- rang et score du parent, rang du site dans le parent et ambiguïté locale ;
- état, siège, périodes de validité et écart à la date de référence ;
- provenance SIRENE/RNE/BODACC et relations officielles structurées.

Sont interdits comme features ou lookups : SIRET, SIREN, `query_id`,
identifiant d'annonce, alias CRM, texte web privé, dirigeant personne physique
et données synthétiques. Les identifiants officiels peuvent uniquement servir
à joindre, dédupliquer, regrouper les sites d'un parent et produire la sortie ;
ils ne sont jamais présentés au modèle.

## 5. Caps et métriques

Les trois profondeurs ont des sens distincts :

- **Master@2000** : plafond naturel de couverture après tous les canaux et le
  rescue, avant éviction vers le shortlist riche ;
- **PreRank@500** : shortlist naturel réellement transmis au ranking riche ;
- **Recall@100** : sortie produit.

Toutes les métriques sont calculées sur tous les labels humains exploitables.
Une vérité absente compte comme une erreur. Aucun positif n'est ajouté à un
pool d'entraînement, d'OOF ou de test.

Avant toute demande d'ouverture du fold 1, un unique run outer OOF 2/3/4 doit
publier :

- Master Recall@2000 exact >= 99,5 %, comme cible diagnostique de plafond et
  de marge ; cette ligne n'est pas un gate produit autonome et ne peut jamais
  annoncer un `GO` ;
- PreRank Recall@500 exact >= 99,4 %, gate d'ingénierie ;
- Recall@100 exact >= 99,2 %, gate produit principal ;
- borne basse Wilson 95 % du Recall@100 exact >= 99,0 % ;
- Recall@100 exact de chacun des outer folds >= 98,8 % ;
- maximum observé de 2 000 références master, 500 SIRET pré-classés/hydratés
  et 100 SIRET publiés ;
- latence chaude end-to-end p95 <= 1 000 ms et p99 <= 2 000 ms.

Le builder scelle en outre `runtime_query_sample.parquet`, exactement 512
lignes `query_id, fold` sans vérité humaine, choisies par tri croissant de
`SHA256(query_id)` puis `query_id`. Ce sidecar est l'unique population admise
pour le profil chaud ; le profiler n'ouvre pas le ledger de labels.

La ligne Master@2000 explique le plafond. La décision scientifique dépend de
PreRank@500, du Recall@100 et de la latence. Le rapport publie également les
courbes naturelles @100/@200/@500/@1000/@2000, sans transformer ces points en
grille de sélection.

## 6. Attribution obligatoire des erreurs

Chaque échec exact doit appartenir à une seule famille :

1. `OFFICIAL_CATALOG_ABSENT` : vérité absente du snapshot officiel ;
2. `PARENT_NOT_RETRIEVED` : aucun candidat du bon SIREN ;
3. `SITE_NOT_EXPANDED` : bon SIREN présent, SIRET humain absent du master ;
4. `EVICTED_MASTER_TO_500` : SIRET présent au master, absent du PreRank@500 ;
5. `EVICTED_500_TO_100` : SIRET présent au PreRank@500, absent du top100.

La présence du SIRET humain dans le snapshot officiel est calculée dans un
sidecar label-side scellé ; elle n'entre jamais dans la génération ni dans les
features. Si ce sidecar n'est pas disponible, la métrique reste calculable
mais l'attribution exhaustive ne l'est pas : le run se termine par `STOP` et
ne peut annoncer `READY_FOR_FOLD1`.

Les volumes bruts et les taux sont publiés par fold, origine CRM, actif/fermé,
SIREN déjà vu/nouveau, site connu/nouveau et mégapole. Les vues exacte et
opérationnelle restent distinctes dans chaque segment.

## 7. Recette unique et décisions

Il n'y a ni grille, ni succession de canaries, ni réglage automatique après
lecture d'un résultat outer OOF. La recette, les caps, les features, le routeur
et l'adaptateur contrastif sont gelés avant le run. Les ablations produites à
partir du même ledger servent uniquement à expliquer ; seule la cascade
complète décide.

Verdicts autorisés :

- `STOP` si une preuve d'intégrité, de population ou d'attribution exhaustive
  manque ;
- `PIVOT_GENERATION` si PreRank@500 échoue ;
- `PIVOT_ADMISSION` si PreRank@500 passe mais Recall@100 échoue ;
- `PIVOT_LATENCY` si les recalls passent mais un SLO échoue ;
- `READY_FOR_FOLD1` si tous les gates décisionnels passent.

Une cible Master@2000 sous 99,5 % est publiée comme
`MASTER_HEADROOM_BELOW_TARGET`, sans déplacer la métrique produit ni créer à
elle seule un verdict positif. `READY_FOR_FOLD1` ne vaut pas autorisation : le
fold 1 ne peut être ouvert qu'après gel content-addressé de la configuration,
des index, des sidecars, des modèles, du code et après accord explicite.
