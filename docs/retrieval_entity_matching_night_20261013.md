# Recherche initiale à 100 candidats le 13 octobre 2026

Le meilleur pipeline mesuré conserve **503 SIRET humains sur 526**, soit
**95,63 % de recall exact brut et 94,71 % après pondération géographique**.
Il gagne huit requêtes, sur sept SIREN distincts, sans perdre une requête
retrouvée par la baseline de départ. L’objectif de 99,2 % n’est pas atteint.

Les gains viennent de preuves de relation et d’une normalisation des noms,
pas d’un nouveau fine-tuning. EmbeddingGemma reste au checkpoint V6 best1000.
Les modèles aval et le serving restent inchangés.

## Résultats comparables

Toutes les variantes ci-dessous ont terminé les mêmes 526 requêtes et leurs
sorties ont été scellées avant évaluation. La recherche émet directement au
plus 100 SIRET depuis le corpus national, sans pool fixe 2000/500 ni reranker.

| Variante | Modification principale | SIRET exacts | Recall brut | Recall pondéré |
|---|---|---:|---:|---:|
| V32 | Baseline de départ | 495/526 | 94,11 % | 93,67 % |
| V81 | Contexte fournisseur et branche lexicale indirecte | 500/526 | 95,06 % | 94,16 % |
| V88 | Contextes corporate et groupe avant le collecteur | 502/526 | 95,44 % | 94,40 % |
| V94 | Adresse propre plus stricte et nouveaux résidus appris | 501/526 | 95,25 % | 94,21 % |
| V95 | Siège et contexte de l’entreprise | 501/526 | 95,25 % | 94,30 % |
| V99 | V88 plus alignement des initiales | **503/526** | **95,63 %** | **94,71 %** |

V99 conserve aussi 507/526 SIREN et 504/526 correspondances opérationnelles.
Leurs estimations pondérées sont respectivement 95,57 % et 94,87 % ; elles
ne remplacent pas le recall SIRET exact.

Ce pilote de développement contient 405 groupes SIREN et a été réutilisé
pour plusieurs investigations. Ce n’est ni un test inédit, ni un OOF complet
de l’encodeur, ni une certification nationale. L’intervalle Wilson brut de
V99 est d’environ [93,52 % ; 97,07 %]. Le bootstrap apparié de la différence
V32→V99 est descriptif et ne corrige pas la sélection parmi les expériences.
Les folds humains 0/1, le test 4323 et le contrôle indépendant 962 restent fermés.

L’audit des jeux TRAIN figés V1 et V6 trouve **zéro recouvrement** avec les
526 requêtes et leurs 405 SIREN, aussi bien comme positifs que comme négatifs
explicites. Le checkpoint initial et les hashes TRAIN ont été reliés à leur
provenance. C’est un signal favorable contre la mémorisation supervisée de
ces entreprises ; ce n’est pas un audit du préentraînement général ni de
tous les états intermédiaires. Surtout, ce pilote reste consommé par notre
sélection de sources et de recettes : absence d’exposition TRAIN et test
inédit sont deux garanties différentes.

## Les preuves qui apportent un gain

- **CATZAM, EUREDEN et EUROVIA** : les noms de personnes morales liées
  fournissent des points d’entrée qui n’existent pas comme noms propres du
  site attendu. Le contexte reste distinct d’un alias ou d’une identité.
- **Deux nouveaux NAOS** : les références clients publiées permettent une
  branche lexicale indirecte vers les entreprises correspondantes. Le simple
  bonus numérique fournisseur, seul, n’avait changé aucun résultat.
- **SB Alliance / Fromageries du Lévézou** : le contexte de consolidation
  publié dans le rapport financier 2021 ajoute une preuve de groupe. V88
  retrouve le SIRET attendu au rang 12, sans renommer la fromagerie ni copier
  l’adresse de SB Alliance. Le document décrit une consolidation à fin 2021,
  pas une propriété actuelle ou toutes les sociétés du groupe.
- **FOURNIER / SODIMOB** : les incidences corporate publiées et les acteurs
  communs apportent un contexte atténué par le nombre de sociétés liées.
  Le SIRET fermé attendu reste admissible et entre au rang 71.
- **ARTEIS SPORT CNSD** : l’alignement des lettres CNSD avec une séquence
  d’initiales du nom légal publié fournit un signal de similarité. V99 retrouve
  le SIRET au rang 37. Aucun texte alias « CNSD » n’est créé ou déclaré officiel.

Le rapport Savencia conservé et les pages 120–121 vérifiées constituent la
preuve documentaire du contexte de consolidation :
:codex-file-citation{path="/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v69/group_reports_20261013_a/savencia_2021_fr/primary.pdf" purpose="source"}

### Ce que le test des textes BODACC permet réellement de dire

Le Gemma figé a aussi encodé 240 passages BODACC dans un diagnostic construit
à partir des anciens ratés. Ce n’est pas un index national indépendant des
labels et ses scores ne sont ni des probabilités, ni un recall de retrieval.

Parmi les 21 ratés publics qui restent après V99, 13 disposent d’un passage
propre hors administration dans cette banque. Pour trois seulement, le
meilleur score dépasse celui d’un passage associé à une autre entreprise de
la banque. Le comparateur peut inclure l’administration : ce n’est pas une
ablation contrôlée de tous les champs. Même ces trois cas n’établissent pas
une identité : INFODESCA et ATOL remontent des actes avec une adresse, et
CUISINES SCHMIDT une description générique de vente de cuisines.

L’orientation retenue n’est donc pas « encoder toutes les annonces et espérer
99 % ». C’est extraire des passages une preuve attribuée à un sujet, un rôle
et une date, conserver la distinction entre projet et opération réalisée,
puis mesurer ce canal contre les vrais distracteurs. Les contextes corporate
et de consolidation testés cette nuit donnent déjà des gains ; la similarité
sur du texte juridique brut n’a, elle, encore produit aucun gain mesuré.

## Les pistes rejetées comme améliorations globales

V94 gagne SAS 5EME ETAGE et FOURNIER, mais perd IT TRATTORIA. Pour ce dernier,
le CRM indique le numéro 1 alors que le SIRET attendu est publié au numéro 3
de la même rue. Une preuve de numéro plus forte peut donc faire monter des
concurrents sans que le bon SIRET soit exclu par une règle. Le gain FOURNIER
ne prouve pas l’efficacité de la nouvelle adresse : ses autres coefficients
et négatifs d’apprentissage ont changé aussi.

V95 gagne cinq requêtes et en perd quatre contre V81. Les cinq gains sont des
sièges, les quatre pertes des établissements locaux. La part moyenne de sièges
dans le top100 passe de 53,7 % à 73,6 %. Le bonus siège général est rejeté.
L’historique des sièges ne résout pas seul ce problème : trois des quatre
établissements perdus n’ont aucun rôle de siège dans les périodes examinées.

La préférence éventuelle pour le siège doit être une hypothèse alternative
conditionnée par les preuves du site, pas une règle appliquée à tous les cas.

## Ce qui reste à résoudre

V99 laisse 23 erreurs exactes, dont deux cibles partiellement diffusibles
qui n’ont pas été reconstituées. Parmi les 21 ratés publics, 20 n’ont aucun
mot du libellé CRM dans les champs de noms indexés du SIRET attendu. Le même
constat apparaît dans l’inventaire SIRENE/RNE/BODACC de noms légaux et de noms
propres du site inspecté : aucune nouvelle correspondance littérale manquante à l’index n’a
été trouvée pour ces 20 cas. Cela ne prouve pas une impossibilité sémantique.

Les cas se répartissent entre enseignes sans pont vers l’exploitant, groupes
ou prestataires, organismes publics, choix du siège plutôt que du site et
adresses anciennes ou imprécises. Quatre ratés trouvent déjà le bon SIREN.

### Noms sectoriels et choix du site : une lacune distincte

L’inventaire précédent ne couvre pas tous les noms de sites des annuaires
sectoriels. La source locale FINESS/Éducation/DINUM compte 410 603 lignes,
dont 204 173 portant un SIRET. La branche globale ajoutée en V32 n’utilise
que les lignes SIREN ; l’index national principal est construit à partir du
dossier SIRENE/RNE/BODACC. Les noms de sites sectoriels ne sont donc pas
exploités par cette branche. Aucun nouveau recall n’est mesuré sur cette piste.

Un contrôle de la source entière identifie 5 330 lignes FINESS dont le SIREN
publié diffère du préfixe du SIRET. Le gestionnaire peut être une relation
utile ; il ne doit pas devenir artificiellement le propriétaire du SIRET.
L’ingestion doit garder ces deux rôles séparés, comme le fait déjà le lecteur
structuré du projet.

FINESS nomme « FOYER DEPARTEMENTAL D’ADOLESCENTES TY BREIZ » le SIRET
20000554400024. SIRENE le situe au 87 rue des Chalatres, l’adresse CRM ; V99
le retrouve déjà au rang 1. Le label humain attend 20000554400073, siège
actuel au 20 rue du Port Boyer. Le nouveau nom sectoriel n’est donc pas une
preuve de récupération du SIRET exact attendu. Il ne doit pas être copié
comme alias propre du siège. La vue opérationnelle existante accepte ce cas,
mais l’erreur exacte reste comptée ; son historique complet n’est pas
réadjudiqué ici.

Sur les quatre ratés dont le SIREN est déjà trouvé, trois cibles sont des
sièges et une ne l’est pas. Pour CERGY et ADOS LOISIRS, le libellé ou l’adresse
CRM décrit aussi un site différent de l’adresse actuelle du siège attendu.
Cela renforce la nécessité de distinguer site et rôle administratif, mais
ne justifie ni un bonus siège partout, ni une correction automatique des labels.

CONVERGENCE MEDIAS appelle une revue du label : le top1 naturel possède le
nom publié et l’adresse 199 rue Hélène Boucher, tandis que le SIRET humain
appartient à INOX et porte une autre adresse dans nos sources. La publication
BODACC de 2008 corrobore CONVERGENCE au 199. Ce rival documentaire ne suffit
pas à déclarer le label faux ; aucun label n’est modifié ou retiré.
[Publication BODACC](https://www.bodacc.fr/telechargements/COMMERCIALES/PDF/B/BODACC-B_20080101_0001_p000.pdf).

Un autre pont existe hors de l’inventaire actuel : le magasin se présente
comme Gedimat Bracq, et ses mentions légales nomment LEON BRACQ ET COMPAGNIE,
SIREN 685520389. Cette preuve concerne l’entreprise, pas encore le choix du
SIRET exact ni la reproduction au cutoff du test. Elle reste un diagnostic,
jamais une entrée choisie depuis le GT dans ce benchmark.
[Présentation du magasin](https://www.materiaux-cambrai.fr/),
[mentions légales](https://www.materiaux-cambrai.fr/mentions-legales/).

## Direction retenue

1. **Conserver V99 comme référence de développement.** Préserver ses sorties
   et ses huit gains ; ne pas remplacer ce pipeline par les bonus adresse ou
   siège qui régressent sur des cas déjà retrouvés.
2. **Développer les ponts de noms par leur rôle.** Séparer enseigne–exploitant,
   prestataire–client, groupe–membre et fonction–organisme. Un nom lié n’est
   pas automatiquement le nom du site, et une relation n’est pas une identité.
   Toute compilation supplémentaire doit couvrir une source entière, être
   datée et ne pas se limiter aux cibles connues des ratés.
3. **Apprendre quand une preuve est fiable.** Distinguer nom de site et nom
   légal dans le score natif, tenir compte du numéro incertain et traiter le
   siège comme repli explicite. Chaque nouvelle hypothèse devra encore émettre
   directement 100 SIRET et publier ses régressions, avant toute promotion.
4. **Valider hors des groupes consommés avant de conclure.** Une recette figée
   doit passer un contrôle représentatif contre les vrais distracteurs, puis
   les gates OOF et SLO prévus au contrat. Aucun résultat ici n’autorise
   l’ouverture du fold1 ou le dégel des modèles aval.

## Fine-tuning : une ambiguïté mesurée, pas une promesse de gain

Le modèle actuel apprend deux tâches : site avec nom et adresse (poids 0,75),
et entreprise avec le nom seul (poids 0,25). L’audit porte uniquement sur les
15 035 lignes TRAIN du jeu figé ; aucun gradient ni test réservé n’a été ouvert.

| Informations disponibles dans la requête | Lignes dont l’entrée exacte désigne plusieurs SIREN |
|---|---:|
| Nom seul | 1 148, soit 7,64 % |
| Nom et code postal | 192 |
| Nom et commune, comparaison diagnostique | 172 |
| Nom, code postal et voie | 10 |
| Nom et adresse complète | 0 |

Ces nombres décrivent l’information disponible, pas la performance d’un
nouvel encodeur. Zéro ambiguïté exacte ne prouve ni la qualité des labels,
ni la robustesse aux variantes, ni le recall sur de nouvelles entreprises.
Les identifiants SIREN restent des clés de regroupement, jamais des features.

Dans le plan de batches réellement exposé avant le checkpoint best1000,
18 batches contiennent 37 positions d’ancres à nom identique mais entreprises
différentes, sans masque protecteur. Dix positions de la tâche entreprise
traitent aussi un texte documentaire identique au positif comme un négatif.
Aucune collision documentaire exacte de ce type n’a été trouvée pour la tâche
site. Ces collisions explicites sont rares : ce n’est pas une explication
quantifiée des 23 ratés, et leur suppression n’assure pas 99,2 %.

La prochaine ablation de fine-tuning doit donc distinguer deux problèmes :

- **Information insuffisante** : conserver un canal de similarité de noms,
  mais ne pas lui demander une identité unique quand l’entrée ne peut la
  déterminer. Un contexte géographique doit avoir un correspondant documentaire
  officiel explicite ; ajouter un code postal à la requête seulement ne suffit
  pas si les vues entreprise n’en contiennent aucun.
- **Négatifs indécidables** : masquer prudemment les collisions d’entrée ou de
  texte qui ne permettent pas une discrimination, sans déclarer positives ou
  identiques toutes les entreprises homonymes. Garder les vrais distracteurs
  locaux et les preuves séparant les sites d’une même entreprise.

Les vues parent décrivent plusieurs noms et sites ; leur supervision doit
rester distincte de celle du site exact. Toute évolution devra être comparée
au pipeline V99 entier, sur des groupes non consommés et le corpus complet,
pas seulement sur la loss ou le canal dense. Aucun nouveau training n’est
lancé dans cette fin de plage ; le checkpoint de référence reste inchangé.

Les stocks historiques SIRENE locaux contiennent les périodes de noms,
d’activité et de siège ; le stock Historique établissements inspecté ne
contient pas les colonnes d’adresse. Une histoire complète des adresses et
coordonnées nécessite donc les autres preuves datées ou des snapshots,
pas seulement ce stock historique.

Le RNA national accessible par le miroir officiel est daté d’octobre,
après le cutoff d’août. Les titres de 1,1 million de SIREN publics ont été
audités mais restent hors benchmark. `titre_court` est souvent une troncature,
pas un sigle. Le fichier historique d’août a renvoyé HTTP403, sans contournement.
[RNA officiel](https://www.data.gouv.fr/datasets/repertoire-national-des-associations),
[miroir national](https://www.data.gouv.fr/datasets/rna-agrege-a-lechelle-nationale).

## Artefacts et traçabilité

Comparaison finale :
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v102/final_paired_20261013_a/report.json`,
SHA `7d985021eb852661dbca09767abaacb2fecb6d6e373746f8b86496b0521a27c7`.
V99 : `v99/initial_alignment_native_20261013_a/report.json`,
SHA `437d7777a937475d44074ad91ee284bfa917043648762c1871e6d8384ed4f2ca`.

Milestones : V88 `44707cd5`, V94 `172a7e99`, V95 `b7d73976`, V98 `0a97e241`,
V99 `550732d8`, comparaison `52645159`, audit siège `d8817f46`, champs de noms
`365efe9e`, inventaire brut et protections de diffusion `b4eb35cc`, ambiguïté
TRAIN `45834634`, information et négatifs exacts `bdf51335`.
Réconciliation du diagnostic BODACC avec les 23 ratés restants : `665ad966`.
Audit de séparation des groupes pour les encodeurs V1/V6 : `e9d76700`.
Noms de sites sectoriels et gestionnaire : `810a6203` ; audit des quatre
ratés site/parent : `caa4367b`.
Les reprises de V99 ont conservé les shards et les bindings. Aucun unit test,
aucune dépense externe, aucun push ou modèle aval modifié.
