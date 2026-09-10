# V30 — recherche conjointe souple nom/adresse puis Master v23

10 septembre2026, accord utilisateur « Go ». Recette unique préenregistrée
avant mesure. V23 reste référence, v29 rejetée n'est pas utilisée comme base.

## Hypothèse et recherche exacte

Un site absent des top512 nom et adresse séparés peut être retrouvé par
leur concordance partielle dans une seule recherche, avant la coupe512.
Cette hypothèse n'est pas une promesse de gain et ne crée aucune preuve absente.

Sur l'index Tantivy v18 scellé, document_type=SIRET, même filtre géographique
INSEE disponible sinonCP. Deux groupes de termes sur le **même document** :

- n-grammes3/4/5 de `query.name` dans `current_name_ngrams`;
- n-grammes3/4/5 de `query.address` dans `address_ngrams`.

Utiliser les payloads hexadécimaux existants, pas les marqueurs ultra-communs
g3/g4/g5. Disjonction interne de termes, un fragment au moins par groupe;
les deux groupes sont requis. Aucune égalité de nom/voie/numéro, aucun terme
`number` obligatoire. Les fragments du numéro éventuellement présent dans
l'adresse sont des termes souples comme les autres. Champs nom/adresse
manquants ou sans gramme : canal vide, requête conservée au dénominateur.

Score BM25 groupe nom divisé par son nombre de grammes distincts, plus score
groupe adresse divisé par son nombre de grammes distincts. Poids fixes égaux,
aucune calibration, variante ou grille. Cette normalisation borne l'effet
de longueur de requête mais ne rend pas les scores probabilistes/calibrés.
Top512 documents par score; départage terminal SIRET, rang de preuve dense
par score. Le SIREN est lu dans le champ officiel stocké, jamais reconstruit
depuis le SIRET. Validation des identités et de l'unicité.

Noms courants typés légaux/commerciaux du parent et enseignes du site,
adresse courante du même document. Pas de concaténation historique supposée
compatible, ni de nom de sibling transféré. Aucune nouvelle source, index,
BAN, géographie ou feature de personne. Les anciens canaux temporels restent.

## Intégration et mesure end-to-end

Ajouter un seul canal `siret_joint_name_address_char` aux signaux v18,
source_siret attaché au témoin et channel_rank=rang dense. Tous les autres
signaux inchangés, y compris l'ancienne géographie v23. Le RRF de ce canal
suit la règle existante `1/(60+rang)`; il ne représente pas une nouvelle
autorité indépendante des autres canaux SIRENE.

Reconstruire l'union naturelle et l'expansion locale INSEE/CP v21; conserver
toutes les références précédentes avant coupe. Réextraire les16 features
v23, recharger les modèles outer v23 sans fit, puis appliquer sa règle
globale2000 et ses ex aequo. Aucun quota ancien/nouveau, aucun score géo v28,
aucune fusion v24, aucun réglage sur les cas audités. Les changements de
distribution de features des modèles gelés restent une limite explicite.

Le grand réservoir est exploratoire, pas un Master produit; seul le
top2000 est la sortie mesurée, aucun LambdaMART SIRET sur le grand réservoir.
Pas d'hydratation500/100 ni de dégel du ranker/decider/risk/accepteur.

Étapes séparées : recherche sans labels, Master sans labels, évaluation
label-side. Toutes les1034 requêtes folds2/3/4; exact humain et opérationnel
scellé séparés. Absence catalogue conservée comme erreur. Folds0/1 fermés.
L'OOF ne rend pas vierge ce panel réutilisé522 difficultés+512 contrôles.

Mesures : hits du canal, nouveautés exactes/parents par rapport au réservoir
v23, expansion naturelle et couverture du nouveau réservoir, Master exact
et opérationnel, gains/pertes vs v23, par fold/cohorte/origine/état et
historique/V2/V3. Attribution officielle réactualisée : catalogue absent,
parent absent, site non développé, éviction au Master. Les gains/pertes
Master sont distingués selon présence dans l'ancien réservoir.

Statut descriptif `NET_GAIN_REUSED_PANEL` si net exact>0, sinon
`NO_NET_GAIN_REUSED_PANEL`. Même un net positif ne promeut pas le produit.
Verdict scientifique `PIVOT_GENERATION`, ou `STOP` pour manque de preuve,
incohérence ou ressources. Pas de recette ajustée après résultat, projection
nationale, garantie99,2%, Wilson/SLO produit ou revendication cross-CRM.

## Contrôles et ressources

Tests avec index synthétique uniquement pour le code (jamais en features
ou évaluation métier), préflight≤16 sans labels, commits code/config avant
complet. Rejouer les scores/Master v23 gelés, vérifier l'additivité du
réservoir, les caps et la parité préflight/complet. Vérificateur distinct
avec replay du canal sur le même panel, modèles et métriques indépendants.

Mac/CATNAT, quatre threads de recherche; DuckDB8Go/deux threads pour le
Master; RSS14Gio, zéro swapout,30min par étape. Index : SHA manifeste/meta
et inventaire tailles/mtime avant/après, pas de claim de rehash intégral.
Sources/code/config/runtime/résultats scellés. Hashes préalables séparés des
temps d'étape, aucun coût externe ni donnée CRM transmise à Claude.
