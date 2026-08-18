# Contrat d'exécution — Retrieval officiel RNE/BODACC v2

Statut : implémentation opt-in. Le retrieval historique et le test fold 1
restent inchangés tant que les gates dev ci-dessous ne sont pas franchis.

## Objectif

Conserver le SIRET exact dans au plus 100 candidats pour au moins 99,0 % du
périmètre identifiable, avec une couverture identifiable d'au moins 80,0 %.
La vue opérationnelle même SIREN/même site est publiée séparément et ne remplace
jamais le label exact.

## Sources autorisées

- snapshots SIRENE courants, historiques et liens de succession déjà scellés ;
- JSON publics du RNE acquis en priorité par l'API Formalités v4 HTTPS
  (`/api/sso/login`, Bearer, `/api/companies/diff`, curseur
  `pagination-search-after`), ou par SFTP/FTPS authentifié ;
- annonces BODACC A/B acquises depuis les interfaces officielles DILA HTTPS,
  FTPS ou Opendatasoft v2.1.

Le FTP et HTTP en clair sont interdits. Les secrets sont lus depuis le Trousseau
macOS et ne figurent dans aucun argument, environnement, log ou manifeste.
PDF, comptes annuels, bénéficiaires effectifs, dirigeants et texte intégral des
annonces sont hors périmètre.

Les enregistrements portant `diffusionCommerciale=false` ou
`diffusionINSEE=N` sont placés en quarantaine et ne sont jamais indexés.

## Preuves et relations

SIRENE reste l'autorité pour l'existence du SIRET, son état courant et son
adresse administrative. RNE et BODACC ajoutent uniquement des valeurs ou liens
officiels datés. Un lien inter-SIREN exige deux identifiants structurés ; aucune
jointure floue nom/adresse n'est admise lors de la construction.

Les relations ne corrigent jamais les labels CRM. Elles créent un canal de
candidats à un saut, borné géographiquement, avec provenance explicite.

## Retrieval et admission

Le moteur combine exact, BM25F mots, q-grams caractères, route directe SIRET,
route `(INSEE,SIREN)` puis expansion de sites, et overlay officiel RNE/BODACC.
L'union interne est plafonnée à 2 000. L'admission LambdaMART utilise seulement
des scores, rangs, accords de champs et provenances ; SIRET, SIREN et identifiants
de sources sont interdits comme features. Les candidats exacts et consensus
multicanaux disposent de slots protégés. La sortie est déterministe et limitée
à 100.

Même SIREN/même site est exclu des négatifs, mais seul le SIRET exact est positif
pour la métrique principale. Les 20 000 lignes synthétiques ne sont pas utilisées
pour ajuster ce retrieval.

## Exécution bornée

1. Construire et sceller les snapshots officiels et l'overlay sans lire de
   labels CRM.
2. Construire l'union sur folds 2/3/4 et entraîner une configuration unique.
3. Évaluer une seule fois sur fold 0 : couverture >= 80,0 %, oracle interne
   >= 99,3 %, Recall@100 exact >= 99,0 %, maximum 100, p95 <= 1 s et p99 <= 2 s.
4. Si tous les gates passent, geler les artefacts et ouvrir fold 1 une seule
   fois. Sinon conclure `PIVOT` sans recherche de grille automatique.

Chaque rapport publie ensemble historique, V2, V3, exact/opérationnel,
actif/fermé, SIREN inédit/site nouveau, nombres bruts et hashes des sources.
Une vérité absente de l'union est une erreur end-to-end. Aucune injection
positive n'est permise.
