# Audit BODACC et dirigeants — 12 octobre 2026

## Décision

Prioriser **les acteurs personnes morales et les liens de groupe explicitement
documentés**, combinés à la géométrie propre du site. Ne pas lancer maintenant
un enrichissement massif par les noms de dirigeants personnes physiques, ni un
nouveau fine-tuning pour compenser l'absence de ces preuves.

Ce mandat produit un audit, pas une modification du retrieval. Aucun nouveau
recall n'est mesuré, aucun modèle ou serving promu. Le meilleur point reste
V32 : 495/526 exact, 94,1065 % brut / 93,6717 % pondéré sur un pilote déjà
consommé, et non une validation nationale de 99 % ou 99,2 %.

## Périmètre et méthode

- Même pilote complet 526 requêtes, folds humains 2/3/4, après génération V32
  scellée et exit owned0 ; 31 ratés et 495 réussites conservés.
- V49 inspecte les snapshots annuels BODACC 2008–2026 : 50 380 026 lignes de
  snapshots, pas 50 M entreprises ni une garantie d'actualisation exhaustive.
  Filtre au 1er août 2026. Recherche par registre, puis vérification des
  identifiants structurés du sujet. Pas recherche inverse exhaustive des
  prédécesseurs absents du registre.
- Les recoupements CRM sont LABEL-SIDE et ne créent ni alias ni features.
  Une correspondance de tokens sert à relire une preuve, pas à certifier un
  lien ou un gain produit. « Correspondance complète » signifie tous les
  tokens informatifs après retrait des mots génériques/formes juridiques.
- V50 inspecte sélectivement le ZIP RNE du 4 mars 2026 : recherche heuristique
  dans l'ordre des membres et six membres voisins, identification exacte du
  sujet. 19/26 SIREN distincts des ratés retrouvés, représentant 24/31 cas.
  L'absence des sept autres cas n'est PAS une absence démontrée du RNE.
- Identités propriétaires masquées non reconstituées ; aucune date de
  naissance, adresse de domicile ou collection de bénéficiaires effectifs
  extraite dans les artefacts. Les noms de sociétés dirigeantes ne deviennent
  pas des alias propres de leurs filiales/sites.
- Gemma, aval, serving, folds0/1 et test4323 restent inchangés/fermés.
  Aucun entraînement, unité de test, dépense externe ou nouveau long run.

## 1. Ce qu'on possède et ce qu'on perd

Les 19 snapshots et leurs manifests existent réellement sur CATNAT.
Les colonnes `listepersonnes` gardent notamment le champ `administration` :
les noms d'acteurs ne sont donc pas systématiquement perdus au téléchargement,
mais ne sont pas exploités par le canonicaliseur BODACC actuel.

Pour 2008–2021 et 2026, la copie utilisée a dix colonnes sélectionnées :
identifiants, publication, personnes, établissements, précédents exploitants
et propriétaires. Pour 2022–2025, les copies possèdent aussi `acte`,
`modificationsgenerales`, `divers`, etc. Ne pas prétendre disposer localement
du texte complet de tous les actes anciens ni des PDF de chaque annonce.

Le canonicaliseur lit les noms/adresses du sujet et les prédécesseurs
structurés. Il exclut explicitement les dirigeants et le texte libre. Le RNE
des rôles indexé contient uniquement des acteurs **ENTREPRISE**, pas des
dirigeants **INDIVIDU**. Le nom/prénom d'un EI annoncé reste une identité du
sujet économique, ce qui est une catégorie distincte.

Références :
[données ouvertes BODACC](https://www.bodacc.fr/pages/donnees-ouvertes-et-api/),
[jeu d'annonces commerciales](https://www.bodacc.fr/explore/dataset/annonces-commerciales/),
[accès RNE INPI](https://www.inpi.fr/ressources/formalites-dentreprises/acces-lapi-formalite-rne).
L'exemple officiel BODACC confirme que `administration` et les actes sont des
champs des annonces, pas des dénominations propres interchangeables.

## 2. Mesure de présence et d'indices, pas de recall

| Observation BODACC du parent | Tous526 | Réussites495 | Ratés31 |
|---|---:|---:|---:|
| Au moins une annonce / sujet structuré retrouvé | 379 | 358 | 21 |
| Champ administration présent | 347 | 327 | 20 |
| Nom propre absent du dictionnaire **parent** actuel | 17 | 16 | 1 |
| Au moins un token informatif CRM dans administration | 98 | 94 | 4 |
| Tous tokens informatifs CRM dans administration | 27 | 24 | 3 |
| Tous tokens informatifs CRM dans acte/modification/divers disponibles | 22 | 21 | 1 |
| Prédécesseur structuré dans une annonce du sujet | 108 | 103 | 5 |

Les noms absents du dictionnaire parent ne prouvent pas une absence de tout
l'index ni un alias admissible de chaque site. Pour le seul raté concerné,
CAMPHES, les noms relevés sont ALTO'PRIX et TOOBA : ils ne ressemblent pas au
nom CRM. TOOBA existe déjà pour d'autres propriétaires/sites dans le
dictionnaire. Le nom propre du SIRET cible est KANDY. Il faut rattacher
chaque enseigne à son établissement/adresse/période, pas la diffuser à tous
les sites du SIREN. Aucun gain ne découle de cette observation.

## 3. Les cas probants : acteurs sociétés, pas personnes physiques

| CRM raté V32 | Preuve recoupée | État site/géométrie | Recette V30 déjà exécutée |
|---|---|---|---|
| CATZAM | Président CATZAM ; RNE type ENTREPRISE, code73 | Actif ; point propre à environ1,93m de la requête | Retrouvé |
| FOURNIER | Président FOURNIER ; RNE type ENTREPRISE, code73 | Fermé ; pas de XY, voie et numéro publics disponibles | Toujours raté |
| EUREDEN GROUP | Président COOPERATIVE EUREDEN ; RNE ENTREPRISE, code73 | Fermé ; pas de XY, voie publique sans numéro | Retrouvé |
| EUROVIA MANAGEMENT | Administrateur EUROVIA STONE ; ENTREPRISE, code65 | Correspondance partielle EUROVIA, pas identité MANAGEMENT | Toujours raté |

Dans les 24 cas dont le sujet RNE est retrouvé, 17 ont des acteurs INDIVIDU.
**Aucun de leurs noms n'a de token informatif commun au CRM de ces ratés.**
Les quatre recoupements observés sont des acteurs ENTREPRISE. Cela rend les
personnes physiques moins prioritaires pour CES entrées nom/adresse, sans
prouver qu'elles seraient inutiles pour un autre CRM contenant un contact,
un exploitant ou un nom de personne. La couverture RNE n'est pas exhaustive.

Il faut éviter les homonymes et les gros bouquets : dans la table RNE entière,
CATZAM intervient pour10 SIREN, COOPERATIVE EUREDEN pour68, FOURNIER pour15.
La chaîne FOURNIER désigne quatre SIREN acteurs distincts. Nom identique ne
signifie donc ni société mère unique ni entreprise cible identique.

V30 (noms d'acteurs RNE typés, sans personnes physiques) est une recette déjà
mesurée, et NON un nouveau test causant isolément un gain. Comparée à V32,
elle récupère3 cas (BLANCHISSERIE, CATZAM, EUREDEN) et en perd5 :493/526 au
lieu de495, pondéré92,2313 % au lieu de93,6717 %. Il serait incorrect de
réactiver globalement V30 sous prétexte de ces deux exemples convaincants.

## 4. Une preuve réellement ignorée dans le texte BODACC

Annonce `A202300971476`, 21 mai 2023, `acte.descriptif` : projet d'absorption
de BERNARD AGRISERVICE (344799390) par BDV SERVICES (383509049). Le texte
explique que les droits sociaux des deux sociétés sont détenus par la même
société mère, COOPERATIVE EUREDEN (841645690).

Cette clause apporte **une preuve de détention de groupe explicite**, plus
précise que la seule présence d'EUREDEN comme président. Le canonicaliseur
ne lit pas ce descriptif et ne produit aucune relation pour CET identifiant
d'annonce. Attention : les relations canoniques 2023 contiennent déjà TROIS
ASSET_TRANSFER entre les deux sociétés via d'autres annonces. Ce n'est donc
pas une preuve que toute la fusion est absente du graphe. La lacune constatée
est le contenu typé de cette annonce et sa clause de société mère.

Un projet et sa date prévue ne prouvent pas une opération réalisée. Une
future extraction doit séparer publication, date d'effet annoncée,
confirmation de réalisation, acteur, rôle et portée établissement/unité
légale. Elle doit aussi préserver la distinction groupe, contrôle, mandat,
transfert de fonds et identité juridique.

## 5. Suite recommandée, non exécutée par cet audit

1. **Géométrie propre + acteurs personnes morales** : récupérer la géométrie
   officielle manquante via BAN locale, comme déjà proposé ; garder qualité
   et empreinte de voie quand pas de numéro. Exposer un canal natif contexte
   acteur→entreprise→site, avec discriminants propres du site, sans recopier
   l'adresse du président ni éliminer les établissements fermés.
2. **Compilation hors ligne des preuves textuelles BODACC** : extraction
   indépendante des labels de quelques relations typées (société mère
   explicitement nommée, absorption/TUP, cession/transfert), avec identifiants
   rattachés à leurs rôles par la phrase et source exacte conservée. Plusieurs
   SIREN dans le texte ne constituent jamais à eux seuls une arête. Commencer
   sur les années dont les actes sont réellement disponibles ; annoncer les
   manques de couverture, ne pas les combler avec des alias CRM.
3. **Ablation propre, pas retour à V30** : apprendre sur TRAIN les interactions
   contexte corporate × preuve locale/ancienne du site, et comparer une seule
   modification au modèle gelé. Tous canaux avant un unique collecteur natif
   ≤100, pas une fusion label-guidée de listes V30/V32 ni un pool fixe2000.
4. Après gel, contrôle représentatif indépendant et métriques exactes/SIREN/
   opérationnelles séparées, gains ET pertes. Aucun accès fold1/test4323 sans
   autorisation. Ce pilote ne peut certifier l'objectif.

La piste n'explique pas à elle seule les cas NAOS→AUBERT, prestataires→clients,
enseignes franchisées ou établissements publics. Elle ne remplace pas le
volet preuves commerciales indépendantes ni la supervision TRAIN plus large.

## Artefacts et intégrité

- V49 : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v49/postseal_bodacc_roles_20261012_a`.
  Processus outil session87230 réellement terminé0 ; audit19 snapshots en20,68s.
  casesSHA `1cf1e21aa70484be5f3dee3f73aef485e130028344a639e0349505f3e4017201`.
- V50 : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v50/postseal_rne_directors_20261012_a`.
  Processus outil session70336 réellement terminé0.
  casesSHA `cd3f82b5ef2b234f0e41292bea2745b697774216aff541e5fede3201d7b1df1c`.
- Implémentations reproductibles : `scripts/embeddinggemma2_v49_postseal_bodacc_roles.py`
  et `scripts/embeddinggemma2_v50_postseal_rne_directors.py`.
- Les manifests sont référencés et les tailles/schémas inspectés ; pas nouvelle
  certification indépendante de chaque SHA des gros ZIP/parquets source.
  Les snapshots historiques, modèles et recettes n'ont pas été modifiés.
