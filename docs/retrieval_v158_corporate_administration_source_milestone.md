# V158 — mentions historiques de sociétés dirigeantes, source seulement

Source B complète, observée le 14 octobre 2026 avant toute lecture de labels.
Recette/parser pré-déclarés `c3ce7c4b`, correction d'exécution du seul alias SQL
réservé `admin` → `admin_text` dans module/namespace neufs `21a186b0`.
Le run A échoué et tous ses artefacts sont conservés ; aucune année complète
n'y avait été construite.

Root : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v158/corporate_administration_20261014_night_b`.
Session native-tool de la source `28938`, chunk final `6cf87f`, exit réel 0 ;
enfant `54674`, owned exit 0. Durée 659,307 s ; pic RSS observé
1 593 131 008 octets. Aucun retrieval natif, fit, index ou serving modifié.

SHA generation :
`8265910a9970c12c78f515b059fbe31e9ce4b0130e2d3f7f28eb0ac6e0d818de`.

## Mesures de source, pas de recall

- Les 19 années canoniques sont lues entièrement, cutoff de publication
  2026-08-01, sujet PM strict directement immatriculé, actuellement corporate
  public `O` et hors EI ; tableaux d'acteurs exclus, pas de produit croisé.
- Une clause doit donner un rôle administratif explicitement suivi de `:`,
  une forme corporate légale explicite et un nom exactement résolu vers une
  seule unité corporate publique. Les représentants naturels, suffixes
  d'adresse/contact, listes et parenthèses sont rejetés.
- 25 076 lignes de relations distinctes `(sujet, acteur, nom publié, rôle)` ;
  23 362 sujets et 10 492 acteurs. Ce ne sont pas 25 076 arêtes de propriété.
- 14 118 de ces lignes correspondent déjà à une paire sujet/acteur RNE ;
  10 958 n'y correspondent pas avant cutoff. La dédup par paire, tous rôles
  RNE confondus, est conservatrice : aucun code RNE n'est assimilé arbitrairement
  à un libellé BODACC. Ces comptes de lignes ne sont pas des comptes de paires
  uniques, ni des preuves que le contexte lexical correspondant était absent
  des autres canaux.
- 8 911 noms corporate ont plusieurs résolutions exactes et sont exclus.
- ALL526 query-only : 290 requêtes ont du contexte brut, 248 du contexte de
  relations non présentes dans la table RNE. Aucun top-k de propriétaires.
- Confiance contextuelle = `.5 / sqrt(degré global de l'acteur)` × couverture
  informative IDF du nom ; maximum sur les noms/acteurs, jamais addition des
  annonces répétées. Ce nombre n'est pas une probabilité d'identité.

## Limites et prochaine source scientifique figée

Le parseur B est délibérément **non exhaustif** : sa règle interdit même une
qualification finale entièrement légale `(SAS)` ou `(SARL)`. Le compteur
`ambiguous_list_or_parenthesis` ne sépare pas ces cas des autres ambiguïtés.
Avant tout label, la nouvelle grammaire V159 a donc été pré-déclarée au commit
`3104884a` : seule parenthèse finale dont le contenu entier matche les formes
légales déjà figées, aucun autre signe de liste/parenthèse, même résolution
unique publique et mêmes scores, avec `qualification_origin` conservée.
Ce n'est pas une correction d'exécution : V159 est une source scientifique
distincte. V158 B reste immuable. Aucun label n'est lu avant les deux sources
complètes et leurs exits réels 0.

Les mentions BODACC sont textuelles, datées par publication, pas par effet
garanti. Elles ne certifient pas une fonction actuelle, la propriété, un
opérateur, une enseigne ou une identité de site. La résolution utilise les
noms corporate SIRENE actuels et le dictionnaire officiel admis : ce dernier
ne conserve plus les dates de chaque alias et n'est pas un référentiel as-of
certifié. Aucun texte administratif brut ni nom naturel n'est exporté comme
preuve ou feature. Aucune généralisation de performance n'est démontrée.
