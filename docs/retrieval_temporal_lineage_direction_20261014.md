# Direction généralisable : des observations historiques, pas des sacs d'alias

Audit source/schema du 14 octobre 2026, sans CRM, labels, scan de valeurs,
anti-join national, nouveau modèle ou recherche. Script `03e41224`, résultat
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v159/temporal_lineage_20261014_night/report.json`,
chunk `49d783`, exit réel 0, 0,115 s. Il revalide le gap **déjà connu** de
V117 (`6e80f66a`), pas une nouvelle découverte promettant un gain.

## Ce qui est réellement complet et ce qui ne l'est pas

| Couche réellement utilisée | Historique présent | Limite concrète |
|---|---|---|
| Canonical BODACC/RNE et dossier national | Anciennes mentions BODACC, objets RNE courants/fermés | Pas les familles StockHistorique INSEE dans ces inputs |
| Index natif `4b332899d8698354` | Mots des noms historiques/supporting sélectionnés du dossier | Caps, pas les n-grammes/exacts historiques ; dates/scope par nom perdus |
| Sources INSEE historiques séparées | Périodes UL/site et liens de succession | Présentes sur disque, pas une preuve que le principal les indexe |
| V129/V130, dictionnaire complet | Variantes des historiques INSEE et observations officielles | Source de features, pas remplacement des postings lexicaux nationaux |
| Lookup runtime `owners.sqlite` | Tous les noms admis par propriétaire | Aplatit source, période et rôle ; ne recrée pas l'index global historique |
| V23, successions natives | Liens économiques INSEE datés, extrémités explicites | Contexte séparé ; ne doit pas promouvoir un nom de successeur en nom propre |

Les SHA du builder natif, de sa politique et des manifestes index→projection
→dossier sont vérifiés exactement. Builder :
`43f4eecb5acee06d67af720a9f70abd1cd28017b88531eedadf2fe8761a30e28` ;
politique : `8ff8576e220908604fba8d946b748ee5d14befe484229276439f20679b0f8677`.
Le dossier déclare les inputs canoniques `951c75c6…` (merge) et `4192748e…`
(bulk RNE compatible en taille/footer/manifeste). Le premier input est lié
par SHA déclaré au merge RNE + BODACC 2008–2026. Les footers de ce fichier
donnent 833 groupes purement BODACC et un groupe allant de RNE à
SIRENE_CURRENT ; le bulk RNE a 775 groupes purement RNE. Ni source range ne
peut contenir SIRENE_HISTORY/SIRENE_SUCCESSION. Aucun hash intégral de 8,7 Go
n'est recalculé ici ; le deuxième fichier n'est donc pas présenté comme un
replay cryptographique de l'input, seulement une identité candidate cohérente.

Le manifeste national annonce `temporal_complete=true`, mais ses comptes
source portent seulement sur SIRENE_CURRENT/RNE/BODACC. Ce flag historique
est contradictoire ; il ne valide ni les cinq familles requises aujourd'hui,
ni toutes les périodes ou tous les champs. Il n'est pas corrigé rétroactivement.

Les stocks INSEE distincts ont 71 355 318 lignes UL et 95 865 102 lignes site.
Leur source canonique séparée déclare 165 291 719 observations acceptées et
9 727 541 relations. Cela ne signifie pas autant de noms uniques nouveaux.
Les historiques site n'ont **ni ancienne voie/numéro/CP ni XY**. Les historiques
UL ne fournissent pas le sigle courant. Les dates de période disponibles ne
permettent donc pas d'inventer l'adresse historique correspondante.

V129 conserve 622 697 477 observations complètes, dont V130 admet
338 540 853 lexicales, 105 564 761 facts et 57 530 682 noms uniques normalisés
sur l'ensemble des cinq rôles. Ses manifestes disent explicitement
`complete_retrieval_index=false`. Ses observations gardent source, période,
champ et scope ; le lookup SQLite rapide n'en garde que owner/name/kind.
Un nom peut donc être accessible au calcul d'une feature **quand le candidat
fait partie de l'univers numérique query-only**, sans être disponible pour
une recherche lexicale nationale indépendante hors de cet univers.

## Les pertes sémantiques réelles

1. Le canonical builder transforme les noms INSEE historiques LEGAL/USUAL/
   SIGN en un unique kind HISTORICAL. La date est conservée, mais pas le rôle
   original dans ce champ. V118 rétablit des rôles depuis les colonnes brutes
   dans une projection distincte ; ce n'est pas une garantie du principal.
2. BODACC `dateparution` devient `valid_from`. C'est une preuve de publication,
   pas forcément le début de validité du nom. Le schéma n'a pas de `date_kind`
   distinguant période INSEE, date d'effet et publication.
3. Le portefeuille groupe les observations par sujet/rôle/nom : `max` des
   dates et `min` d'IDs de preuve. Ce résumé n'est pas la liste des épisodes
   ni une observation réellement unique. La couche temporelle séparée peut
   aussi utiliser min/max : elle n'est pas une conservation lossless d'intervalles.
4. Les documents natifs conservent des champs LEGAL/TRADE/SITE/HISTORICAL/
   SUPPORTING et des listes de sources/IDs, mais pas le mapping précis
   `(nom ↔ sujet propre ↔ source ↔ intervalle ↔ adresse observée)`.
   Les historiques parent et site sont concaténés dans le même champ.
5. Les caps sont 6 noms historiques/site, 12/parent, 4 supporting/site et
   6/parent. Les mots historiques passent aujourd'hui dans la recherche
   V23→V115 avant le même collecteur 100 : ils ne sont pas tous absents.
   En revanche, n-grammes et exacts sont uniquement courants.
6. Un lien de succession conserve ses endpoints et dates dans le dossier,
   mais `linked_sirets`/`linked_sirens` du document natif aplatit direction,
   type et temps. V23 relit les liens économiques datés originaux séparément.
   Il ne faut donc ni déclarer « aucune succession utilisée », ni supposer
   que tout lien générique du document certifie une continuité physique.

La table canonique `OfficialRelation` ne garde pas les flags INSEE
`continuiteEconomique`/`transfertSiege` comme attributs dédiés. Le canal V23
les vérifie sur la source brute, avec date de lien avant cutoff ; une future
annexe doit garder ces attributs plutôt que se fier au seul type SUCCESSION.

## TODO concret : une annexe d'observations officielles typées et datées

Une seule direction cohérente : rendre les mêmes variantes accessibles
au retrieval national, avec leurs **observations**, sans leur inventer une
identité de site. Réutiliser les tables V129/V130 plutôt que reminer des alias
CRM ou reconstruire tous les dossiers courants.

- Pour les mêmes SIRET : enseignes/noms usuels historiques restent attachés
  au SIRET qui les a publiés. Pour un SIREN : raison sociale/nom commercial
  restent preuve parent, pas enseigne de chaque site. Les SIGN au seul SIREN
  restent contexte de site non vérifié, pas alias propre propagé.
- Garder `original_name_kind`, champ exact, source/evidence, public-scope,
  période ou publication avec `date_kind` explicite. Même chaîne de nom
  peut pointer vers plusieurs épisodes ; ne pas remplacer ces épisodes par
  un intervalle min/max artificiel. Dédupliquer l'encodage de chaîne seulement.
- Adresses : seulement celles réellement publiées pour cette observation.
  Une adresse/XY actuelle utilisée comme scaffold reste marquée actuelle ou
  estimée ; aucun transfert au prédécesseur, parent, voisin ou co-site.
- Graphe : endpoints, direction, type, date et continuité économique INSEE
  distincts. La succession est une preuve contextuelle, pas une autorisation
  de copier nom/adresse ni d'assimiler les opérateurs. Publication BODACC et
  effet/liaison INSEE ne sont pas interchangeable.
- Postings mots/chaînes ou n-grammes historiques vers leurs propriétaires
  et observations, avec scores MAX des doublons de provenance. Le nom est
  le signal ; SIREN/SIRET ne sont que les références techniques de projection.
- Les postings des canaux sont combinés **avant** l'unique collecteur national
  initial100. Pas deux top100 puis RRF, pas pool2000/500 ou reranker masqué,
  pas quotas limitant les sites d'un propriétaire.

## Décision de calcul et contrôles préalables

L'anti-join source-only global à programmer séparément mesurerait les paires
historiques publiques `(sujet propre, scope, nom normalisé)` réellement
absentes des noms sélectionnés du principal. Distinguer absence native,
présence BODACC/RNE équivalente, présence seulement dictionnaire-features,
variante exclue volontairement pour scope/protection, et date inconnue.
Joins public/non-EI sur les colonnes business autorisées, sans nom de personne,
CRM ou GT. Pas de comparaison de chaînes seule qui confondrait propriétaires.
Ce compte global n'a **pas** été mesuré ici : le gap de famille et de schéma
n'en donne pas le nombre ni un gain garanti de recall.

Coût réel connu : Mac 24 Gio ; principal natif 98 Gio sur disque ; dictionnaire
V130 environ 3,2 Gio physiques (8,0 Go d'outputs complets déclarés, provenance
incluse) ; deux historiques bruts environ 1,73 Go. Un second clone du principal
n'est pas une bonne hypothèse de ressource. Le nombre de nouveaux postings,
leur taille et le temps d'indexation restent à mesurer sur données non labellées
avant admission ; aucun ETA ou plafond arbitraire n'est inventé. Préférer
annexe compacte, jointures disk-backed et corpus/encodeur courant réutilisés.
Un prototype partiel ne vaut jamais corpus national validé.

Avant tout score humain : source/masques/périodes/normalisation et score
pré-déclarés, couverture ALL526 intacte, calendrier/source eligibility
documentés, recette/code scellés, vraie génération initial100 complète et
owned exit0 ; seulement ensuite gains **et pertes** appariés sur les mêmes
segments. Validation indépendante/encoder OOF et jeux fermés restent des
gates distincts. Aucun résultat actuel ne démontre 99,2 %.
