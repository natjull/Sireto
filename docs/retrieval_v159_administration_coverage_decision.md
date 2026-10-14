# Dirigeants corporate BODACC : couverture supplémentaire, pas gain de recall

Diagnostic séparé `8dbb8bc5`, exécuté seulement après les sources V158 B **et**
V159 complètes/scellées et leurs exits réels 0. Session `28144`, chunk final
`b06067`, exit réel 0 ; reçu sous
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v159/coverage_diagnostic_20261014_night/owned_tool_exit.json`.

Les 526 labels et résultats de retrieval restent inchangés. Le périmètre de
source comporte 512 cibles corporate publiques ; 12 EI et 2 cibles protégées
restent dans le dénominateur pipeline de 526, mais ne sont pas profilées par
cette source. Classification existante réutilisée, aucune nouvelle règle de
label ni correction manuelle. Aucun jeu fermé, fit ou recherche natif.

## Mesure décisive

| Couverture du vrai propriétaire humain | V158 B | V159 |
|---|---:|---:|
| Propriétaire avec une relation dans la source | 10/512 | 48/512 |
| Contexte avec au moins un token CRM informatif | 1 | 3 |
| Contexte positif via paire sujet/acteur absente RNE | 1 | 2 |
| Requête informative entièrement dans un même nom d'acteur | 0 | 1 |
| Nouveau contexte positif sans overlap RNE déjà admis | 0 | 0 |
| Contexte positif sur les 22 ratés V115 | 0 | 0 |
| Contexte positif sur les 20 ratés V140 | 0 | 0 |
| Contexte positif sur les 19 ratés V142 actuels | 0 | 0 |

V159 porte des relations pour deux vrais propriétaires parmi les anciens
22/20 ratés, mais aucun mot informatif CRM n'y trouve d'appui. Parmi les 19
ratés actuels, un propriétaire a une relation et aucun n'a de contexte
positif. Une relation officielle disponible n'est donc pas forcément une
preuve pertinente pour cette requête.

Le contexte corporate RNE effectivement admis contient déjà un overlap de
requête pour 49 cibles publiques ALL526. Il est lu en streaming depuis son
artefact scellé, sans lire le JSON activité de 3,5 GiB, sans exporter de noms
personnels ou de profils protégés.

## Ce que sont les trois recouvrements V159

- `KPMG AUDIT NORMANDIE`, rôle publié commissaire aux comptes titulaire,
  publications 2010-06-13 et 2012-09-23 : seul token partagé `NORMANDIE`,
  pas une identité ou marque de l'entreprise. Degré global 854 ; confiance
  0,006837. La paire historique est absente RNE, mais un overlap de contexte
  corporate existe déjà chez ce propriétaire. Ce cas était déjà un hit.
- `TSEIN FINANCES`, rôle publié président, publications 2013-07-10 et
  2015-04-29 : token `TSEIN`, requête informative complète mais pas égalité
  de nom entier ; degré 1, confiance 0,5. Paire historique absente RNE,
  mais le mot est déjà présent dans le contexte corporate admis. Déjà hit.
- `AMS AUTO MANAGEMENT SERVICE`, rôle publié président, publications
  2016-05-20 et 2017-04-20 : seul token `AMS`, recouvrement partiel ; degré
  11, confiance 0,068876. Paire déjà présente dans RNE. Déjà hit.

Tous ces acteurs sont résolus exactement vers une unique unité corporate
actuellement publique ; leurs noms ne deviennent pas des alias du site.
Les dates sont de publication, pas des garanties de fonction actuelle.
Ces constats sur un panel consommé ne valident pas une généralisation.

Le rapport publie les coefficients corporate gelés et le soutien
administratif déjà calculé. Les deux contextes de paire absente ont ce
soutien ; le bonus théorique non ajouté est environ 0,012436 et 0,902479.
Il ne s'agit ni d'un score total, ni d'un rang, ni d'une probabilité. Aucun
bonus d'information de cette source n'est possible sur les 19 ratés actuels.

## Direction

L'omission de champ était réelle et les relations historiques récupérées
sont nombreuses. **Mais cette source ne démontre aucun apport orthogonal
pour les cas manquants de ce panel.** Pas de justification scientifique pour
consacrer cette nuit un nouveau run national à V158/V159 dans l'espoir de
corriger ces ratés. Un éventuel effet de quantification ou déplacement de
concurrents ne serait pas une nouvelle preuve d'identité et ne justifierait
pas une recette réglée sur ces cas.

Conserver ces sources comme données corporate historiques typées,
non exhaustives et non déployées ; décision de run final à la racine.
Ne pas utiliser les noms naturels pour forcer un résultat, ne pas assimiler
dirigeant, groupe, propriétaire, opérateur et enseigne. Aucune performance
99 %, nationale ou indépendante, n'est revendiquée.
