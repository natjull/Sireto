# V21 — correction sans réglage CRM : +3 hits, sélection encore insuffisante

9 septembre 2026. Verdict : **`PIVOT_GENERATION`**.

## Résultat principal

La recette figée retrouve les 61 sites que v20 ne développait pas, mais seuls
trois survivent au plafond de 2000. Le défaut de présence est corrigé; le
défaut de sélection reste ouvert. Aucun poids n'a été ajusté après ce résultat.

| Mesure, mêmes 1034 dossiers de développement | V20 | V21 |
|---|---:|---:|
| Exacts difficiles présents avant sélection | 325 / 522 | 386 / 522 |
| Exacts difficiles conservés au Master2000 | 213 / 522 | 216 / 522 |
| Contrôles exacts conservés | 512 / 512 | 512 / 512 |
| Total exact | 725 / 1034 | 728 / 1034 |
| Total opérationnel | 727 / 1034 | 730 / 1034 |
| Nouveaux hits / anciens hits perdus, relativement à v20 | — | 3 / 0 |

Le taux exact de ce panel enrichi en erreurs est 70,4062 %, pas le taux
national. La projection conditionnelle du Master est 98,7018 %; la borne
du réservoir est 99,4230 %. Ces deux projections supposent la conservation
du préfixe historique sur les 23 571 requêtes nationales et ne constituent
pas un run national. La seconde est un oracle avant sélection, pas un
Master2000 effectivement obtenu.

386 rescues ne laisseraient que cinq erreurs de sélection par rapport aux
381 nécessaires conditionnellement à 99,4 %. Aujourd'hui 170 sont évincés :
la marge n'est pas acquise, et les gates PreRank500/Recall100 ne sont pas
mesurés. Aucun positif absent n'a été ajouté.

## Choix applicables sans connaître les bonnes réponses

- Conserver l'appartenance de **tous** les parents trouvés, sans transformer
  le rang v17 trié par identifiant en pertinence.
- Une même preuve répétée dans un canal ne compte qu'une fois.
- Des preuves de même qualité ont le même rang de score; les identifiants
  ne départagent que les scores finaux parfaitement égaux, au besoin.
- Conserver la formule, les plafonds et les critères de v20. Aucun nouveau
  poids, seuil, nom, commune, secteur ou dossier privilégié; aucun modèle fit.

La modification a été préenregistrée dans `04c2d070`, avec le contrat
`docs/retrieval_v21_membership_contract.md`. Ce n'est pas une preuve que les
choix hérités de v20 seraient universellement optimaux : ils n'ont simplement
pas été réoptimisés sur les erreurs connues dans ce sprint.

## Ce qui reste perdu

| Attribution officielle exhaustive | V20 | V21 |
|---|---:|---:|
| Parent absent | 132 | 132 |
| Parent présent, site non développé | 64 | 3 |
| Exact évincé au passage réservoir → 2000 | 112 | 170 |
| Exact absent du catalogue | 1 | 1 |

Le suivi apparié confirme exactement : 61 anciens sites non développés
entrent dans le réservoir; trois deviennent des hits et 58 sont évincés.
Les 112 anciennes évictions restent des évictions. Les 725 anciens hits
restent tous des hits. Aucune sélection de sous-population ne masque une perte.

| Segment | Requêtes | Exact v21 | Opérationnel v21 | Gains / pertes vs v20 |
|---|---:|---:|---:|---:|
| Fold 2 | 353 | 247 | 249 | 0 / 0 |
| Fold 3 | 346 | 241 | 241 | 2 / 0 |
| Fold 4 | 335 | 240 | 240 | 1 / 0 |
| CRM récent | 672 | 447 | 449 | 3 / 0 |
| CRM historique | 362 | 281 | 281 | 0 / 0 |
| MATCH_EXACT | 874 | 646 | 648 | 3 / 0 |
| AMBIGUOUS | 7 | 7 | 7 | 0 / 0 |
| UNRESOLVED | 153 | 75 | 75 | 0 / 0 |
| V2 non renseignée | 1034 | 728 | 730 | 3 / 0 |
| V3 non renseignée | 1034 | 728 | 730 | 3 / 0 |

Les vues historique all-human et états sont également scellées dans le
manifeste. La vue opérationnelle reste séparée : deux hits supplémentaires,
25 requêtes avec successeur actif de vérité officiellement fermée dans le
Master, 20 avec plusieurs actifs acceptables conservés. Ce sont des métriques
de présence, pas de précision top1. L'ensemble acceptable hérite de la preuve
officielle v20 et de ses limites (compléments d'adresse non mesurés).

## Ce que les contrôles disent de la généralisation

**28 tests passent**, dont 12 tests v21 : renommage des identifiants, ordre
des lignes, duplication d'une preuve, colonnes de labels/qualification/alias
inutilisées, isolation entre requêtes, rang v17 arbitrairement modifié,
adresses manquantes, témoins hors géographie, cap et préfixe, fold interdit,
dénominateur sans exclusion des vérités absentes, spill borné, reproductibilité
multicanal entre un et quatre threads. Les fixtures artificielles sont des
tests de logiciel, jamais des features ni un corpus d'apprentissage.

Sur les vraies données sans labels, les scores et sélections des 16 requêtes
du préflight final sont également **strictement identiques** à ceux obtenus
pour ces mêmes requêtes au milieu des 1034. Ces invariances portent sur la
nouvelle agrégation à sources extraites fixées; elles ne certifient pas les
index et choix historiques en amont.

Cela écarte certains mécanismes de mémorisation et défauts de calcul, mais
**ne prouve pas la généralisation à un autre CRM**. Les 1034 dossiers ont été
réutilisés dans de nombreuses versions; même sans apprentissage de poids,
choisir une stratégie après diagnostic reste une adaptation aux données de
développement. Redécouper ces dossiers ne les rendrait pas vierges. De plus,
les trois gains sont tous dans l'origine récente, et non uniformément répartis.

La preuve externe reste manquante et explicitement marquée comme telle dans
le manifeste. Avant toute affirmation cross-CRM : recette entièrement gelée,
export nouveau et indépendant, échantillon non sélectionné par le succès du
retrieval, audit des doublons entreprises/sites, segments nouveaux/déjà vus,
une seule évaluation sans retuning. Fold1 reste fermé; il ne remplace pas à
lui seul un test sur un autre portefeuille.

## Incidents techniques et ressources

La première tentative complète s'est arrêtée avant tout artefact de résultat
et avant toute évaluation : DuckDB essayait d'écrire son spill sur le disque
interne presque plein (environ 171 Mio libres observés). Le répertoire
`retrieval_v21_membership/177f6e1c35b10f85` est conservé incomplet, non admis.

Le commit `ae197652` place le spill sur le volume de sortie, borné à 12 Gio.
Le disque CATNAT avait environ 804 Gio disponibles. Les temporaires des tests
sont aussi placés sur CATNAT. Le préflight temporaire créé dans ce sprint a
été déplacé vers `retrieval_v21_membership_preflight/177f6e1c35b10f85-initial-archived`;
son manifeste historique garde ses anciens chemins. Aucun fichier utilisateur
n'a été supprimé ni aucun nettoyage général du Mac effectué.

Le contrôle des deux premiers préflights, sans labels, a également relevé
des écarts flottants <=2,78e-17, sans changement de sélection. `e677a6e6`
fixe l'ordre d'addition des canaux et l'ordre d'export. Cette correction
numérique ne change aucun critère ni poids. Le préflight final puis l'unique
génération complète réussie utilisent ce commit, avant l'unique évaluation.
Configuration scientifique inchangée pendant tous ces contrôles techniques.

| Phase finale | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Préflight sans labels, 16 requêtes | 1,547 s | 0,40 Gio | 0 |
| Génération complète | 16,379 s | 5,39 Gio | 0 |
| Évaluation | 0,537 s | 1,32 Gio | 0 |

Ces temps excluent la validation préalable des hashes, et ne mesurent pas
les SLO de latence produit. Génération : 3 900 144 références exploratoires
sur tout le panel, 1 973 172 au Master; minimum 205, médiane et maximum 2000
par requête. Aucun LambdaMART ne score le réservoir exploratoire.

Une vérification indépendante depuis le ledger humain confirme les 728 hits,
les caps, le préfixe, l'inclusion exact→opérationnel, les transitions appariées,
la reproductibilité préflight→complet et tous les hashes des artefacts finaux.

## Scellement et suite

Configuration SHA-256 :
`177f6e1c35b10f8572e55179890732571962e8f38f3fa088df1f4e3c7293c41b`.
Racine : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

| Manifeste relatif à la racine | SHA-256 |
|---|---|
| `retrieval_v21_membership/177f6e1c35b10f85-canonical-external/manifest.json` | `6115d0cec5ec054d6e4af991f11f280383329cee5da4f1f32017a18120a52bc7` |
| `retrieval_v21_membership_eval/177f6e1c35b10f85/manifest.json` | `6023dccb031a3007bfd322107d6c5deeb89e23ece0516880cfac78b358461292` |
| `retrieval_v21_membership_preflight/177f6e1c35b10f85-canonical/manifest.json` | `e1f00c09eb1d0713f815432a2b6e22c35e950837fae9ddda7916560b15630558` |

La prochaine question porte sur la sélection des 2000, pas sur une nouvelle
liste de dossiers à sauver. Toute nouvelle allocation devra être justifiée
par des preuves génériques disponibles en production, préenregistrée avant
mesure, sans réglage sur les folds externes. Un modèle SIRET ne peut pas
réparer une éviction antérieure à son entrée et doit rester limité à 2000
références. Les 132 parents absents et la très faible marge oracle demeurent
des limites de génération distinctes.

**`PIVOT_GENERATION`**, aucune promotion produit, aucun run national,
aucun fold0/fold1, aucune revendication de généralisation acquise.
