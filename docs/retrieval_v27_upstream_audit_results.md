# V27 — pourquoi les proches disparaissent, et où passent les noms

10 septembre 2026. Verdict **`PIVOT_GENERATION`**. Audits terminés; aucun
nouveau canal intégré, aucun gain de Recall revendiqué. Protocole/code
`fb28a1fd`, amendement d'exécution géographique `1c526ca5`.

## Conclusion

Les **12 entreprises proches mais absentes ne sont pas perdues à une frontière
de commune**. Elles sont éliminées par les budgets d'une recherche ordonnée
d'abord par distance : dix au top512 parents, deux en amont au top4096 sites.
Il y a donc un problème concret de sélection des voisins à examiner, pas une
preuve qu'élargir le rayon ou supprimer la commune suffira.

Pour les noms, **aucune perte des champs SIRENE courants autorisés examinés**
entre les fichiers bruts, le dossier et le portfolio. Même absence de perte
entre RNE canonique et dossier/portfolio sur les valeurs vérifiées. Les huit
correspondances exactes supplémentaires trouvées dans les historiques SIRENE
sont déjà présentes dans les sources du canal temporel existant. Ce n'est pas
une nouvelle source de huit récupérations.

## 1. Géographie : attribution des exclusions

Rejeu de la fonction v16 inchangée : même stock SIRENE, même résolution BAN
CRM, mêmes distances, ordre et caps. Les sorties anciennes et rejouées sont
égales par ensembles de valeurs, y compris les nombres flottants (hors choix
du témoin SIRET parent ex aequo). Aucun label consulté par la génération des
voisins. Les vérités servent seulement ensuite à l'attribution.

| Cause exclusive du canal géographique | Toutes les 1034 | 132 parents absents des preuves |
|---|---:|---:|
| Parent présent dans les voisins v16 | 457 | 0 |
| BAN non résolue | 168 | 30 |
| Aucun site parent avec coordonnées valides | 132 | 31 |
| Sites valides uniquement hors commune CRM | 34 | 7 |
| Aucun site parent dans les 4096 premiers sites locaux | 147 | 47 |
| Parent après les 512 premiers parents | 96 | 17 |

Ce sont les causes du **canal géographique**, pas l'attribution end-to-end du
retrieval multicanal. Une absence de coordonnées ou de point BAN reste une
inconnue, jamais une preuve de distance. Si le parent a à la fois des sites
hors commune proches et des sites locaux éloignés, la cause exclusive suit
le chemin local; le drapeau complémentaire publie ce cas. Le plus proche
site est exclu par la commune pour 45 requêtes, dont 8 des 132 absents.

### Les douze proches

- Dix : premier site local aux rangs 642 à 3703, parent aux rangs **618 à
  3465** avant la coupe512.
- Deux : premier site local au-delà de4096; le rang exact au-delà du plafond
  n'a pas été mesuré, il n'est pas inventé.
- Distances au site parent le plus proche : **154,4 à 493,4 m**.
- Aucun de ces douze plus proches sites n'est exclu par la commune.
- Tous les512 parents conservés sont plus proches ou à égalité de distance
  avec le site local du parent manqué : contrôle indépendant sur les sorties.
- Neuf points BAN non déclarés faibles, trois faibles; pas de garantie de
  précision, ni preuve de même bâtiment/rues adjacentes.

Le code utilise déjà les accords numéro et similarité de voie, **uniquement
après la distance dans l'ordre de tri**, donc essentiellement pour les ex
aequo. Les ajouter une nouvelle fois comme simples tie-breaks ne résoudrait
pas cette perte. Aucun rayon ni plafond augmenté à partir de ces douze labels.

## 2. Noms : contrôle de l'amont du portfolio

Jointure des parents du panel vers tous leurs sites dans les fichiers sources.
Champs autorisés : dénomination légale, dénominations usuelles1/2/3, sigle,
enseignes1/2/3 et dénomination usuelle d'établissement. Les noms/prénoms de
personnes et représentants ne sont pas utilisés dans ces comparaisons CRM.

| Source | Paires champ/valeur/sujet distinctes | Absentes du dossier pour cette source | Absentes du portfolio global |
|---|---:|---:|---:|
| SIRENE courant, unité légale | 1093 | 0 | 0 |
| SIRENE courant, établissements | 26679 | 0 | 0 |
| SIRENE historique, unité légale | 1491 | 1491 | 520 |
| SIRENE historique, établissements | 29809 | 29809 | 1834 |

Les stocks historiques ne figurent pas dans les inputs de ce dossier v3 :
leurs absences sous provenance SIRENE_HISTORY sont une **source non branchée
à ce dossier**, pas des pertes silencieuses de l'import courant. Certaines
valeurs sont déjà présentes via une autre source ou un autre rôle. Une
couverture globale n'établit pas une présence sur chaque document/site.

RNE canonique : **2023 paires valeur/sujet/type de nom**, aucune absente du
dossier ni du portfolio global. Contrôle structurel sans similarité CRM et
sans réexport de ces valeurs : le rôle LEGAL/USUAL seul ne garantit pas que
la valeur n'est pas un nom de personne.

Sur les1034 requêtes, les champs SIRENE courants autorisés donnent212 égalités
nominales CRM, les historiques215. Il y a **8 égalités historiques absentes
du portfolio** : 5 contrôles, 3 difficultés historiques; 6 requêtes déjà
HIT au Master v24 et 2 évincées réservoir→Master. Aucun des132 parents absents
n'a d'égalité CRM avec les champs SIRENE courants ou historiques examinés.

Vérification de nouveauté : les **8/8 noms sont déjà présents dans les
projections sources scellées du canal temporel**, référencé dans les recettes
v16/v18. Tous les fichiers source SIREN/SIRET de ce canal ont été vérifiés
contre les SHA du manifeste d'index pour cette jointure. Cela ne mesure pas
leur rang naturel ni ne relit tous les documents physiques de cet index.

### Ce qui reste non vérifié

Le ZIP RNE brut local représente15 135148871 octets compressés,
177 361078320 octets décompressés,5470 membres. Aucun index membre/SIREN
retrouvé; les identifiants des observations canoniques ne donnent pas ce
chemin. **Pas de scan intégral de ces données dans cet audit court.** La
complétude brut RNE→canonique sur les1034 reste inconnue. L'inspection du
parseur n'est pas une preuve empirique de complétude. Aucune conclusion
« toutes les sources brutes sont impeccables » n'est autorisée.

## 3. Intégrité, ressources et résultats inchangés

1034 requêtes, folds2/3/4 uniquement, dont522 difficultés et512 contrôles;
aucun label corrigé/exclu, aucun statut retiré. Compteurs Master v24 recopiés
séparément : **823 exacts / 825 opérationnels**, dont311/313 difficultés et
512/512 contrôles. V24 non promue, v23 référence de recherche. Pas de nouveau
Master2000/PreRank500/Recall100, Wilson, SLO ou national; modèles aval gelés.
Panel réutilisé, aucune revendication de généralisation inter-CRM.

**76 tests passent**. Vérification supplémentaire reproductible :

```sh
TMPDIR=/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v21-test-runtime.GkBAkd PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/audit_retrieval_v27_results.py
```

Verdict `PASS_V27_OUTPUT_RECONCILIATION` : hashes de résultats, labels du
ledger bornés aux folds2/3/4, égalité préflight16/complet, égalité numérique
exacte du replay géographique, présence parent et domination par512 voisins,
absence de pertes SIRENE courantes, huit noms déjà présents dans les sources
temporelles. Ce n'est pas un géocodeur indépendant ni une validation du brut RNE.

- Noms : corps d'audit14,983 s; RSS676478976 octets (~0,63 Gio); zéro swapout.
- Géo : première tentative sous4 Go DuckDB arrêtée sur mémoire, sans
  manifeste final admissible. Aucun résultat scientifique tiré de cet échec.
- Rejeu géo sous8 Go DuckDB/deux threads :41,565 s; RSS9531883520 octets
  (~8,88 Gio); zéro swapout, sous le plafond14 Gio.
- Temps ci-dessus hors vérification des hashes des grandes sources; pas
  des mesures SLO produit. Mac/CATNAT, aucune dépense externe.
- Préflights admissibles : `preflight16-final` pour chaque audit. Les essais
  techniques antérieurs sont écartés; conversion Arrow corrigée avant gel.

Les manifestes publient historique/cohorte, V2/V3, folds, origine, type de
label et étape de perte, sans toucher au dénominateur. Aucun X/Y brut exporté;
les Parquet de replay ne sont pas ajoutés à un pool de candidats.

## 4. Choix proposé pour la prochaine expérience — non exécutée

Tester **la sélection des voisins avant la coupe**, plutôt que simplement
élargir un rayon ou réimporter les mêmes anciens noms. Hypothèse : une
compatibilité souple nom/adresse et une qualité BAN explicite pourraient
être plus utiles qu'un ordre où la distance domine toujours.

Avant un fit ou un gros run : vérifier que ces indices distinguent réellement
les bons parents dans les voisins naturels, contrôles inclus. Le bon parent
ne doit jamais être fourni au générateur, et une adresse différente ne doit
pas devenir une exclusion dure. Recette unique préenregistrée, développement
OOF2/3/4, budgets identiques et gains/pertes exacts/opérationnels séparés.
Les deux absences avant4096 imposent en outre d'examiner la génération amont;
un reclassement du pool existant ne peut pas les récupérer.

Ce diagnostic ne prouve ni douze récupérations futures, ni la possibilité
d'atteindre99,2 %. Verdict scientifique : **`PIVOT_GENERATION`**.

## Autorités

Configuration SHA256
`f6f0ed019069defc4c208654853a2f6e1ae351e4bb91377a046c877e3790e790`.
Sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/` :

- `retrieval_v27_names/f6f0ed019069defc/manifest.json`, SHA256
  `130b4e119870aa18b196e40f2c0db8494d18cb890141e994c6c10a150dc2d234`.
- `retrieval_v27_geo/f6f0ed019069defc-memory8/manifest.json`, SHA256
  `201c6a05c31cdd8d1a747545eea7f03a285edde46bed5d25f4a155a2d8a18504`.

Le premier répertoire géo `f6f0ed019069defc` demeure incomplet/non admissible.
Les manifestes réussis scellent séparément l'implémentation utilisée, toutes
les sources consultées pour l'audit principal et les résultats Parquet.
