# Audit aval — ranker, decider et décision AUTO

## Verdict

**Le retrieval sélectif à 100 peut être conservé. Le bundle historique
`ranker + decider + risk model` ne doit pas être promu ni simplement
recalibré.**

La cible est :

```text
top-100 retrieval gelé
  → calcul canonique des preuves candidat
  → ranker final SIRET unique
  → top-1 + top-2 + preuves directes + forme de la scène
  → accepteur entraîné sur la correction SIRET exacte
  → AUTO_MATCH ou REVIEW
```

Cela ne consiste pas à jeter tout le travail historique. Les signaux utiles du
decider sont conservés, mais appris dans le ranker final. La décision AUTO
reste une tête distincte, car elle utilise des informations qui n'existent
qu'après le classement : écart entre les deux premiers, densité de la scène et
concurrence entre établissements.

## Défaut principal de la référence historique

La référence annoncée à 74,5 % d'AUTO et 99,84 % de précision n'est pas
certifiable avec les artefacts versionnés :

- 1 428/2 512 scènes, soit 56,847 %, ont exactement le même SIRET en top-1 et
  top-2 ;
- ces 1 428 scènes artificiellement sans concurrence sont toutes acceptées ;
- sur les 1 084 scènes dont les deux premiers SIRET sont distincts, la
  couverture AUTO tombe à 444/1 084, soit 40,959 % ;
- ces 444 AUTO contiennent 6 erreurs, soit 98,649 % de précision ;
- sur l'ensemble du fichier reproductible, il y a 1 866 succès sur 1 872 AUTO,
  soit 99,679 %, et non 99,84 %.

Le chiffre de 99,84 % correspondrait à une réinterprétation humaine de trois
des six erreurs. Aucun artefact d'adjudication versionné ne relie cette
correction au benchmark ; elle ne peut donc pas servir de certification.

## Où se situe la perte

| Étape historique | Succès | Taux |
|---|---:|---:|
| Vérité présente dans le top-20 | 2 490/2 512 | 99,124 % |
| Bon SIRET classé premier | 2 150/2 512 | 85,589 % |
| AUTO correct parmi les AUTO | 1 866/1 872 | 99,679 % |

Sur ce benchmark historique, la perte principale intervient donc entre le pool
de candidats et le premier choix. Le retrieval sélectif récent a par ailleurs
déjà franchi son objectif global à 100 candidats sur son périmètre exact.

## Le decider apporte bien un signal

Les deux anciens modèles ont été rejoués sur exactement les mêmes scènes V7 :

| Split | Ranker Hit@1 | Decider calibré Hit@1 | Delta |
|---|---:|---:|---:|
| train | 79,069 % | 83,731 % | +4,662 points |
| dev | 75,850 % | 78,883 % | +3,034 points |
| test historique | 76,179 % | 80,189 % | +4,009 points |

Le decider ne doit donc pas être supprimé sans reprendre ses signaux. Mais il
n'est pas nécessaire de conserver deux modèles qui évaluent les candidats :
un ranker final peut apprendre directement ces preuves sur le top-100 gelé.

## Audit des features et des contrats

- Le registre courant contient 54 features candidat. Les datasets et modèles
  V7 n'en contiennent que 47.
- Les 7 features V8 restantes sont présentes dans le code mais absentes des
  données V7 ; elles ne sont donc pas qualifiées.
- Les trois features sémantiques du dataset ranker V7 sont constantes à zéro.
- Les modèles historiques sans noms de features embarqués dépendent
  entièrement d'un ordre externe.
- La sélection automatique de la « dernière » meta est lexicographique et
  choisit actuellement un bundle legacy.
- Une meta référence le même fichier comme ranker et decider.
- Les quatre risk models audités utilisent des seuils différents et ciblent
  tous la correction SIREN, pas SIRET.
- Dans `routing_eval_v7.parquet`, 1 539/16 621 lignes, soit 9,259 %, ont le bon
  SIREN mais le mauvais SIRET. Un risk model ciblé SIREN les apprend comme des
  succès.
- L'accepteur V9 existant reçoit 20 informations de scène, mais aucune des
  sept preuves directes auditées de nom et d'adresse.
- Les datasets V7 ne conservent que les requêtes ayant exactement un positif
  dans leur pool. Leurs Hit@1 sont donc conditionnels, pas end-to-end.

## Conséquence d'architecture

Le problème historique n'est pas « XGBoost était une mauvaise idée ». Le
problème est l'exécution :

1. scènes dupliquées qui rendent la décision artificiellement facile ;
2. vérité SIREN utilisée pour sécuriser un produit évalué au SIRET ;
3. versions de modèles et seuils ambiguës ;
4. contrats de features incompatibles ;
5. erreurs de retrieval absentes de certaines données aval ;
6. accepteur V9 privé des preuves directes du candidat choisi.

Le prochain modèle doit donc être reconstruit sur les vrais pools du retrieval
gelé, sans doublon de SIRET, sans injection de positif dans les scènes
d'évaluation et avec une cible SIRET exacte.

## Reproductibilité

Audit en lecture seule exécuté par
`scripts/audit_downstream_architecture.py` au commit `a59fb0f`.
Le test sélectif final déjà consommé n'a pas été lu.

Artefacts immuables :

- dossier :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/downstream_architecture_audit_a59fb0f` ;
- `summary.json` :
  `ca6c2f9b7196b82016852d11722748262cd60d918ca00eb2cd855857a91eefaa` ;
- `report.md` :
  `dd89fbb08b7f00f7d62e07161fae629d60bb8f254e9bd296a71667f900e432d0` ;
- `manifest.json` :
  `428d97398ac81f096562c14a8120cb0f1cc826e6f5ba44587de541cd818576c9`.

## Décision

**`PIVOT AVAL`** : conserver le retrieval et les preuves utiles, mais remplacer
le bundle historique par `ranker final SIRET + accepteur exact-SIRET`.
