# V4.12 — Décision sur la preuve retrieval finale

## Décision

**`PIVOT_NEW_HOLDOUT_REQUIRED`**

V4.12 franchit le gate développement, mais ne possède pas de preuve finale
indépendante réutilisable. Le test final historique et V4-Fresh sont déjà
consommés. Aucun nouvel export CRM local admissible n'existe.

Cette décision interdit :

- de présenter le Recall@100 dev de 100 % comme une performance finale ;
- de rejouer un ancien test ou holdout ;
- de transférer la performance finale d'une autre politique retrieval ;
- de dégeler le ranker ou l'accepteur sous la directive active.

## Pourquoi le résultat final existant ne s'applique pas

Le test final sélectif a mesuré une admission multicanal :

- `current_sparse` pondéré 2 ;
- nom mots et caractères ;
- adresse mots ;
- tête SIREN ;
- égalités nom et adresse ;
- quotas overlay 1/10 ;
- profondeur interne 5 000 ;
- budget final 100.

Cette politique a obtenu :

- sparse seul : 2 059/2 128 = 96,758 % ;
- admission multicanal : 2 116/2 128 = 99,436 %.

V4.12 unit est une autre politique :

- sparse actif seulement ;
- nom, adresse et rescues déterministes ;
- profondeur 500 ;
- RRF `k=60` ;
- `include_closed=false` ;
- aucun overlay multicanal ou sibling SIREN.

Identités V4.12 :

- policy SHA-256 :
  `340daf0ec22349d4d0f1de77c4bdebb61d3ba7d4822753ef852959f29a59d818` ;
- source :
  `167f5058e4329f24c629e4464ffa7c8991f6350739caeee3080fbad616be11d0` ;
- build :
  `d2915fe7747b9b219e7a0dce400052847c913417bd240c7d94df6fb8bafedd1a`.

La performance de l'admission multicanal ne peut donc pas être transférée à
V4.12. Le contraste 100 % dev / 96,758 % sparse sur l'ancien test confirme
précisément le besoin d'une population nouvelle.

## Inventaire des populations

Le registre V4.11 couvre les 23 609 lignes de la source CRM locale :

- 23 384 consommées par historique et V4-Fresh ;
- les 225 restantes, atypiques et sans `SERVICE ID`, consommées ensuite par
  le challenge descriptif V4.11.

Sont également consommés :

- le test final historique ;
- le holdout V4-Fresh ;
- le random V4.8 ;
- le holdout neural dérivé d'une source déjà utilisée.

L'inventaire de chemins et métadonnées ne révèle aucun nouvel export CRM
postérieur au registre du 28 juillet 2026. Les fichiers plus récents sont du
code, des rapports, des locks ou des artefacts dérivés.

## Ce qui reste validé

Sur le dev historique V4.12 :

- couverture identifiable : 1 217/1 456 = 83,585 % ;
- Recall@100 : 1 217/1 217 = 100 % ;
- zéro vérité absente ;
- plafond 100 respecté.

Ce résultat justifie de conserver V4.12 comme candidat retrieval. Il ne
justifie ni certification finale, ni déploiement, ni dégel aval.

## Condition de sortie du pivot

Un nouvel export CRM, indépendant des 23 609 lignes enregistrées, doit être :

1. reçu brut et hashé avant toute qualification ;
2. contrôlé contre le registre de consommation ;
3. qualifié sans utiliser retrieval, rang ou score ;
4. scellé avec `MATCH_EXACT`, `AMBIGUOUS` ou `UNRESOLVED` et preuves
   traçables ;
5. séparé physiquement entre requêtes sûres et oracle ;
6. évalué une seule fois avec V4.12 gelée, 100 candidats maximum ;
7. publié avec couverture, Recall@100, nombres bruts et Wilson 95/99.

L'utilisateur ne sera pas sollicité comme validateur. La qualification devra
reposer sur les preuves déterministes disponibles ; les cas non démontrables
resteront `AMBIGUOUS` ou `UNRESOLVED`.
