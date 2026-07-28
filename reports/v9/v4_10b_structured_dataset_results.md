# Résultats dataset V4.10b — features structurées sans fuite de population

Date : 28 juillet 2026

Verdict : **`GO_FREEZE_TRAINING_PLAN_V410B`**

## Résultat

Le build V4.10b est disponible ici :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10b_structured_acceptor/3ad8e97ce0118e8c`

Il remplace pour tout futur entraînement le build V4.10
`0d6b87fd50fb550c`, invalidé avant le premier fit par l'audit statistique.
Les trois parquets de scènes restent physiquement identiques ; le changement
porte sur le catalogue et l'ordre autorisé au modèle.

| Bloc | Compte | SHA-256 d'ordre |
|---|---:|---|
| `CURRENT80` | 80 | `e50086608ca3e60071e2575fbd8a0ca7c8ba99fe87251894ee04bf9b1b57cfe5` |
| structuré V4.10b | 641 | `4ff0eb4e8cc33850742bf4d9c0ddb599cc9abb500d6b60bb3e5dc6a80b9cd13b` |
| structuré standardisé | 157 | `c1769136cb80f9f2273406a1045f223f99a088f4270b4dc8ef9097e8234d61ed` |
| structuré binaire non standardisé | 484 | `7f0a1c01d8ed402c577b128cfe1aeb05b342772af0b132b597a432cce8409e89` |

Les 58 copies sémantiques sont cataloguées avec une expression `alias_of`
typée et vérifiées ligne par ligne. Les 16 signaux d'instrumentation
retrieval rejoignent les 75 signaux de provenance déjà `audit_only`.
Aucune suppression n'a été apprise par corrélation ou à partir des labels.

## Intégrité

- 7 003 scènes historiques, 94 scènes difficiles consommées et quatre scènes
  descriptives verrouillées ;
- valeurs `CURRENT80` bit à bit identiques entre V4.10 et V4.10b sur les
  trois populations ;
- mêmes hashes parquet :
  - historique :
    `23016e2e47df2b06f57f66a7b4ba518689eb22a03ce3c1968d14f58738ecc260` ;
  - hard :
    `88d364a0c496515d5e2e57d8e13bb5976e0212c7bff21b076aeaa041fd4b2536` ;
  - locked :
    `203b57ba2d352cea50791f14b6f8a805d4c00da06a598afae922da84d99439e9` ;
- catalogue :
  `be41a019a24e0d9a24a1be17baa408c06515a0b17b90d8e184c1f5a09f718d92` ;
- manifeste :
  `4c62f1bd3e1fc77cda806946d3751d397ee1723095d3fc73b34e8dd782006614` ;
- zéro fit, zéro seuil, zéro random, zéro dev frais et zéro test ;
- 369 tests passent.

## Portée

Ce verdict autorise seulement le gel d'un nouveau plan d'entraînement
V4.10b et l'implémentation de son runner. Il ne valide aucun accepteur et
n'autorise ni population fraîche ni test final.

Commits :

- politique V4.10b : `eb85597`, clarification `d500fe2` ;
- builder et tests : `f78a9ba`.
