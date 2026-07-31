# V4.12 — Contre-audit métier complet des 53 changements de top 1

Date : 31 juillet 2026  
Périmètre : les 53 dossiers `REVIEW` historiques restants dont la correction exploratoire `nom légal + siège` change le premier choix. Aucun réentraînement, aucun test final et aucune promotion produit.

La table dossier par dossier est publiée dans [`v412_review_rerank_counteraudit_53.csv`](v412_review_rerank_counteraudit_53.csv). Elle contient le CRM, les deux choix comparés, le SIRET adjudiqué, la fiabilité, l'effet de la correction et la famille d'erreur.

## Bilan

| Résultat | Nombre | Part |
|---|---:|---:|
| SIRET exact identifiable | 50 | 94,3 % |
| Ambiguïté métier réelle | 3 | 5,7 % |
| Non résolu | 0 | 0,0 % |
| Choix exploratoire exact | 43 | 81,1 % des 53 ; 86,0 % des 50 exacts |
| Top 1 initial exact, donc correction régressive | 3 | 5,7 % |
| Ni l'ancien ni le nouveau choix n'est exact | 4 | 7,5 % |

Les 50 SIRET exacts constituent des labels utilisables. Les trois ambiguïtés (`AXES - SYGMALAB`, `GROUPE VIKINGS`, `LIGUE AURA HANDBALL`) sont utilisables uniquement pour entraîner ou tester l'abstention ; elles ne doivent jamais être converties en positifs de ranking.

## Ce que le contre-audit démontre

La correction capte un vrai défaut du ranker : dans cette population choisie parce qu'elle change, elle répare 43 dossiers. Les gains proviennent principalement de quatre situations :

- une société immobilière, une filiale ou une autre entité du groupe partage l'adresse avec l'organisation CRM ;
- le ranker préfère un établissement spécialisé alors que le CRM désigne l'organisation générique ou son siège ;
- un ancien exploitant ou une ancienne structure reste lexicalement proche ;
- le nom légal exact était sous-pondéré par rapport à l'adresse.

Mais elle n'est pas déployable comme règle :

- elle dégrade trois dossiers fiables : `IDEF 86`, `CCI EMERAINVILLE` et `AVELIS GROUP` ;
- elle ne peut pas traiter quatre vérités situées ailleurs : deux SIRET exacts sont absents du pool (`GROUPE DELAMBRE`, `SIX ARES`) et deux sont présents mais classés seulement 23e et 30e (`CLINIQUE DE TOURNAN`, `ALCYACONSEIL`) ;
- elle ne sait pas résoudre les trois CRM qui désignent réellement deux entités possibles.

## Preuves et politique d'adjudication

Chaque dossier a été contrôlé d'abord contre le snapshot SIRENE local et les 100 candidats réellement présentés au ranker. Les preuves externes ont été recherchées pour les collisions, changements d'exploitant, transferts et ambiguïtés. Les sources privilégiées sont l'Annuaire des Entreprises, les avis INSEE, les sites officiels, les mentions légales et les actes RNE/BODACC publiés.

Exemples déterminants :

- [IDEF 86](https://idef86.fr/contact/) et son [avis INSEE](https://api-avis-situation-sirene.insee.fr/identification/pdf/26860086300115) confirment que le top 1 initial est le site de l'adresse CRM ;
- la [CCI Seine-et-Marne](https://www.seineetmarne.cci.fr/sites/default/files/2025-08/liste_des_numeros_siret.pdf) publie explicitement `18770918300086` pour le CFA UTEC d'Émerainville ;
- le site [Avelis](https://avelisimmobilier.com/contact/) identifie `AVELIS GROUP` au site de Torcy, ce que confirme le registre de l'établissement `75320135900035` ;
- le registre de [GROUPE DELAMBRE](https://www.pappers.fr/entreprise/groupe-delambre-405356601) identifie `40535660100022` à l'ancienne adresse CRM et son transfert en 2025 ;
- [SIX ARES](https://www.pappers.fr/entreprise/six-ares-842393126) identifie `84239312600029`, radié après fusion en 2024 ;
- l'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/327113627) donne `32711362700019` comme exploitant de la Clinique de Tournan ;
- l'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/402559603) distingue le siège ACOR `40255960300012` de son établissement Habitat inclusif ;
- l'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/843623349) rattache explicitement la clinique de Caen à `84362334900167` ;
- les sites officiels [AXES / SYGMALAB](https://www.axes-44.com/) et [Ligue AURA Handball](https://aura-handball.fr/ligue-auvergne-rhone-alpes-handball/comites-aura-handball) montrent pourquoi leurs CRM ne permettent pas un choix légal unique.

La politique reste « identité active au snapshot » ; lorsqu'une adresse CRM correspond sans doute à un SIRET fermé ou transféré, la date inconnue est conservée dans la cause d'erreur et aucune entité actuelle co-localisée n'est substituée silencieusement.

## Décision

Verdict : **`PIVOT_FROM_RULE_TO_TRAINABLE_SIGNAL`**.

Le signal `nom légal + rôle du site` mérite d'entrer dans un apprentissage candidat, mais pas sous forme de bonus fixe. Le prochain entraînement devra :

1. réunir les 27 labels exacts R30 et les 50 labels exacts de ce contre-audit, en conservant séparément les six ambiguïtés des deux lots ;
2. ajouter explicitement des négatifs co-localisés, intra-SIREN et anciens exploitants ;
3. produire des prédictions hors échantillon par groupes SIREN, sans utiliser les trois ambiguïtés comme positifs ;
4. évaluer séparément les erreurs de retrieval, de ranking et d'abstention ;
5. rester un candidat de développement : toute promotion exigera un nouveau lot indépendant jamais utilisé dans cette analyse.

Ce bilan autorise donc un entraînement borné sur développement. Il n'autorise ni une règle en production, ni une réouverture du test final déjà consommé.
