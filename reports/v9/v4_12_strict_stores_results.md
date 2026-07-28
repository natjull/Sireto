# V4.12 — Résultats du Gate A des stores stricts

Date d'exécution : 28 juillet 2026

## Verdict

**`GO_V412_STRICT_STORES_SANDBOX`**

Le futur moteur unitaire peut lire, sous la sandbox prévue, les seules
entrées CRM autorisées et les trois magasins nécessaires au retrieval :

- les partitions SIRENE locales ;
- les caches TF-IDF correspondants ;
- le lookup SIRET du snapshot SIRENE.

Les deux contrôles indépendants post-build rendent également `GO`, avec
respectivement **20 848 / 20 848** et **11 898 / 11 898** contrôles réussis.

Ce gate certifie les magasins, leurs contenus, leurs limites de lecture et
l'isolement du worker. Il **ne mesure encore ni Recall@100, ni qualité du
classement, ni latence par requête**. Il n'autorise ni entraînement de modèle,
ni ouverture du test final, ni déploiement.

## Artefacts publiés

| Élément | Valeur |
|---|---|
| Build ID | `9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4` |
| Certification | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/certifications/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4` |
| Audit | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4` |
| Verrou d'exécution | `config/v4_12_strict_stores_execution_lock.json` |
| SHA-256 du verrou | `265bc418d95a1de1902773b7f5548b5607a2b1360722192658dbddb544a0630d` |
| Commit source scellé | `72cf9f9911802280779489dd7972991e97f11048` |
| Manifeste certification | `6854b115f06eb1b1d0dacc8dfc44e2d7d6d470427d80628253ca09a79a443d2d` |
| Manifeste audit | `3b29779079ed37e6975338e590893185df197bed3be9508d96a8be6eaee177a7` |
| Ledger des lectures | `73489853e5e343bcf3072dd311615dd4201e8cc89510b130c6118eda184fed51` |

Les deux racines publiées sont en `0555` et leurs neuf fichiers sont en
`0444`.

## Résultats du probe réel

| Contrôle | Résultat |
|---|---:|
| Requêtes dev | 1 456 |
| Clés géographiques distinctes | 648 |
| Routage INSEE | 1 449 |
| Routage code postal | 7 |
| Partition manquante | 0 |
| Partitions vérifiées | 648 |
| Lignes physiques lues dans les partitions | 8 030 285 |
| Caches TF-IDF vérifiés | 648 |
| Lignes du pool filtré et dédupliqué | 4 764 472 |
| Cache manquant | 0 |
| Cache reconstruit | 0 |
| Écriture de cache | 0 |
| SIRET contrôlés dans le lookup | 10 000 |
| SIRET absents du lookup | 0 |
| SIRET supplémentaires | 0 |
| Taille du lookup | 42 322 035 SIRET |
| Plafond du lookup par appel | 100 SIRET |
| Pic mémoire RSS | 1 956 773 888 octets |
| Durée totale du worker | 165,458 s |

Le temps du probe se décompose en 60,475 s pour les partitions, 48,216 s
pour les caches et 21,296 s pour le lookup. Ces durées vérifient la capacité
à ouvrir les magasins ; elles ne représentent pas la latence du futur moteur
requête par requête.

## Isolement vérifié

Le worker a confirmé :

- une lecture autorisée réussie ;
- le refus effectif de l'oracle ;
- le refus effectif de la racine d'audit de l'oracle ;
- le refus effectif des écritures hors staging ;
- le refus effectif du réseau.

Les déclarations enregistrées sont toutes négatives : aucun label, oracle,
modèle, résultat candidat historique ou réseau n'a été ouvert ; aucun cache
n'a été reconstruit et aucune écriture n'a eu lieu hors staging.

La liste blanche de l'enfant contient 1 945 fichiers. Le ledger parent
contient exactement 1 954 fichiers et a été entièrement rehaché par les deux
audits indépendants, soit 7 225 618 142 octets.

La frontière de confiance reste volontairement explicite : la sandbox
contrôle les données du projet accessibles au worker et bloque réseau/fork,
mais le runtime local partagé (`/System`, `/usr`, `/opt/homebrew`) n'est pas
présenté comme protégé contre un acteur administrateur ou un processus du
même compte.

## Incidents conservés

Trois arrêts sûrs ont précédé le build publié :

1. un hash de bibliothèque Python tronqué à 63 caractères a révoqué le
   premier verrou avant toute exécution ;
2. le premier worker complet s'est arrêté sur `Path.cwd()` refusé par la
   sandbox, sans publication ;
3. le second worker a terminé son calcul mais APFS `noowners` a refusé la
   promotion d'une racine déjà passée en `0555`, toujours sans publication.

Chaque cause a été reproduite, corrigée, retestée, contre-auditée et suivie
d'un nouveau verrou. Aucun artefact incomplet n'a été promu.

## Décision et suite autorisée

Le Gate A est franchi. La seule suite autorisée est de préenregistrer puis
d'implémenter le **moteur unitaire V4.12** :

1. lire une requête CRM sûre ;
2. choisir sa partition sans utiliser la vérité ;
3. produire au plus 100 candidats avec le retrieval gelé ;
4. vérifier une parité exacte avec la référence historique sur les
   1 456 requêtes dev ;
5. faire comparer les candidats à l'oracle par un évaluateur séparé.

Le ranker, le decider, le risk model et l'accepteur restent gelés jusqu'au
gate retrieval.
