# Incident de lancement du retrieval unitaire V4.12

## Verdict

La première tentative autorisée par le verrou `bf82e74` s'est arrêtée avant
le démarrage du moteur de retrieval. Elle n'a traité aucune requête, n'a
publié aucun candidat et n'a ouvert ni oracle, ni historique, ni modèle.

Le verrou `bf82e74` est révoqué. Une nouvelle exécution exige un verrou
recalculé sur le correctif audité `58fabf3`.

## Symptôme observé

Le processus enfant s'est terminé en environ cinq secondes avec :

```text
ModuleNotFoundError: No module named 'xgb_matcher'
```

Aucun manifeste final worker ou parité n'a été créé. Les répertoires de
publication sont restés vides.

Le staging privé de cette tentative est conservé comme trace d'incident :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/v4_12_unit_retrieval/.run-3ab14e8bd3c94b01b8529c58e45d5dee
```

Il contient uniquement les contrôles et sources privés copiés avant le
lancement, le runtime privé et des répertoires de sortie vides.

## Cause

Le paquet Python privé était bien copié et scellé dans le staging, mais la
politique Seatbelt autorisait la lecture des fichiers Python individuels sans
autoriser la découverte du répertoire de paquet. Python ne pouvait donc pas
résoudre le module `xgb_matcher.v412_unit_retrieval`.

## Correctif

Le commit `58fabf3` :

- autorise la lecture du seul répertoire de paquet privé scellé ;
- fixe `PYTHONPATH` sur le staging privé ;
- désactive le user-site et les ajouts implicites de chemins Python ;
- teste la vraie commande `python -m xgb_matcher.v412_unit_retrieval --help`
  sous le profil Seatbelt final ;
- supprime uniquement le staging enregistré de la tentative courante lorsque
  le worker retourne un code non nul.

Le correctif n'ajoute aucun accès au dépôt, à l'oracle, à l'historique, aux
modèles ou au réseau. Les écritures restent limitées au staging privé.

## Vérifications

Deux revues indépendantes ont conclu `GO_IMPORT_PATCH` et
`GO_IMPORT_PATCH_2`. Les tests natifs macOS atteignent le parseur réel du
worker sous sandbox. La suite ciblée passe 153 tests et la suite complète
823 tests.

La relance reste interdite jusqu'à la création et au double contre-audit d'un
nouveau verrou d'exécution.
