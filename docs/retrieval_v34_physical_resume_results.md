# V34 — correction, accélération et reprise du run complet

11 septembre 2026. **PASS_PHYSICAL_RESUME**, pas un gate de qualité top100.
Le run23571 a réellement repris à18:06 sur le Mac, processus57540, sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/`.

## Résultat technique

- Correction BAN `4bd7a39c` : départage scalarisé uniquement lorsque les
  priorités existantes étaient toutes égales et provoquaient un TypeError.
- Runtime/reprise `68b11655`, parallélisme borné `26f0f974` : projection
  géographique sans perte des sidecars, caches bornés par lecteur et arguments
  complets, libération des index entre étapes, recherches directes par
  sous-lots8 et deux processus seulement pour les requêtes legacy indépendantes.
-52 tests passent; preuve JUnit dans `retrieval_v34_fast_resume/unit_tests.xml`.
-21 sorties communes comparées exhaustivement, multiensembles compris : raw,
  BAN, adresse, géographie, parents, relations, preuves, direct, joint, pools,
  et deux Masters avec tous les scores/rangs. Pilote32 identique au lot ancien;
  lot704 réparé identique en ordre inverse et sous-lots8.64 entrées distinctes
  couvertes par ces contrôles complets, pas une évaluation de recall.
- BAN balayée sur23571 entrées, sans erreur. Les704 sorties BAN anciennes
  sont exactement identiques. Aucun label de vérité ouvert par ces contrôles.

## Mesures, pas une promesse de durée

| Exécution | Temps pour32 | Pic RSS agrégé pertinent | Nouveau swapout |
|---|---:|---:|---:|
| Ancien lot32, même entrée que le pilote |172,41s|6,73Gio, processus unique|0|
| Optimisation séquentielle après rehash des sources |138,55s|7,02Gio|0|
| Pilote final parallèle, offset32 |75,90s|10,23Gio parent/enfants|0|
| Lot704 final parallèle |79,11s|10,14Gio parent/enfants|0|
| Répétition inverse, quatre sous-lots8 |184,06s|6,02Gio parent/enfants|0|

La répétition8 sert à tester l'indépendance du découpage, pas à choisir le
débit produit. Les22 anciens lots complets ont une médiane169,46s, plage
127,72–187,90s. Les caches et la difficulté des entrées influencent les temps;
le rapport~2,2x entre le pilote et son ancien lot n'est pas une garantie
nationale. La projection conservatrice des deux pilotes est15,70h restantes
et9,17Gio de sorties, à réestimer sur les nouveaux lots réellement exécutés.

Premier lot **inédit** après reprise (offset736) :112,90s, pic agrégé9,43Gio,
zéro nouveau swapout. Avancement effectivement scellé768/23571, lot suivant
en cours. Ce point est plus représentatif que les pilotes déjà réchauffés :
projection linéaire restante~22,4h, encore fondée sur un seul nouveau lot.
Comparaison indicative à l'ancienne médiane169,46s :~33% de temps en moins,
pas une démonstration appariée de gain sur toute la population. Ne pas annoncer
16h ou2x comme durée/accélération garantie du complet.

Deux premiers prototypes conservant trop de lecteurs/caches ont été arrêtés
par le garde ressources. Ils sont conservés comme échecs et exclus du gate.
La version finale conserve14Gio de plafond RSS agrégé pour l'étape parallèle,
30min/lot, zéro swapout supplémentaire et36h/run; pas d'augmentation du budget.

## Autorités et reprise

- Préparation :
  `16dbf6bcfbcd4bff503c1ad90d75b54ece4ac6bb671386fe9115b4ec2c677c45`.
- Gate final :
  `01f4601df70c9c6df33f3e7b3075f661aee472f497b04d19de866fd5581ee342`.
- Les22 anciens lots/704 requêtes restent intacts. Les enveloppes d'adoption
  citent leurs manifests et le nouveau certificat de compatibilité.
- Le lot704 est déjà recalculé et scellé par le gate; le nouveau travail
  commence à736. Checkpoint32, processus neuf par lot, évaluation séparée
  seulement après scellement de toute la génération23571.
- `launch.json` épingle l'évaluateur, la politique et les accélérateurs;
  `generation/progress.json` publie l'avancement. Les logs enfants détaillent
  le legacy. Une erreur publie `status.json` STOP, sans retry silencieux.
- Caffeinate empêche la veille inactive pendant le traitement, sans modifier
  les préférences système. Aucune dépense, aucun GPU/service externe.

Sources, modèles, caps et recette v23/v30 restent inchangés. Succession v32
commune, Master2000 uniquement. Folds0/1 fermés, aval gelé; métriques exacte et
opérationnelle et segments historique/V2/V3 restent séparés. **Aucun nouveau
recall ni READY_FOR_FOLD1 n'est annoncé par cette reprise physique.**
