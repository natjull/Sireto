# Étape2-11 — plafond du SUR, avenant avant tout fit

Références immuables : risque `2cf01765`, attestation `9f032ebc`, code `d65c3dc7`, G0 `9a8ecdd6`. Banc entier17427 dossiers/11483 entreprises, mêmes cinq CV entreprise, mêmes proper/calibration25%/test et mêmes OOF imbriqués R3 content+S1+S2-BASE. Folds population2/3/4 seulement ; aucun jeu fermé, label/pool, signal, ranker, NB, encodeur ou adaptateur réentraîné/modifié. SIREN/SIRET réservés aux cibles/jointures/groupes, aucune nouvelle feature :46 colonnes et paramètres XGBoost/monotonies200arbres du risque précédent strictement conservés.

## Protocole figé

1. **SIREN** : cinq risques nouveaux, cible = SIREN du top1 R3 égal au SIREN humain original (même si vérité SIRET absente des100). Fit sur les OOF proper uniquement ; mêmes calibration et test. Seuils98/99% : plus petit score observé dont borne inférieure Wilson bilatéral95>=cible, ex aequo indivisibles ; si aucun,0SUR/précision indéfinie. Comptes/précision SIREN/exacte/UNION/Wilson/couverture par CV, fold population et poolé. C'est une acceptation entreprise, jamais une validation de site.
2. **UNION** : aucun nouveau fit, relecture/rejeu des cinq risques UNION gelés. Même règle calibration, cibles96/97/97,5/98%, sans choix test. Reproduire strictement la politique97% précédente.
3. **Générateur, diagnostic label-side seulement** : éligible = au moins un positif UNION dans les100 figés. Primary : même architecture réapprise sur OOF proper éligibles uniquement, calibration éligible uniquement, seuil98%, test éligible uniquement. Contraste sans fit : mêmes poids UNION gelés, seule calibration/test conditionnés par présence. Ce double contrôle distingue retrait des absences et changement de distribution du fit ; il ne réentraîne aucun signal. Rapporter dénominateur éligible et couverture rapportée aussi au banc complet. Ces « plafonds » sont conditionnels au modèle/protocole, pas des maxima mathématiques ; filtre impossible à connaître en production, jamais une politique ni une amélioration end-to-end.
4. **Co-adresse** : masque historique scellé des1046 dossiers de la famille exclusive `colocation`, `c37a1a7b`, identique à `FT/family.npy`. Il est défini parmi les erreurs récupérables de l'ancienne base, pas parmi toutes les scènes réellement co-adressées ni les31fauxSUR scène récents. Sur l'UNION risque97% INCHANGÉ : N/SUR/corrects/erreurs/précision/Wilson/couverture dans/hors masque, sommes vérifiées, part des fauxSUR et différence descriptive de précision. Label-side uniquement, sélection historique biaisée, aucune interprétation causale ni filtre déployable.

## 526, ressources, preuve et publication

526 consommés **uniquement pour le point1**, aucun fit/calibration/choix sur eux. Reprendre le top1 de la moyenne des cinq vecteurs R3 déjà figé. Entreprise acceptée si les cinq top1 prédisent le même SIREN que ce top1 moyen et si leurs cinq risques SIREN passent chacun leur seuil appris G0. Accord SIREN, pas accord SIRET : site toujours `A_VERIFIER`. Rapporter98/99, aussi chaque paire de modèles ; aucune nouvelle cible de site.

Smokes réels CV0 pour chacune des deux nouvelles cibles, entraînement sur100 entreprises proper entières choisies par SHA `20261020|ceiling-smoke|SIREN`, mêmes entrées/calibration/test scellés ; vérifier modèles et probabilités sérialisés/rejoués, sans choisir de paramètre/checkpoint. Puis cinq fits SIREN et cinq fits UNION conditionnels ; tous les autres scores réutilisés. Identités candidats/vérités/OOF/rôles et SHA des fichiers consommés avant/après ; seuils revérifiés indépendamment par la fonction Wilson contractuelle.

Namespace neuf `H/step2_risk_ceiling_v1_20261020`. Un worker CPU détaché à la fois, threads1, pas de Metal/encodage/DuckDB nécessaire. Budget7200s total hors attentes documentées, inclut smokes/IO/publication ; réserve publication60s. Garde habituelle reprise sans modification :7,75Gio si aucune VM identifiée,5,5 sinon, attente60s si VM active et free+inactive<8Gio ou superviseur V142/descendant>4Gio ; jamais toucher `hydraulic`. Stop CATNAT libre<10Gio, aucune suppression. Si résultat incomplet, STOP explicite, aucun partiel promu.

Publication automatique isolée `docs/step2_risk_ceiling_v1.{json,md}` + ligne append-only `handover.md`, refuse chemins de publication sales/concurrence. Sources historiques et fichiers étrangers/`*.pending.*` intouchés, aucun push. Rapport français<=30lignes, exactement3tableaux, une recommandation ; détails par CV/fold dans JSON et annexe Markdown. Wilson descriptif, pas garantie98/99%, pas de nouveau jeu indépendant, pas de déploiement.

## Résultats

## Fait
G0 17427 dossiers/11483 entreprises, mêmes5CV/OOF/46features ; aucun signal réentraîné.
Fichiers `scripts/step2_risk_ceiling_v1.py`, `docs/step2_risk_ceiling_v1.{json,md}`, `handover.md`.
## Preuve
Durée 26.3s, pic 0.38Gio ;12fits risque avec smokes,40seuils Wilson indépendants/rejeux/SHA/groupes vérifiés.
| Cible SIREN, Wilson calibration | SUR | Précision test %, Wilson95 | Couverture % |
|---|---:|---|---:|
| 98% | 834 | 97.00 [95.61;97.96] | 4.79 |
| 99% | 0 | indéf. | 0.00 |
526 descriptifs entreprise98/99 : 98%=0SUR, P=indéf., C=0.00%, 99%=0SUR, P=indéf., C=0.00% ; site A_VERIFIER.
| UNION, cible Wilson calibration | SUR | Précision test %, Wilson95 | Couverture % |
|---|---:|---|---:|
| Complet 96% | 7635 | 96.41 [95.97;96.81] | 43.81 |
| Complet 97% | 4459 | 97.78 [97.30;98.17] | 25.59 |
| Complet 97.5% | 2369 | 98.02 [97.37;98.50] | 13.59 |
| Complet 98% | 0 | indéf. | 0.00 |
| Oracle présence98% frozen | 4304 | 98.28 [97.85;98.63] | 26.36 |
| Oracle présence98% refit | 3929 | 98.14 [97.67;98.52] | 24.07 |

| Co-adresse historique, risque UNION97% | SUR/N groupe | Précision %, Wilson95 | Couverture groupe % |
|---|---:|---|---:|
| dans | 35/1046 | 2.86 [0.51;14.53] | 3.35 |
| hors | 4424/16381 | 98.53 [98.13;98.85] | 27.01 |
| total | 4459/17427 | 97.78 [97.30;98.17] | 25.59 |
Co-adresse : 34/99 fauxSUR ; masque historique1046, pas une mesure causale.
## Écarts / choix
Présence UNION :16325/17427. Oracle filtre proper/cal/test ; contraste poids figés. Couvertures oracle conditionnelles, jamais end-to-end.
## Reste / risques
Wilson descriptif/adaptatif, entreprise corrélée,526 consommés ; aucune promotion. Détails par CV/fold publiés.
Recommandation unique : Conserver A_VERIFIER ; ne pas promouvoir de statut SUR sur ces jeux consommés.

### Lecture scientifique et attestation finale

Avenant/code figés `8841d541`, publication automatique `a5ab226d`. `H/step2_risk_ceiling_v1_20261020/postverify.py` passé sans fit :12 modèles risque/44846 vecteurs test-pilote rejoués depuis les entrées OOF scellées,107 références SHA vérifiées,40 seuils indépendants, rôles entreprise disjoints, cibles exactes/UNION d'origine et politique97% précédente inchangées. Preuve `postverification.json` SHA `ca8a6a5bdc2d685d41cb8169f406279985f41098868478386ec93afe0ab84996`. Run/publication26,345s ; jusqu'au contrôle complémentaire321,670s depuis lancement, toujours<2h ; pic412631040octets=0,3843Gio. Aucun signal réentraîné,12 seuls fits risque (deux smokes inclus), aucun worker restant ni VM/jeu fermé/label/pool/fichier étranger/pending/push touché. Une ligne vide de présentation sépare les tableaux2/3 : rapport externe30lignes et3tableaux, code scientifique et métriques inchangés.

**Niveau de cible** : SIREN98% ne produit des SUR que dans le CV entreprise0 (distinct du fold population0 fermé). Calibration888/897, Wilson inférieur98,10%, mais test809/834=97,00%, Wilson[95,61;97,96]%. Les quatre autres CV et les cinq CV à99% s'abstiennent. Abaisser la granularité au SIREN ne valide donc pas ici un statut entreprise98%. Les526 donnent0SUR avec consensus des cinq modèles aux deux cibles, sans aucune adaptation sur eux.

**Générateur** : sur16325 éligibles (1102 absences conservées dans le bilan principal), les cinq CV admettent un seuil98% après conditionnement. Poids UNION figés :4230/4304, précision98,28%, couverture conditionnelle26,36% (4304/17427=24,70% si rapportée au banc initial). Risque conditionnel réappris :3856/3929,98,14%, couverture conditionnelle24,07% (22,55% du banc initial). Ces taux rapportés au banc initial restent eux-mêmes oracles, pas des couvertures produit. Les bornes Wilson test inférieures sont respectivement97,85% et97,67%, **pas98%**. Conditionner les données sur la présence rend le seuil admissible en calibration, sans prouver ce que donnerait un générateur modifié, ni suffire à valider98% sur test.

**Niveau de confiance et co-adresse** : la cible calibration97,5% donne2322/2369=98,02% test, mais borne97,37%, donc pas de certification98%. À97%, les1046 erreurs co-adresse historiques contiennent35SUR, dont1correct/34faux, soit34,34% des99fauxSUR pour0,78% des4459SUR. Hors masque :4359/4424=98,53%, Wilson[98,13;98,85]%, contre4360/4459=97,78% au total, écart descriptif+0,751point. Couvertures rapportées au banc entier :0,2008% dans/25,3859% hors (la table affiche les couvertures internes aux groupes). La définition du masque connaît les erreurs de l'ancienne base : cela ne prouve ni un effet causal général de la co-adresse ni une règle de rejet utilisable sans label.

**Conclusion** : présence des bons candidats et ambiguïtés d'une sous-famille contribuent au plafond observé ; le niveau SIREN seul ne lève pas la difficulté, et aucune politique demandée ne valide une borne test98% sur tout le banc. Le sous-groupe hors masque n'est pas une politique disponible. **Recommandation unique : conserver A_VERIFIER, sans promouvoir un SUR à98% sur ces jeux consommés.**

### SIREN98% par CV entreprise

| CV | SUR | Précision SIREN %, Wilson95 | Couverture % |
|---|---:|---|---:|
| 0 | 834 | 97.0024 [95.61231897627125, 97.96147016320643] | 24.4002 |
| 1 | 0 | indéfinie | 0.0000 |
| 2 | 0 | indéfinie | 0.0000 |
| 3 | 0 | indéfinie | 0.0000 |
| 4 | 0 | indéfinie | 0.0000 |

### SIREN99% par CV entreprise

| CV | SUR | Précision SIREN %, Wilson95 | Couverture % |
|---|---:|---|---:|
| 0 | 0 | indéfinie | 0.0000 |
| 1 | 0 | indéfinie | 0.0000 |
| 2 | 0 | indéfinie | 0.0000 |
| 3 | 0 | indéfinie | 0.0000 |
| 4 | 0 | indéfinie | 0.0000 |
