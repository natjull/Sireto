# V4.1 — Audit représentatif du CRM

Date : 27 juillet 2026  
Décision contractuelle : **`PIVOT_LABELS`**  
Décision de déploiement : **`STOP_DEPLOYMENT`**

## Conclusion directe

Oui : les modèles V4.1 ont été entraînés et évalués sur un ensemble beaucoup
plus facile que le CRM réel.

Ce n'est pas une intuition déduite de la baisse de couverture. Le tirage
aléatoire aveugle de 250 lignes du shadow le montre :

- seulement 91 cas ont un SIRET exact déterminable automatiquement ;
- 15 sont ambigus ;
- 144 restent non résolus ;
- la couverture mécanique des preuves n'est donc que 106/250 = **42,4 %**.

À l'inverse, le dev V4.1 venait majoritairement de cas pour lesquels un SIRET
actif unique était déjà démontré par le nom et la géographie. Il mesurait très
bien le comportement du système sur ces cas propres, mais pas sur un CRM
ordinaire : noms d'équipement, sociétés déménagées, adresses partagées,
anciens SIRET, groupes multi-sites et saisies incomplètes.

La V4.1 actuelle ne doit pas être déployée. Deux problèmes indépendants sont
maintenant démontrés :

1. le retrieval A perd de bons SIRET sur les cas sales ;
2. l'accepteur automatise certaines correspondances manifestement
   incompatibles quand une adresse commune masque un mauvais nom ou un mauvais
   type d'établissement.

Il ne faut pas entraîner immédiatement un nouveau XGBoost sur les anciens
labels. Le prochain cycle doit d'abord réparer le retrieval, puis constituer
des labels représentatifs difficiles.

## Protocole

L'audit a été préenregistré avant la lecture des preuves dans
`docs/v4_1_representative_audit_contract.md`.

L'échantillon figé contient 800 `SERVICE ID` distincts :

| Strate | Cas |
|---|---:|
| Tirage aléatoire dans les 19 025 lignes | 250 |
| Aucun candidat actif | 50 |
| AUTO proches du seuil | 100 |
| AUTO à score élevé | 150 |
| REVIEW proches du seuil | 150 |
| REVIEW à score faible | 100 |

Les preuves SIRENE ont été construites sans lire la décision, la confiance,
la prédiction ou le rang du modèle. La jointure avec les sorties V4.1 n'a été
faite qu'après publication des labels provisoires.

Les labels sont **provisoires et déterministes**, pas des validations
humaines. Ils peuvent localiser des défauts et invalider une revendication de
sécurité ; ils ne permettent pas de certifier une précision.

## Ce que les 800 cas révèlent

| Label provisoire | Nombre |
|---|---:|
| `MATCH_EXACT` | 242 |
| `AMBIGUOUS` | 16 |
| `UNRESOLVED` | 542 |

La forte proportion de non-résolus, 542/800, est le résultat central. Les
labels propres construits automatiquement sélectionnent par nature les cas
évidents. Ils ne peuvent pas, seuls, apprendre à un accepteur où s'abstenir
dans les scènes réellement difficiles.

Sur les 250 cas réellement aléatoires :

| Mesure | Résultat |
|---|---:|
| `MATCH_EXACT` provisoires | 91/250 |
| `AMBIGUOUS` provisoires | 15/250 |
| `UNRESOLVED` | 144/250 |
| Cas conclusifs | 106/250 = 42,4 % |
| Décisions AUTO V4.1 | 147/250 = 58,8 % |

Le système automatise donc davantage de cas que la méthode locale de preuve
n'arrive à en résoudre. Cela n'est pas automatiquement une erreur, mais
interdit de considérer le score de l'accepteur comme une preuve suffisante.

## Autopsie du retrieval

Parmi les 242 `MATCH_EXACT` provisoires, le bon SIRET est premier dans les
sorties shadow pour 237 cas et absent des dix candidats exportés pour cinq.
La reconstruction du pool gelé confirme que ces cinq vérités sont absentes du
Top-100 de la variante A :

| Retrieval | Vérités conservées | Recall@100 provisoire |
|---|---:|---:|
| A, actuellement retenu | 237/242 | 97,934 % |
| B, à code et budget constants | 240/242 | 99,174 % |
| C, à code et budget constants | 240/242 | 99,174 % |

Sur le seul tirage aléatoire, A obtient 90/91 = 98,901 % et B/C 91/91 =
100 % observé. Ces petits effectifs ne constituent pas une certification.

Les cinq pertes ne demandent ni dense retrieval ni nouveau modèle :

- trois SIRET sont récupérés aux rangs 1, 3 et 1 par B, qui exploite la preuve
  SIRET/SIREN déjà présente dans le CRM ;
- deux SIRET actifs sont trouvés par le sparse avant la dernière barrière,
  puis supprimés parce que le magasin global d'état ne les contient pas.

Ce magasin comporte 14 378 332 établissements, contre 42 322 035 lignes dans
le snapshot complet utilisé comme autorité. Le code traite actuellement
« absent du magasin rapide » comme « non actif ». Il faut au contraire
interroger un index d'état complet, épinglé au même snapshot.

Le détail reproductible des cinq cas est conservé dans
`reports/v9/v4_1_representative_retrieval_misses.csv`.

## Autopsie des décisions AUTO

Sur les 242 cas exacts provisoires :

- 225 sont envoyés en AUTO ;
- les 225 prédictions correspondent au SIRET exact provisoire ;
- 17 sont envoyés en REVIEW, dont cinq parce que la vérité manque au
  retrieval.

Ce résultat est bon, mais ne couvre que les cas faciles à prouver. Dans le
tirage aléatoire, 57 décisions AUTO restent `UNRESOLVED`. Une inspection
conservatrice a trouvé cinq contradictions nettes :

| CRM | SIRET proposé | Contradiction |
|---|---|---|
| Collège Saint Charles | Association de parents d'élèves | mauvais type d'entité |
| ECITON | KINOBE / TERRAGAIA | mauvais nom, adresse seule |
| Mairie de Merville-Franceville | École municipale | mauvais établissement |
| Welcoop Logistique | Eclatec Logistique | mauvais nom, lieu-dit seul |
| Médiathèque Jacques Prévert | Institut de beauté | mauvais type d'entité |

Même en supposant, de façon volontairement favorable, que les 142 autres AUTO
du tirage aléatoire sont tous corrects, ces cinq contradictions plafonnent la
précision à 142/147 = **96,60 %**. Ce nombre est une **borne supérieure
provisoire issue d'un audit IA**, pas une estimation certifiée de la précision.
Il suffit néanmoins à réfuter l'idée que le dev à 99,832 % démontrait la
sécurité du système sur le CRM.

Les cas et leurs preuves sont publiés dans
`reports/v9/v4_1_representative_auto_contradictions.csv`.

## Pourquoi nous avons tourné en rond

La boucle précédente optimisait correctement les modèles sur la mauvaise
question :

> « Parmi les cas où une règle stricte sait déjà désigner un SIRET actif
> unique, le modèle sait-il le retrouver et être confiant ? »

La vraie question est :

> « Sur le CRM tel qu'il est, quels cas peut-on automatiser sans confondre une
> entité avec une autre qui partage son adresse, son groupe ou sa commune ? »

Tant que les exemples d'entraînement difficiles ne possèdent pas de vérité
indépendante, modifier les features, le seuil ou XGBoost améliore surtout le
score sur les cas évidents. C'est bien un problème d'exécution scientifique
et de construction du benchmark, pas la preuve que l'architecture
retrieval/ranker/accepteur est mauvaise.

## Suite recommandée

L'ordre utile est maintenant le suivant :

1. construire un index complet `SIRET → état actuel` depuis le snapshot
   SIRENE autoritaire et supprimer le comportement « absent = fermé » ;
2. promouvoir provisoirement B et rejouer l'audit retrieval gelé, toujours
   avec 100 candidats maximum ;
3. faire valider humainement un premier lot de cas représentatifs difficiles,
   en priorité les AUTO non résolus, les adresses partagées et les cas
   multi-établissements ;
4. reconstruire les scènes d'entraînement out-of-fold avec ces vrais cas
   difficiles, y compris les erreurs de retrieval et de ranker ;
5. seulement alors réentraîner le ranker et l'accepteur, sans réutiliser le
   futur test ;
6. geler un seuil sur dev et certifier une fois sur un nouvel échantillon
   indépendant.

Les cinq contradictions suggèrent des signaux utiles — cohérence forte du nom,
type d'entité et danger d'une adresse seule — mais il serait incorrect de
coder cinq règles ad hoc sur ces cinq lignes. Elles doivent devenir une famille
de cas d'entraînement et d'évaluation.

## Verdict

**`PIVOT_LABELS`** : 542/800 cas restent non résolus et empêchent encore une
comparaison fiable de tous les étages sur le CRM réel.

Le premier correctif d'ingénierie est néanmoins précis et indépendant des
labels : **index d'état SIRENE complet + retrieval B**. Il doit précéder tout
nouvel entraînement.

La direction d'architecture n'est donc pas à jeter. En revanche, le chemin
historique « cas faciles → scores excellents → extrapolation au CRM » est
abandonné. La prochaine version sera jugée d'abord sur des cas représentatifs,
pas sur les seuls cas qu'une règle sait déjà résoudre.

## Artefacts

- échantillon aveugle :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative/e06cf0d79849aad4` ;
- preuves et labels provisoires :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_evidence/e696f22d68c0210f` ;
- jointure et synthèse :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_summary/2d18ef172f32aefc`.

## Provenance Git

- contrat et échantillon : `015d718`, `5f8ea00` ;
- preuves aveugles : `361c138`, `edf0858` ;
- synthèse après levée de l'aveuglement : `771be6b`.
