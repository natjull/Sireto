# Gate retrieval V4 — résultats

## Verdict

**`GO_RANKER_V4`**

La politique de retrieval gelée conserve le bon SIRET actif :

| Périmètre | Succès | Recall@100 | Wilson 95 % |
|---|---:|---:|---:|
| Noyau V4 historique | 4 930 / 4 932 | 99,959 % | [99,852 % ; 99,989 %] |
| Ajout frais au fit | 819 / 819 | 100 % | [99,533 % ; 100 %] |
| Fit combiné | 5 749 / 5 751 | 99,965 % | [99,873 % ; 99,990 %] |
| Nouveau dev indépendant | 305 / 305 | 100 % | [98,756 % ; 100 %] |

Le plafond absolu reste 100 candidats. Aucune requête ne le dépasse, aucun
positif n’a été injecté, aucun SIREN exact n’est partagé entre fit et dev, et
ni le holdout scellé ni l’ancien test n’ont été lus.

Le résultat observé franchit le gate de 99,0 %. Avec seulement 305 exemples
dev, sa borne Wilson inférieure est 98,756 % : il s’agit donc d’une validation
pour poursuivre l’expérience, pas d’une garantie statistique de 99 % en
production.

## Ce que le calcul montre

Sur les 1 124 cas frais exacts :

- le SIRET de vérité est présent dans la partition géographique pour
  1 124/1 124 cas ;
- au moins un canal interne voit la vérité pour 1 124/1 124 cas ;
- la fusion gelée la conserve dans 100 candidats pour 1 124/1 124 cas ;
- le TF-IDF principal seul atteignait 811/819 sur le fit frais et 304/305 sur
  le dev ; les autres canaux ont récupéré les neuf cas manquants.

Cela confirme l’orientation simple retenue : le socle TF-IDF n’était pas à
abandonner. La fusion déterministe autour de ce socle suffit ici à atteindre
le gate retrieval, sans dense, sans GPU et sans location externe.

## Les deux misses du fit

Les seuls échecs appartiennent au noyau historique réutilisé :

- requête `6818`, Action Paris 75019, vérité `75330823808275` : visible dans
  les canaux internes mais éliminée par l’ancienne compression à 100 ;
- requête `8109`, MANGO SERRIS, vérité V4 active `83225768700018` : l’ancien
  label était `40325913800523`; la nouvelle vérité est visible dans les canaux
  internes mais l’ancienne liste top-100 ne la conserve pas.

Ils ne remettent pas en cause le gate. Un ranker ne peut toutefois pas
apprendre une requête dont le positif est absent : ces deux scènes devront
être exclues du fit candidat, ou reconstruites sous une règle préenregistrée
avant l’entraînement. Elles ne seront jamais corrigées par injection du
positif.

## Protocole et ressources

- Contrat préenregistré : commit `510868b`.
- Builders et tests : commit `e566c25`.
- Suite complète : 136 tests passants.
- Calcul des quatre audits de canaux : environ 9 minutes cumulées sur le Mac.
- GPU ou service payant : aucun.
- Données de calcul et artefacts lourds : SSD `/Volumes/CATNAT_DATA`.

Le champ `policy_status=EXPLORATORY_DEV_SELECTED_NOT_PROMOTABLE` présent dans
les sorties intermédiaires est un libellé historique codé en dur dans
l’évaluateur d’admission. La politique utilisée était bien gelée avant ce
nouveau dev ; le verdict contractuel fait foi. Le code n’a pas été modifié
après lecture du dev.

## Artefacts

- Entrée fraîche exacte :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_retrieval_inputs/801bdf2a1032116b/`
- Canaux fit :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_v4_fit_e566c25/`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_v4_fit_e566c25/`
- Canaux dev :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_v4_dev_e566c25/`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_v4_dev_e566c25/`
- Admissions finales :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_v4_fit_e566c25/`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_v4_dev_e566c25/`
- Bundle du gate :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/retrieval_v4/ddefe3daaacdf5ef/`

## Étape suivante autorisée

Construire le dataset candidat V4 pour le ranker :

1. 5 749 scènes fit exactes dont le positif est réellement présent ;
2. scènes `AMBIGUOUS` fit pour apprendre l’abstention, sans les transformer en
   faux positifs ;
3. 305 scènes dev exactes conservées intactes ;
4. entraînement du ranker avec séparation OOF par SIREN ;
5. mesure Hit@1 SIRET sur le nouveau dev avant tout accepteur.

Le holdout `holdout_sealed` reste fermé jusqu’au gel du ranker, de l’accepteur
et du seuil.
