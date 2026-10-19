# Gemma étape 2 — avenant budget 10 h, 19 octobre 2026

## Pré-déclaration avant lancement

Décision explicite du directeur scientifique, brief `sireto-finetune-2` : le
STOP4h `ab2a4e57` est conservé. Nouvelle expérience et namespace, aucun résultat
historique modifié. Contrat scientifique initial `docs/step2_gemma_finetune.md`
maintenu sauf changements explicites ci-dessous. Aucune promotion production.

### Population, textes, négatifs

Même split entreprise graine20261019 du banc17427/11483, folds2/3/4, exclusions
pilote405 entreprises/test4323/control962/folds0/1 héritées et re-vérifiées.
Évaluation initiale3405 dossiers/2297 entreprises ; apprentissage14022/9186,
dont proper11224/7348 et calibration2798/1838. Gradient proper seulement,
calibration et pilote jamais dans le gradient. Confirmation descriptive526,
content externe + empilement/confiance groupée comme le code précédent.

Réutiliser uniquement les textes/métadonnées admissibles scellés du précédent
prepare (et les vecteurs BASE non adaptés), jamais les anciens poids adaptés,
ni ceux du mini-run. Identifiants réservés aux jointures/groupes/exclusions.
La vérité humaine exacte et les positifs union restent immuables. Candidat
exact absent/texte manquant/sans négatif sûr : compte explicite sans gradient,
mais toutes les lignes choisies demeurent dans l'évaluation end-to-end.

Prendre au plus les15 PREMIERS négatifs admissibles par requête, déjà ordonnés
même adresse puis même SIREN puis rang G0, après masque union/texte identique
au positif/entreprises calibration-évaluation-pilote. Utiliser ces15 (ou moins)
dans chaque InfoNCE ; pas99, pas de minage web ni inter-requête. Le masque de
la perte ne laisse que le positif et les négatifs propres à l'ancre. Les100
candidats restent nécessaires au ranking `85+S1+S2` : ne pas prétendre que
limiter les négatifs réduit automatiquement tous les encodages du ranker.

### Une époque et reprise

Configuration fixée `config/step2_gemma_finetune_10h.json` : LoRA NEUF r8/alpha16
Q/V/projection, baseBF16 gelée, mean-pooling masqué/L2, prompts et textes du
code colocated inchangés, batch16/micro2 par gradient cache V2, InfoNCE.05,
AdamW3e-5/decay.01/clip1, warmup3%, cosine vers10%, UNE époque maximum.
Aucune HPO, aucune sélection de checkpoint sur l'évaluation. Checkpoint
atomique toutes1800s au plus (plus premières étapes, fin et yield propre),
optimiseur/cursor/ordre/liaison SHA conservés. Arrêt budget avant fin d'époque
= incomplet, jamais convergence revendiquée. Encodages par shards1024 hashés,
reprise sans recalculer les shards achevés ; erreurs scientifiques fatales.

### Mini-run mesuré, projection et réductions avant fit complet

Le superviseur détaché commence par un nouveau mini-run réel de60 entreprises
sélectionnées par le même hash sans sélection d'erreurs,2pas de gradient15
négatifs, encodages/ranking/SUR et36 pilotes descriptifs. Ce mini compte dans
le budget10h, ses poids sont exclus du complet. Les scores du mini ne règlent
aucun paramètre ni le périmètre ; seuls débit, durées et comptes sont lus.

Projection complète AVANT entraînement :
- BASE : durée mini/new_texts × nombre exact de textes manquants ;
- adapté : durée mini/all_texts × nombre exact de textes nécessaires ;
- fit1epoch : secondes de gradient mini/pas × ceil(paires admissibles/16) ;
- CPU ranking : max(secondes/ligne de fit des deux rankers mini) × deux recettes
  × (lignes train + proper), avec réserve plancher600s pour NB/pilote/audit ;
- CPU sémantique : durée mini × nombre total de dossiers/mini-dossiers ;
- multiplier cette somme par1,20 puis ajouter600s fixes ; budget restant après
  mini/préparation, plafond absolu36000s total. Projection conditionnelle,
  jamais annoncée comme durée complète mesurée.

Si projection dépasse le reliquat : entreprises évaluation ordonnées par hash
`20261019|10h-evaluation|SIREN`, préfixes entiers des fractions1/.75/.5/.25/.125 ;
conserver TOUTES les lignes de chaque entreprise. Si cela ne suffit pas, garder
évaluation.125 et réduire proper par hash `20261019|10h-training|SIREN` et mêmes
fractions. Calibration conservée, jamais entraînée par Gemma. Premier périmètre
qui tient retenu, jamais de choix par succès/label de l'évaluation ; tous les
essais de coût/comptes/exclusions de budget publiés. Si aucun ne tient : STOP.
Le split initial reste scellé ; une réduction est un échantillonnage annoncé,
pas une nouvelle partition ni un changement des labels. BASE/adapté voient
EXACTEMENT les mêmes lignes/volumes/rôles. Aucun dataset réduit silencieux.

### Mesures et ressources

Recette ranking/S1 crossfit/SUR marge et bootstrap entreprise10000 inchangés,
masque famille co-adresse `c37a1a7b`, top1 exact/union et Wilson95, delta apparié
et IC95 entreprise, gains/pertes, précision/couverture SUR avec seuil interne
union>=98% sans garantie test. Anti-par-cœur train in-sample vs évaluation et
cosine proper vs évaluation ; confirmation526 déjà consommés descriptive.
Même règle de recommandation : audit indépendant hors production seulement
si borne basse IC95 du delta union externe>0 ; sinon conserver S2-BASE.

Calcul TOTAL CPU+Metal+encodages+évaluation+mini+reprises <=10h, majoré par le
temps mur des processus (I/O inclus) ; seules attentes documentées de ressources
sont exclues. Superviseur et workers strictement séquentiels, plafond propre
7,5Gio RSS+driver sous8Gio, MPS.2. hydraulic jamais contrôlée : attendre/yield
du seul worker détenu si VM active/inconnue ET free+inactive<10Gio. Aucun web,
cloud, installation, jeu fermé, label/pool modifié, test unitaire ni push.

Namespace neuf `H/step2_gemma_finetune_10h_20261019/{mini,full}`. Sources, code,
config, protocole et sélection scellés. Audit des masques15/groupes, modèles
sérialisés/rejoués et bootstrap avant publication. Reprises resource-only au
plus12, mémoire plafonnée déterministe => STOP plutôt que répétition inutile.
Publication automatique isolée `docs/step2_gemma_finetune_10h.{json,md}` et
`handover.md`, rapport français<=30lignes, y compris STOP avec raison exacte.
Une seule attente bloquante de l'agent, jusqu'à la fin du superviseur détaché.

## Résultats

## Fait
Statut COMPLETE_DEVELOPMENT_NOT_PRODUCTION ; aucune promotion, labels et candidats inchangés.
Durée réelle calcul total : 6.024 h (mini inclus), Metal 5.862 h ; mur 6.050 h.
Pic worker RSS+driver 4.91 Gio (<8) ; attente ressources 1.5 min.
## Preuve
Rôles retenus : {'evaluation': {'queries': 3405, 'companies': 2297}, 'proper': {'queries': 11224, 'companies': 7348}, 'calibration': {'queries': 2798, 'companies': 1838}, 'training': {'queries': 14022, 'companies': 9186}}.
Gradient : 638 pas, 1époque, 10205/10205 paires exposées, jusqu’à15 négatifs sûrs.
Projection préalable : 7.88 h ; réduction de budget : False.
base exact 2449/3405=71.92% ; UNION 2691/3405=79.03%.
base SUR : précision UNION 732/752=97.34% ; couverture 752/3405=22.09% (cible interne98%, non garantie).
base écart train−évaluation UNION -1.60 points (train in-sample).
base526 descriptifs : content exact/UNION 386/408 ; stack 388/416.
adapted exact 2458/3405=72.19% ; UNION 2702/3405=79.35%.
adapted SUR : précision UNION 298/302=98.68% ; couverture 302/3405=8.87% (cible interne98%, non garantie).
adapted écart train−évaluation UNION -0.58 points (train in-sample).
adapted526 descriptifs : content exact/UNION 382/406 ; stack 391/420.
Δ UNION adapté−BASE +0.323 points, IC95 entreprises [-0.5675072257507658, 1.2433009521977485].
Δ exact +0.264 points, IC95 entreprises [-0.6065935576405904, 1.1884710114669952].
Même adresse : 194 cas ; Δ UNION/IC95 {'delta_pp': 8.762886597938147, 'paired_company_bootstrap95_pp': [2.5641025641025643, 15.228426395939087], 'simultaneous95_pp': [1.5957446808510638, 16.25630494732129], 'gains': 26, 'losses': 9}.
## Écarts / choix
Split entreprise initial80/20 conservé ; calibration16% totale hors gradient, évaluation et pilote hors gradient.
Négatifs15 seulement pour InfoNCE ; le ranking requiert toujours ses100 candidats. Dernier checkpoint, aucun réglage par score.
## Reste / risques
Amont déjà consommé/non full-OOF ; train in-sample optimiste ;526 descriptifs, aucune validation production.
Recommandation unique : Conserver S2-BASE, sans promotion de cet adaptateur.
