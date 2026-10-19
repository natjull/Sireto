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
