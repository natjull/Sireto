# Étape 2 — choisir un SIRET parmi les 100

## Décision Nathan du 15 octobre 2026

Autorisation : **« Vas-y. A. »**. Accepter le plafond actuel de recherche
(ordre de grandeur discuté ~97–98 %, mesure exacte V142 **507/526 = 96,39 %**)
et passer à « choisir 1 parmi 100 » avec un statut de confiance.
L'objectif initial99,2 % est **suspendu, non supprimé et non atteint**.
Cette décision prime sur le gel aval des directives précédentes uniquement
pour cette étape 2 ; elle ne vaut ni certification ni promotion production.

## Référence gelée et sortie

- Étape 1 : V142 `bano_native_20261013_night/all_new_sources_max_bano`,
  `outputs.json["learned_native_ooc"]`, listes ordonnées de 100 candidats.
- SHA256 outputs : `fb691a70311392c103955e1b4fe71ee03fbdabbb2dcc87d2fb32e9bf6c5d3218`.
- L'essai cible site V173 (503/526, commit `85a66d60`) est rejeté ; V142 reste
  la référence. Aucune régénération, expansion ou substitution du pool.
- Produit cible : **un SIRET issu de ces 100 + `SUR` ou `A_VERIFIER`**.
  `SUR` signifie une politique d'acceptation mesurée, pas une garantie absolue.
- Ranker, decider, modèle de risque et accepteur dégelés **étape 2 seulement**.
  Le présent jalon autorise inventaire et baseline, **aucun apprentissage**.

## Mesures obligatoires

Publier les comptes n/N, taux et Wilson bilatéral95 brut, au total et par
fold2/3/4 : précision des `SUR` exacte et opérationnelle, taux de `SUR`,
précision top1 globale exacte et opérationnelle. Si aucun `SUR`, sa précision
et son intervalle sont non définis, jamais 100 % par convention.
Publier aussi top1 SIREN, site opérationnel et union exact ∪ site, recall100,
rangs exacts 1 / 2–3 / 4–10 / 11–30 / 31–100 / absent.
Les 19 absences exactes restent des erreurs end-to-end, sans exclusion.
Une baseline sans accepteur ne mesure pas la précision des `SUR`.

Deux vues opérationnelles restent distinctes : politique historique
`docs/siret_operational_equivalence_policy.md` (exact ou même SIREN/même site
attesté par le sidecar), et **site opérationnel** du contrat `fb604665` :
même SIREN ET (adresse CRM concordante OU distance admissible), seuil200 m si
les deux positions sont précises au numéro, sinon500 m si précision de voie ;
positions communales/inconnues/absentes exclues de la preuve distance.
L'exact n'est pas automatiquement un succès site ; union = exact OU site.
Aucune de ces vues ne réécrit la vérité humaine ni ne certifie l'occupation fibre.

## Protections inchangées

- Aucun label modifié, positif injecté, cas difficile retiré ou règle client.
- Développement uniquement folds2/3/4 ; fold0, fold1, test4323 et control962
  fermés. Les anciens modèles ne prouvent pas une absence de contamination :
  vérifier leur provenance et les groupes d'entreprise avant réutilisation.
- Aucun identifiant SIREN/SIRET, dérivé d'identifiant CRM ou alias CRM comme
  feature ; identifiants réservés aux jointures/groupes.
- Features calculables sans vérité ; sidecar label-side scellé, évaluation seule.
- Aucun web, dépense externe ou GPU loué ; Mac/CATNAT seulement.
- Contrats/résultats historiques, sources, sorties et travaux étrangers immuables.
- Apprentissage/calibration futurs hors groupes, seuil choisi sans fold évalué ;
  pas de seuil optimisé sur les 526 puis présenté comme validé sur ces mêmes526.
- Pilote consommé, encodeur non entièrement OOF et affectation V142 fondée sur
  le SIREN humain : chiffres de développement, pas validation indépendante
  ni routage déployable. Aucun jeu fermé ouvert par cette décision.
- Jalons isolés commités et cités dans `handover.md` ; aucun push.

Livrables du jalon : inventaire `docs/step2_downstream_inventory.md`, script
`scripts/step2_baseline_v142.py`, résultats `docs/step2_baseline_v142.{json,md}`.
