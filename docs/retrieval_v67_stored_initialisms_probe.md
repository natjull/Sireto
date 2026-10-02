# V67 — réparation de la lecture des alias, avant nouvelle mesure

V65 lisait legal_names_exact/trade_names_exact/site_names_exact via doc().
Ces champs sont INDEXÉS mais NON STOCKÉS. Les listes lues étaient donc vides,
et les4 nouvelles preuves toutes nulles. L'audit natif8 scènes n'a pas détecté
ce défaut car ses valeurs attendues ET natives étaient nulles sur ce petit
échantillon. **La conclusion V65 « sigles rejetés » est invalide.** Ne pas
présenter un test de parité all-zero comme preuve de couverture réelle.

Préenregistrement avant V67 : mêmes méthodes/hyperparamètres/splits/budgets
V65, mais lire les3 champs STOCKÉS legal_names/trade_names/site_names, que
le constructeur v3 copie dans les champs exacts indexés. Vérifier pour TOUT
positif ET négatif512 la cohérence du nom exact courant cache V56 vs alias
stocké avant fit ; STOP si un seul désaccord. Publier nombres d'alias et de
preuves de sigle. Offline : normalisation exacte d'alias comme le constructeur,
sans retirer des chiffres que le terme indexé contient encore. Query-only
signal() continue à masquer les SIREN/SIRET, aucun identifiant comme feature.

Pas de copie depuis V58 diagnostic choisi par vérité, aucun nouveau modèle
aval, vocabulaire/projection toujours STRICTEMENT split-local, poids de tête
non négatifs, choix internes seulement. Parité native8 scènes et preuve
exacte exhaustive ;80s/4Gio/0swap/CPU2, superviseur90s, pas de budget élargi.
V65 et V66 sont conservés immuables. V66 a STOP360s après120 sorties sans
labels ; ne pas les reprendre/évaluer partiellement. Sa recette HEAD_WORD
ne servait aucun sigle, mais le choix face aux alternatives V65 était mal
étayé. V67 doit rétablir cette comparaison avant toute qualité nouvelle.

Le présent test ne produit aucun candidat et ne mesure PAS le recall national.
S'il sélectionne une autre recette, nouveau protocole initial100 nécessaire.
Même recette n'autorise pas de transformer V66 partiel en128 complet.
Fold0 non lu/fold1 fermé, all-human/sidecar et objectif99,2% inchangés, pas
de READY_FOR_FOLD1. Graphe/BAN/XY/temporalité toujours pas ajoutés ici.
