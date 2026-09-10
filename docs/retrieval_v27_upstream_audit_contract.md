# V27 — attribution géographique et import des noms

Préenregistrement du 10 septembre 2026. Audits seulement, sur les 1034
requêtes scellées v26, folds 2/3/4, contrôles inclus. Aucun nouveau canal,
modèle, label, seuil de retrieval ou candidat injecté. Plafonds produit
2000/500/100 inchangés, ranker/decider/risk/accepteur gelés, folds 0/1 fermés.

## Géographie

Rejouer la fonction v16 et ses sources scellées sans la modifier, avec les
seuls champs CRM pour résoudre BAN et générer les voisins. Vérifier la
concordance avec les artefacts v16. Jointure truth-side ensuite : présence
du parent, coordonnées locales disponibles, entrée dans les 4096 sites,
rang parent puis cap512. Publier une cause exclusive par requête : présent,
BAN non résolue, aucun site parent valide, frontière de commune, troncature
4096 sites, troncature512 parents. Une contradiction résiduelle impose STOP.

Le zoom annoncé couvre les 12 absents v26 dont un site est à ≤500 m, mais
toutes les 1034 requêtes et tous les segments sont conservés. Aucun rayon
optimisé. Vérifier coordonnées/sources et reproduire les distances v26; aucune
preuve de même bâtiment ni relabellisation par proximité. Ne pas persister
de X/Y brut. Les sorties de replay sont des copies diagnostiques de l'ancien
canal, pas des candidats ajoutés à un pool. Le regroupement SIREN historique
v16 par préfixe est seulement rejoué puis recroisé avec le SIREN officiel.

## Import des noms

Comparer les champs SIRENE bruts actuels et historiques autorisés au dossier
name_evidence puis au portfolio, au niveau du parent et de tous ses sites.
Allowlist UL : denominationUniteLegale, denominationUsuelle1/2/3UniteLegale,
sigleUniteLegale (si présent). Sites : enseigne1/2/3Etablissement,
denominationUsuelleEtablissement. Exclure nomUniteLegale, nomUsageUniteLegale,
prénoms et représentants de toute similarité/feature. Historique non branché
dans les inputs du dossier ≠ perte silencieuse à l'import actuel.

Mesurer présence des valeurs et correspondances CRM exactes / tokens sans
seuil optimisé; absences de source conservées. Les correspondances à des
valeurs absentes du portfolio sont un plafond label-side, jamais des rescues.
Comparer RNE canonique → dossier → portfolio par valeurs et identité
officielle : contrôle structurel seulement, sans similarité CRM pour ces
champs dont le rôle ne suffit pas à garantir l'absence de noms de personnes.
Pas de réexport de leurs valeurs. Comparaisons normalisées, provenance et
type de sujet séparés; couverture globale du portfolio ne prouve pas une
présence sur chaque site.

Limite explicite : le ZIP RNE contient 177 Go décompressés en 5470 membres,
sans index membre/SIREN retrouvé. Pas de scan intégral déclenché dans cet
audit court. L'extraction brut RNE → canonique reste non vérifiée sur cette
population; inspection du code non équivalente à une mesure exhaustive.

## Exécution et publication

Préflight ≤16, puis commit du code/config avant mesure complète. Sources et
sorties hashées, budget par audit 30 minutes/RSS14 Gio/zéro swapout, DuckDB
4 Go et temporaires externes. Mac/CATNAT, aucun coût externe. Publier ensemble
exact/opérationnel antérieurs, historique/V2/V3 et segments. Aucun nouveau
Recall ni gate; panel réutilisé, aucune revendication nationale/cross-CRM.
Verdict PIVOT_GENERATION si intégrité satisfaite, STOP sinon. Consigner les
résultats et les commits dans handover.md. Une éventuelle correction du canal
fera l'objet d'un préenregistrement distinct; elle n'est pas automatique.
