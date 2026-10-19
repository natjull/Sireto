# Audit local des SUR UNION97 — protocole figé avant lecture des dossiers

## Grille A/B/C préenregistrée

Population : tous les faux SUR UNION de la politique risque97 publiée (attendus99/4459), folds population2/3/4 uniquement. Comparateur :100 vrais SUR UNION tirés sans remise par `random.Random(20261021).sample` dans les indices G0 croissants ; aucun choix selon les faits. Comparaison toujours au SIRET humain original, jamais au meilleur positif du pool.

Mesures symétriques, identiques pour faux et vrais :
- Nom N = maximum Jaro-Winkler entre nom CRM et noms publics officiels courants du site et de l'unité légale, normalisation historique `step2_baseline_v142` ; absence de noms d'un côté => C. Seuil de forte concordance0,92 ; écart significatif0,08.
- Adresse S = accord strict INSEE/numéro positif/suffixe/voie normalisée (normalisation historique). Le vainqueur doit avoir S vrai. Adresse adverse : soit S vrai également (égalité), soit contradiction connue sur numéro/suffixe/voie/INSEE tous renseignés, soit distance CRM-adversaire typée>500m contre CRM-vainqueur≤50m. Une absence de donnée seule ne prouve pas une contradiction. Suffixe vide est absence de suffixe déclaré, pas preuve d'un suffixe caché.
- Distances euclidiennes Lambert93 ; seuls points CRM et officiels de qualité numéro/voie sont admissibles. Distance absente/imprécise ≠0. Une distance typée CRM-vainqueur>500m bloque un verdict A/B même si l'adresse textuelle concorde.
- Un candidat domine si N≥0,92 ET : (adresse strictement meilleure selon la contradiction précédente ET N≥N adverse), OU (deux adresses strictes ET N−N adverse≥0,08). C'est une dominance mesurable nom/adresse, pas une preuve d'occupation. Pas de départage par score du modèle.
- **A** : seul le label domine ; erreur modèle fortement étayée par la concordance locale.
- **B** : seul le prédit domine ; conflit de cible compatible avec une autre entité/un autre site labellisé. Le rôle payeur/maison mère/voisin/ancien exploitant n'est jamais affirmé sans preuve locale explicite.
- **C** : tous les autres cas, dont conflit nom/adresse, homonymie, informations manquantes, candidats identiques et concordances ex aequo. Pour les vrais SUR exacts, C peut signifier simplement « aucune différence à départager », pas « occupation incertaine » ; publier ce sous-compte.

NAF, création, état au stock, siège/secondaire, date de début de période et vraie fermeture historique sont rapportés pour chaque candidat mais ne départagent pas A/B : date du CRM inconnue ; `dateDebut` n'est pas la date de fermeture. Même SIREN est un lien établi ; entre SIREN différents, co-adresse/nom/NAF ne prouvent ni filiation ni succession. Aucun web, aucun autre jeu, aucun label/pool/statut changé. L'objectif occupant fibre ne remplace pas la politique officielle même SIREN/même site.

## Preuve de préenregistrement

Ce protocole et sa copie JSON sont écrits avant lecture des faits individuels et des tableaux de prédictions. Leur SHA256 et l'heure UTC sont conservés dans le résultat final. Les deux documents et une ligne append-only de `handover.md` seront les seuls fichiers modifiés, dans un commit isolé. Code d'exécution reproductible conservé dans le JSON (aucun script ni log externe créé), Python sans bytecode, DuckDB1thread/512Mio et garde RSS3,75Gio.

## Résultats

**Audit exhaustif : 99/99 faux SUR ; témoin aléatoire : 100/4360 vrais SUR.** Aucun réglage de la grille après lecture. Les199 dossiers, deux fiches officielles par dossier, prédicats, noms/sources, numéros/voies, distances/qualités, NAF, création, fermeture historique et siège figurent dans le JSON.

| Population auditée | N | A : label domine | B : prédit domine | C : sans départage |
|---|---:|---:|---:|---:|
| Faux SUR UNION97 | 99 | 0 (0,00%) | 47 (47,47%) | 52 (52,53%) |
| Vrais SUR UNION97 tirés, graine20261021 | 100 | 0 (0,00%) | 2 (2,00%) | 98 (98,00%) |

**Attention au témoin** :96/100 vrais SUR ont prédit=label exact, donc C signifie ici « aucune différence à départager ». Parmi les quatre paires distinctes :2B et2C ; les deux B sont déjà corrects UNION (même SIREN, autre site humain). Le taux C brut98% n'est donc PAS un taux d'incertitude sur l'occupation et les2% de B ne sont pas un taux de labels faux. Il n'y a pas de classe A symétrique possible lorsque prédit et label sont identiques.

Par fold population2/3/4, faux : A0/0/0, B15/14/18, C19/19/14 ; vrais : A0/0/0, B0/0/2, C30/39/29. Les CV entreprise0–4 ne sont pas les folds population fermés0/1.

### Lecture des47 B, sans correction de vérité

- **27** : deux adresses strictes, nom prédit meilleur d'au moins0,08 ; **20** : adresse prédite strictement meilleure, nom fort et non inférieur. La grille admet explicitement une égalité sur l'un des deux axes, jamais une compensation d'une adresse moins bonne par un meilleur nom.
- **47/47** opposent des SIREN différents ;41 prédits sont actifs au stock,27 labels sont fermés. Ces nombres se recouvrent et ne démontrent pas une succession. Aucun lien inter-SIREN ni rôle payeur/maison mère n'est attesté par les tables utilisées.
- **A=0** signifie seulement qu'aucun label ne domine sous cette grille exigeante ; cela ne démontre pas l'absence d'erreurs modèle. Les52 C restent indécidables, certains prédits pouvant être erronés. Les47 B sont des **conflits de cible plausibles**, pas47 erreurs de label démontrées.
- Tous les398 accès de fiche trouvent un établissement au stock ; deux fiches sont non publiques : leurs champs protégés ne sont pas divulgués. Distances CRM typées :181/199 vers le prédit,178/199 vers le label ; ailleurs, aucun zéro imputé. L'occupation fibre, la date CRM et les liens capitalistiques ne sont pas observés.

### Trois exemples anonymisés

Pseudonymes de noms, voies et communes ; identifiants retirés. Les similarités ci-dessous sont calculées sur les noms originaux, **pas** sur les pseudonymes. Sélection reproductible : premier B par indice G0 avec deux adresses strictes, premier B avec adresse meilleure, premier C ; aucun exemple ne modifie la grille.

| Cas | Nom/adresse CRM anonymisés | Prédit | Label humain | Critères et verdict |
|---|---|---|---|---|
| E1 | « Hôtel Alpha »,1 rue R1 | « Hôtel de Alpha »,1 rue R1 ; actif/siège | « Exploitant individuel X »,1 rue R1 ; fermé/siège | N=0,960 contre0,531 ; deux adresses strictes ; distances24,17m/24,17m ; NAF55.10Z des deux côtés : **B**, occupation/succession non prouvée. |
| E2 | « Studio Bêta »,2 rue R2 | Enseigne « Studio Bêta »,2 rue R2 ; actif/siège | « Groupement Gamma »,rue R3 sans numéro ; fermé/secondaire | N=1,000 contre0,599 ; adresse stricte oui/non ; distances typées0,11m/2661,37m ; NAF70.21Z/70.22Z : **B**, lien juridique inconnu. |
| E3 | « Prestataire Delta »,10 rue R4 | « Prestataire Delta »,13 place P ; actif/siège | « Commune Delta »,place P sans numéro ; actif/siège | N=0,984 contre0,652 ; aucune adresse stricte ; distances typées100,72m/135,93m ; NAF62.03Z/84.11Z : **C**, bon nom ne compense pas la mauvaise adresse. |

### Précisions officielles et plafond diagnostique conditionnel

Sur les4459 SUR inchangés : **4360/4459 UNION =97,7798%** ; **4151/4459 exacts =93,0926%**. Les99 faux UNION restent tous des erreurs officielles. Couverture inchangée :4459/17427=25,5867%.

**Si les47 B comptaient comme corrects : (4360+47)/4459 =4407/4459 =98,8338%, soit +1,0540point.** C'est exclusivement le **plafond diagnostique conditionnel demandé**, pas une précision officielle, une garantie ni un maximum mathématique atteignable : les B ne sont pas confirmés, les C non résolus, les données datées et la grille conventionnelle. Ne pas extrapoler ce chiffre au banc entier ni déduire un taux de bruit certain. La politique opérationnelle officielle n'autorise pas une substitution inter-SIREN.

### Preuves et reproduction locale

Préenregistrement UTC `2026-10-19T20:58:11.452253+00:00`, avant lecture des cas ; SHA256 du document initial `3bae9a0648cc1525164263ffcbd2ddb42d9a42effd0474a49bf8818be04e17fd`, copie intégrale dans le JSON. Code d'exécution intégré, SHA256 `7f7be27c1f584780c6528387ad99f689b2a907ccd751a76201bf965a0e136f65`. Les seuils97 par CV sont relus, jamais recalibrés ; masques97 et top1 publiés reproduits.

Commandes réellement exécutées depuis la racine, sans fichier script/log supplémentaire :

```bash
AUDIT_MODE=smoke PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import json;exec(compile(json.load(open("docs/step2_sur_label_audit.json"))["execution_python"],"<sur-label-audit>","exec"))'
AUDIT_MODE=full PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import json;exec(compile(json.load(open("docs/step2_sur_label_audit.json"))["execution_python"],"<sur-label-audit>","exec"))'
```

Smoke réel10 dossiers :5faux C/5vrais C,8,56s, RSS0,463Gio. Complet199 dossiers :12,89s, RSS0,556Gio, garde3,75Gio ; les10 fiches smoke sont reproduites à l'identique. Le premier appel complet a dépassé le délai outil10s ; relance détachée, code/résultat/journal intégrés au JSON, aucun worker restant. Aucun fichier externe créé pour ce calcul court.

Vérification indépendante réelle : recomposition des199 verdicts à partir des faits bruts sauvegardés, mêmes comptes ;38 sources rehashées identiques avant/après ; bindings publiés CRM/labels/géométrie vérifiés et matrice exacte17427×100 égale à l'identité des SIRET humains originaux. Les sources sont détaillées avec SHA dans le JSON : masques `H/step2_risk_ceiling_v1_20261020/decisions.npz`, choix `H/step2_risk_model_v1_20261020_c/full/predictions.npz`, cohortes G0/optionD, stock SIRENE au2026-08-01, étude géographique au2026-07-01 et historique officiel des fermetures. DuckDB1thread/512Mio, aucun apprentissage, web, jeu fermé, VM, label/pool/statut modifié. Les sorties sont exclusivement ces deux documents et une ligne de journal ; commit auto-référencé par `git log -1 --format=%H -- docs/step2_sur_label_audit.json`.

**Conclusion : presque la moitié des faux SUR (47/99) présentent une meilleure concordance locale pour le prédit, mais cet audit ne prouve aucune erreur de label ni un plafond intrinsèque de précision ; les52 autres restent à départager.**
