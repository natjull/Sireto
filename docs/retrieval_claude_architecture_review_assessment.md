# Revue d'architecture — synthèse critique et orientation proposée

10 septembre2026. Mandat utilisateur : prise de hauteur sur l'architecture,
en partant du micro v23/v29, pas seulement une revue des six pertes.
Dossier initial `8532b17c`. Deux consultations Claude Opus5 via `claude -p`,
effort demandé `xhigh`, sans outils ni accès au disque, sans données CRM
brutes. La première reçoit contrats, rapport et code v21/v23/v28/v29;
la seconde reçoit nos objections, le dossier et le premier avis. Aucun
nouveau run retrieval, fit, changement de budget, index ou split.

Avis bruts conservés dans `retrieval_claude_architecture_review_raw.md` et
`retrieval_claude_architecture_review_followup_raw.md`. Ce sont des avis
externes, pas des contrats exécutables. Total indicatif CLI des deux appels
0,50989375 USD, auxiliaires Haiku inclus; pas une attestation de coût facturé
en supplément de l'abonnement. Aucun achat ou changement de facturation.

## 1. Ce qui est établi

Le dossier v29 vérifié reste inchangé :817 exacts contre822, quatre gains/neuf
pertes, contrôles512. Le micro lu après ce résultat donne six évictions dont
les preuves propres ne changent pas : score parent, RRF direct et rang site
identiques. Le rang relatif parent se dégrade et affecte le score final.

| Cas anonymisé | Rang parent v23→v29 | Rang site | Rang SIRET v23→v29 |
|---|---|---:|---|
|A|28→41|35|1618→2061|
|B|125→225|1|1639→2243|
|C|166→235|2|1700→2083|
|D|239→321|2|1813→2048|
|E|92→114|9|1844→2033|
|F|284→343|2|1961→2068|

Ceci s'explique par `retrieval_v23_joint_parent.make_master` : RRF direct
plus inverse de60+rang parent+rang site. Cinq de ces six cas n'ont aucun
RRF direct. Ce constat établit une dépendance du score à la composition du
pool, pas qu'un score absolu empêcherait les évictions : tout topK est
compétitif. Aucune liste de cas ou information label-side n'alimente le code.

Deux problèmes différents demeurent :

- couverture :136 vérités hors du réservoir v23, impossibles à récupérer
  en reclassant ce même réservoir;
- sélection :76 vérités présentes mais évincées avant2000 en v23. Un modèle
  aval opérant seulement sur2000 ne peut pas les récupérer.

Les canaux SIRET directs existent déjà. L'absence du SIREN dans l'union ne
prouve donc pas que l'ordre parent→site soit la cause du manque. Les preuves
officielles peuvent être absentes, trop faibles, mal combinées ou trop loin
dans une liste; les catégories de perte ne suffisent pas à départager cela.

## 2. Ce que la consultation a corrigé

Claude a explicitement retiré :

- la promesse d'expériences « additives » grâce à une échelle absolue;
- l'inexistence supposée du LambdaMART SIRET : les fonctions v12 existent,
  mais n'ont pas été activées pour publier500/100 sur v23/v29;
- « site-first » comme nouveauté en soi, puisque dix canaux directs existent;
- l'emploi du mode opérationnel initial comme état courant malgré le sidecar;
- un pré-score SIRET élargi qui contournerait le cap sous un autre nom;
- la prétendue absence d'information du delta−5 et sa dichotomie causale
  entre une mauvaise formule et une inférence hors distribution.

Réserves restantes sur le second avis : une absence dans les canaux testés
ne démontre toujours pas l'absence de tout signal; un échec d'une formule
n'invalide pas une classe entière de sélections; une réévaluation géographique
n'exige pas logiquement toujours un refit. Le test proposé reste incomplet :
il renvoie à une transformation/constante non spécifiée. Ce n'est **pas**
un protocole prêt à exécuter. Pas de nouvelle recette adoptée sur cette base.

## 3. Orientation proposée par Codex après revue

### Priorité architecture : combiner les preuves avant la coupe de recherche

Hypothèse : un site peut être moyennement ressemblant sur le nom et sur
l'adresse, donc absent des listes tronquées séparées, mais bon lorsque ces
indices sont combinés **dans la même recherche**, sur un document de site
compatible. La fusion après recherche ne peut sauver un candidat qui n'a
franchi aucune des coupes. Il faut tester cette hypothèse, pas la déclarer
responsable des132 absences.

Vérification locale complémentaire après Claude : dans
`src/xgb_matcher/hierarchical_retrieval.py`, `search` (~ligne710) utilise
`query.name` pour les canaux de noms et `query.address` pour l'adresse.
`_fielded_name_query` (~687) combine les champs de noms; `_combined_query`
(~556) ajoute les contraintes document/géographie, pas une concordance
souple de nom et de voie. Cette lecture porte sur ce backend précis,
pas sur une absence démontrée de toute tentative conjointe dans le dépôt.

Le premier test envisageable est donc une requête multi-champs nom+voie
souple sur les champs déjà indexés, sans numéro obligatoire et sans exiger
deux égalités exactes. Un champ texte concaténé unique, proposé par Claude,
n'est pas indispensable : préserver les champs évite de confondre un token
de nom avec un token de voie. Un pilote sur l'index existant peut précéder
une reconstruction site-période ou un audit RNE177Go exhaustif.

Règles de preuve à préserver : nom d'enseigne lié au bon site, noms légaux
du parent autorisés comme tels, pas de combinaison d'un nom de sibling avec
l'adresse d'un autre, pas de paire historique nom/adresse temporellement
incompatible fabriquée par concaténation. Une représentation site-période
ne devient pertinente que si les champs existants ne permettent pas cette
cohérence et qu'un gain potentiel est démontré.

### Deuxième axe : admission des références sous budget

Mesurer une sélection qui conserve les preuves et ne double-compte pas
plusieurs canaux corrélés comme des autorités indépendantes. Les scores
parent appris ne sont pas automatiquement calibrés; remplacer un rang par
une fonction arbitraire de score n'est pas une solution établie.

Une comparaison limitée à réservoir v23 fixe peut isoler l'effet d'une
recette d'admission. Elle devra avoir formule, éventuel apprentissage OOF,
coûts et métrique exacte2000 préenregistrés. Aucune protection rétroactive
des six cas, aucun oracle et aucun modèle SIRET au-delà2000. Cette piste ne
remplace pas le travail de génération des absences.

### Troisième axe conditionnel : planifier les recherches selon le contexte

Ne considérer une politique de recherches/admission adaptée à la qualité
nom/adresse/BAN que si les mesures montrent un bénéfice de routes différentes
et qu'une évaluation moins contaminée peut être construite. Elle ne doit
ni utiliser les statuts de qualification comme features, ni apprendre des
exceptions CRM, ni multiplier les appels au-delà des SLO. Pas de priorité
immédiate à un contrôleur complexe ou à la réouverture du dense historique.

## 4. Feuille de route proposée, non exécutée

1. **Expérience de preuve conjointe bornée**, contrat à écrire : vérifier le
   mécanisme sur l'index existant, recette unique, tous labels conservés,
   candidats naturels et mesure finale exacte sous2000 contre v23. Mesurer
   ensemble nouveautés de génération, gains/pertes d'admission, volumes et
   temps. L'interprétation doit pouvoir distinguer un nouveau candidat absent
   de l'ancien réservoir d'un simple reclassement. Pas de nouvel index national
   ou d'audit exhaustif comme prérequis automatique.
2. **Mesure500/100 et généralisation conditionnelles** : raccorder une
   référence Master stable au ranker SIRET v12 existant sous le bon protocole
   inner/outer OOF et après revue du coût. Il n'est pas nécessaire d'attendre
   une génération parfaite pour mesurer ces étages, mais leur recall reste
   borné par le Master. Définir une population non utilisée pour choisir les
   recettes et représentative du périmètre visé; le fold1 n'est pas un jeu
   disponible pour tuner. Une nouvelle tranche du même CRM n'est pas une
   preuve cross-CRM, et un hash d'échantillonnage n'efface pas une exposition
   antérieure. Aucun dataset ni split créé pendant la consultation.
3. **Selon les pertes restantes** : si une preuve locale existe mais est mal
   requêtée, investir dans requêtes/représentations; si elle manque réellement,
   instruire sources/temporalité et l'identifiabilité. Une conclusion de
   limite d'information exige davantage que l'échec de quelques canaux.
   L'exact et l'opérationnel restent séparés, tous labels au dénominateur.

Le résultat de la revue est une direction de recherche argumentée, pas la
garantie de l'objectif ni un accord d'exécution. Contraintes2000/500/100,
folds0/1 fermés et modèles aval gelés inchangés. Verdict scientifique actuel
`PIVOT_GENERATION`; v23 conservée. Aucun calcul retrieval en cours.
