# Audit des 86 labels éloignés — échantillon du 15 octobre 2026

Protocole figé avant les cas : `38b2f267108c50ac0b47e378a3546b7acf6ba04c`.
**30 requêtes tirées sans remise parmi 86**, graine 20261015, ordre lexicographique puis `random.sample`.
Folds 2/3/4 exclusivement ; sources officielles locales, aucun web, retrieval ou label modifié.
Revue qualitative par agent des 30 dossiers locaux, sans validation humaine indépendante ni preuve du raccordement fibre. Les catégories respectent la priorité préenregistrée ; une différence déclarative ne prouve pas deux sites physiques.

## Catégories observées et extrapolations indicatives

| Catégorie | Observé /30 | % | Wilson 95 % | Attendu sur 86 (intervalle indicatif) | Part des 440 (intervalle indicatif) |
|---|---:|---:|---:|---:|---:|
| SIEGE | 7/30 | 23.3 | [11.8 ; 40.9] | 20.1 [10.1 ; 35.2] | 4.6 % [2.3 ; 8.0] |
| AUTRE_SITE | 4/30 | 13.3 | [5.3 ; 29.7] | 11.5 [4.6 ; 25.5] | 2.6 % [1.0 ; 5.8] |
| AUTRE_ENTREPRISE | 0/30 | 0.0 | [0.0 ; 11.4] | 0.0 [0.0 ; 9.8] | 0.0 % [0.0 ; 2.2] |
| ADRESSE_CRM_DOUTEUSE | 5/30 | 16.7 | [7.3 ; 33.6] | 14.3 [6.3 ; 28.9] | 3.3 % [1.4 ; 6.6] |
| COORD_IMPRECISES | 9/30 | 30.0 | [16.7 ; 47.9] | 25.8 [14.3 ; 41.2] | 5.9 % [3.3 ; 9.4] |
| INDETERMINE | 5/30 | 16.7 | [7.3 ; 33.6] | 14.3 [6.3 ; 28.9] | 3.3 % [1.4 ; 6.6] |

**Les valeurs sur 86 et 440 sont des extrapolations, pas des comptes observés.**
Sur les 440 mesurables, le nombre attendu est le même que sur les 86 : seule la strate éloignée est extrapolée.
Calcul : 86 × (compte/30), puis division par 440 pour la part ; jamais 440 × (compte/30).
Les 354 autres mesurables et les 67 non mesurables ne sont pas audités. Wilson brut sans correction de population finie,
sans pondération ; intervalles indicatifs, non simultanés, ne couvrant pas l’erreur de classement qualitatif.

## Lecture métier et limites

- **11/30** constats administratifs SIEGE ou AUTRE_SITE, pas autant d’erreurs humaines certifiées.
- SIEGE est prioritaire uniquement si une autre adresse textuelle est étayée : un siège à la même adresse n’est pas classé SIEGE.
- Une différence de numéro, une autre entrée ou une adresse postale peuvent expliquer un constat SIEGE sans autre site physique.
- **9/30** adresses concordantes ou quasi : l’écart XY ne suffit donc pas à accuser le label.
- Les quasi-concordances restent des hypothèses (numéro SIRENE absent, ancienne route, avenue/allée) : la règle stricte n’est pas assouplie.
- ADRESSE_CRM_DOUTEUSE signifie adresse insuffisante ou malformée, pas mauvaise cible humaine démontrée.
- INDETERMINE conserve les absences de preuve ; AUTRE_ENTREPRISE exige davantage que zéro concordance du SIREN.
- Sites et sièges : stock du 2026-08-01 ; points SIRENE du 2026-07-01 ; BAN CRM v17 existant.
- Point numéro/CFE n’est pas une entrée fibre certifiée ; sans contrôle terrain, aucune source de point fautive n’est identifiée.
- Ni chronologie du CRM, ni occupation à la date fibre : les labels fermés peuvent refléter une vérité historique.

## Même SIREN à l’adresse CRM selon la règle préexistante

- Scan exhaustif des **1837 établissements** des **29 SIREN** du tirage, pas seulement le top100.
- **12/30** avec concordance ; **12/30** avec concordance déjà dans le top100.
- **7/30** avec adresse stricte ; **5/30** uniquement par distance.
- **7/30** labels fermés ; **2/30** avec au moins une concordance fermée.
- « Oui » peut signifier proximité à 200/500 m, pas occupant exact : preuve et état sont détaillés dans le JSON.
- « Non » = aucune concordance dans ce stock selon cette règle, jamais absence réelle de l’entreprise.
- Rangs émis à base 1 ; A=actif, F=fermé, ?=état inconnu ; HT100=non émis. Tous les témoins sont dans le JSON.

| query_id | Catégorie | Même SIREN au CRM ? rang / état / preuve | Phrase (≤20 mots) |
|---|---|---|---|
| 5010 | AUTRE_SITE | Oui : 1/A/adresse | Mesuré : label fermé rue du Long Douet ; même entreprise active à l’adresse CRM, rang 1. |
| 4360 | INDETERMINE | Non ; rang/état sans objet | Inconnu : 14 contre 26 place Turenne ; aucun site concordant, ancien numéro ou déménagement non départagés. |
| NEWCRM:58e9008a310b509ce47da72b1aaffd1a | SIEGE | Oui : 16/A/distance | Mesuré : mairie siège avenue de la Muzelle ; établissement communal proche du CRM, par distance seulement. |
| NEWCRM:7d92ccf89904477aad39be1ab49ecdf9 | SIEGE | Oui : 2/A/distance, 20/A/distance | Mesuré : mairie siège avenue Edmond Grasset ; écoles communales proches du CRM, rangs 2 et 20. |
| 11185 | ADRESSE_CRM_DOUTEUSE | Non ; rang/état sans objet | Mesuré : voie CRM uniquement numérique ; siège YATOO plausible, position CRM non fiable pour conclure. |
| 12944 | COORD_IMPRECISES | Non ; rang/état sans objet | Mesuré : même rue et commune ; numéro absent du label, points distants de 3 913 mètres. |
| 1151 | COORD_IMPRECISES | Non ; rang/état sans objet | Supposé : route et ancienne route de Trouville désignent le même site ; géocodage CRM faible. |
| NEWCRM:de2cddee252d0afb4bc1883f8734802d | COORD_IMPRECISES | Non ; rang/état sans objet | Mesuré : même route de Bazus, numéro label absent ; coordonnées de rue éloignées malgré le nom officiel compatible. |
| 389 | SIEGE | Non ; rang/état sans objet | Mesuré : siège rue Pasteur contre place Espace Georges Brassens ; autre entrée du collège non exclue. |
| NEWCRM:9a36e53253268cef68169de0db776a9d | SIEGE | Oui : 1/A/adresse, 4/A/distance | Mesuré : label mairie siège ; école maternelle active exactement à l’adresse CRM, rang 1. |
| NEWCRM:3ac8233f7524db85a30659a5bd58194c | COORD_IMPRECISES | Oui : 1/A/adresse, 2/A/adresse | Mesuré : adresse strictement identique au CRM ; coordonnées éloignées de 287 mètres, sans preuve d’un autre site. |
| NEWCRM:2836cc0263fe9ef7540bed3a8a0d3266 | INDETERMINE | Non ; rang/état sans objet | Inconnu : label CPAM fermé à une autre adresse ; aucun établissement concordant ni occupant local identifié. |
| NEWCRM:ba537927a8ded41fa84d977502486fbe | INDETERMINE | Non ; rang/état sans objet | Inconnu : carrière au lieu-dit Le Danger, CRM rue du Vivier ; lien avec EUROVIA non établi localement. |
| NEWCRM:bead59a4ba32bffc4c244e2a144802ce | ADRESSE_CRM_DOUTEUSE | Non ; rang/état sans objet | Mesuré : numéro CRM absent, même avenue que le siège ; point de rue insuffisant pour identifier le site. |
| 5121 | AUTRE_SITE | Oui : 1/A/distance | Mesuré : label non-siège fermé avenue de Paris ; site actif route de Marcoing à 31 mètres, rang 1. |
| NEWCRM:803b3525a250be994e57f391f41b6b69 | COORD_IMPRECISES | Oui : 2/F/adresse | Mesuré : label fermé exactement à l’adresse CRM ; coordonnées divergentes, nom FOURNIER également porté par un autre SIREN. |
| 13566 | COORD_IMPRECISES | Oui : 1/A/adresse | Mesuré : numéro, voie et commune identiques ; écart de 2 528 mètres, sans différence d’adresse déclarée. |
| NEWCRM:b38f9f7fc040bc393760ce781998bcbd | SIEGE | Oui : 1/A/distance, 6/A/distance, 7/A/distance, 8/A/distance, 9/A/distance, 10/A/distance, 13/A/distance | Mesuré : mairie siège avenue Edmond Grasset ; école active rue Barbedette à 58 mètres, rang 1. |
| 3827 | COORD_IMPRECISES | Non ; rang/état sans objet | Mesuré : même allée et commune, numéro absent du label ; géocodage CRM faible, écart de 1 340 mètres. |
| NEWCRM:a9ea8e8f260c573f5c932ac03ddbf382 | SIEGE | Oui : 1/A/distance, 3/A/distance, HT100/F/distance | Mesuré : mairie siège rue de Fonton ; établissements communaux proches chemin Pavé, rangs 1 et 3. |
| 6573 | ADRESSE_CRM_DOUTEUSE | Non ; rang/état sans objet | Mesuré : CRM sans numéro au lieu-dit ; label laitier plausible dans la commune, site exact non localisé. |
| 13608 | AUTRE_SITE | Oui : 1/A/adresse | Mesuré : label CAMSP fermé rue d’Autriche ; CAMSP actif exactement à l’adresse CRM, rang 1. |
| NEWCRM:12c75e845505aed3a1caeda0084af4a6 | AUTRE_SITE | Oui : 1/A/adresse | Mesuré : label CARGLASS actif boulevard Mireille Lauze ; autre CARGLASS actif exactement au CRM, rang 1. |
| 9689 | SIEGE | Non ; rang/état sans objet | Mesuré : siège au 656 contre 666 dans le CRM ; erreur de numéro ou géocodage possible. |
| NEWCRM:6fcf2acdd8cb13aafa51156788396813 | INDETERMINE | Non ; rang/état sans objet | Inconnu : ZAC et parc du Beau Pré peuvent désigner le même lieu ; numérotation et point précis manquent. |
| 16478 | INDETERMINE | Non ; rang/état sans objet | Inconnu : 12 contre 77 rue Maurice Thorez ; aucun MONOPRIX concordant, erreur CRM ou autre site non départagés. |
| 1763 | ADRESSE_CRM_DOUTEUSE | Non ; rang/état sans objet | Mesuré : numéro CRM absent, même route que le label ; point de rue éloigné, site fibre inconnu. |
| 8637 | COORD_IMPRECISES | Non ; rang/état sans objet | Supposé : avenue contre allée de l’Europe, même numéro et commune ; coordonnées éloignées, type de voie incertain. |
| NEWCRM:fd76dea1d399b9fa7b3c56744b253e67 | COORD_IMPRECISES | Non ; rang/état sans objet | Mesuré : même impasse et commune, numéro label absent ; point SIRENE de rue à 602 mètres. |
| 15303 | ADRESSE_CRM_DOUTEUSE | Non ; rang/état sans objet | Mesuré : CRM et label sans numéro avenue Louis Renault ; point de rue ne localisant pas le raccordement. |

## Éléments manquants et nuances par cas

- `5010` — **Mesuré** : Label non-siège fermé au 8 rue du Long Douet ; 50014515600047 actif au 153 rue Géraldine Mock, concordance stricte et 0,03 m. **Supposé** : Ancien site sélectionné au lieu du site CRM courant. **Inconnu** : Date de collecte et occupation effective lors du raccordement.
- `4360` — **Mesuré** : Label non-siège fermé au 26 place Turenne, CRM au 14 ; aucun établissement du SIREN ne valide la règle. **Supposé** : Ancien numéro, déménagement ou coordonnées erronées possibles. **Inconnu** : Historique de numérotation et occupant du 14 au moment du CRM.
- `NEWCRM:58e9008a310b509ce47da72b1aaffd1a` — **Mesuré** : Siège actif au 48 avenue de la Muzelle ; CRM rue du Câble sans numéro ; établissement communal actif à 350 m, rang 16. **Supposé** : Le label pourrait viser la mairie plutôt que le site raccordé. **Inconnu** : Numéro CRM et identité de l’occupant exact ; une proximité à 350 m ne certifie pas le site.
- `NEWCRM:7d92ccf89904477aad39be1ab49ecdf9` — **Mesuré** : Siège avenue Edmond Grasset ; CRM 41 avenue Roger Salengro ; deux établissements scolaires actifs à 77 et 181 m. **Supposé** : Cible administrative mairie plutôt qu’un site communal. **Inconnu** : Occupant du 41 ; les deux concordances sont par distance seulement.
- `11185` — **Mesuré** : CRM 4 / 99 52 68 88 ; siège YATOO au 361 avenue des Romarins ; aucun site concordant. **Supposé** : Un numéro de téléphone paraît avoir remplacé la voie ; nom YATOO compatible avec le CRM. **Inconnu** : Adresse CRM véritable ; impossible d’établir qu’elle correspond à une autre adresse physique.
- `12944` — **Mesuré** : Même rue Pierre Adolphe Bobierre et commune ; CRM numéro 7, numéro absent SIRENE ; nom officiel identique. **Supposé** : Écart de géocodage ou positionnement imprécis plutôt qu’autre site ; aucune source de point fautive identifiée. **Inconnu** : Position exacte du site de la Géraudière et origine de l’écart XY.
- `1151` — **Mesuré** : Même commune ; CRM 23 route de Trouville, label sans numéro ancienne route de Trouville ; BAN faible, similarité 0,630. **Supposé** : Adresse quasi concordante ; le qualificatif « ancienne » peut aussi désigner une voie distincte, non établi localement. **Inconnu** : Tracé des deux voies, emplacement du manoir et point fautif ; classement supposé, pas preuve de label erroné.
- `NEWCRM:de2cddee252d0afb4bc1883f8734802d` — **Mesuré** : CRM 9 route de Bazus ; label même route et commune sans numéro ; dénomination usuelle officielle PAUL DANIEL ; qualité XY 12. **Supposé** : Position SIRENE imprécise dans la rue ; source fautive non démontrée. **Inconnu** : Numéro réel et occupation lors du raccordement.
- `389` — **Mesuré** : Label siège 37 B rue Pasteur ; CRM 1 place Espace Georges Brassens BP 20054 ; aucun site concordant. **Supposé** : Adresse administrative plutôt que l’adresse CRM, mais deux entrées du même ensemble restent possibles. **Inconnu** : Plan du collège, entrée fibre et adresse postale effective ; siège seul ne prouve pas une erreur humaine.
- `NEWCRM:9a36e53253268cef68169de0db776a9d` — **Mesuré** : Siège mairie 23 avenue Général de Gaulle ; école maternelle 21660048600023 active au 20 rue Dominique Mitjaville, concordance stricte. **Supposé** : Label de mairie au lieu de l’école explicitement nommée dans le CRM. **Inconnu** : Occupation effective au moment du raccordement ; autre concordance à 65 m non probante isolément.
- `NEWCRM:3ac8233f7524db85a30659a5bd58194c` — **Mesuré** : 2 rue Maurice Gouy, même INSEE ; label et autre école active à la même adresse, rangs 1 et 2. **Supposé** : Incohérence de points plutôt que différence d’adresse. **Inconnu** : Point fautif ; nom CRM différent de la dénomination officielle, identité de l’occupant fibre non certifiée.
- `NEWCRM:2836cc0263fe9ef7540bed3a8a0d3266` — **Mesuré** : Label non-siège fermé 1 avenue du Général de Gaulle ; CRM 10 avenue Georges Duhamel ; aucun site du SIREN concordant. **Supposé** : Ancienne agence ou mauvaise cible possibles ; CNAM et CPAM ne sont pas assimilées juridiquement. **Inconnu** : Identité et historique de l’agence Duhamel ; preuve d’une autre entreprise au site CRM absente.
- `NEWCRM:ba537927a8ded41fa84d977502486fbe` — **Mesuré** : Label non-siège actif CARRIERES KLEBER MOREAU au lieu-dit Le Danger ; CRM EUROVIA MANAGEMENT 1 rue du Vivier, aucun site concordant. **Supposé** : Entité de groupe, autre accès de carrière ou entreprise différente possibles. **Inconnu** : Périmètre de la carrière, liens officiels entre entités et occupant du CRM ; aucune absence réelle d’entreprise démontrée.
- `NEWCRM:bead59a4ba32bffc4c244e2a144802ce` — **Mesuré** : CRM avenue de l’Hippodrome sans numéro ; label EVERIAL au 1691 de la même avenue et commune ; écart 773 m. **Supposé** : Point CRM de rue éloigné d’un siège plausible pour le nom CRM. **Inconnu** : Numéro CRM réel et position du raccordement ; une autre adresse du siège n’est pas établie.
- `5121` — **Mesuré** : Label fermé au 14 avenue de Paris ; 82014321200026 actif au 21 route de Marcoing, CRM au 25, distance 30,53 m. **Supposé** : Ancien site retenu au lieu du site CRM courant. **Inconnu** : Exactitude du numéro 25 ; concordance du site actif par distance seulement.
- `NEWCRM:803b3525a250be994e57f391f41b6b69` — **Mesuré** : 350 rue des Clauwiers, même INSEE, label siège fermé SODIMOB ; FOURNIER RETAIL est aussi présent à l’adresse stricte. **Supposé** : Incohérence XY certaine entre sources, mais identité juridique du site fibre non tranchée. **Inconnu** : Entité raccordée et point fautif ; la concordance d’adresse du label interdit AUTRE_ENTREPRISE selon le protocole.
- `13566` — **Mesuré** : 15 avenue du Professeur Jean Rouxel, même INSEE ; nom CYRISEA compatible, siège actif, concordance stricte. **Supposé** : Un des points représente mal l’adresse textuelle. **Inconnu** : Point fautif et emplacement du raccordement.
- `NEWCRM:b38f9f7fc040bc393760ce781998bcbd` — **Mesuré** : Siège avenue Edmond Grasset ; CRM 3 rue Barbedette ; école active au 1 à 58 m ; six autres concordances par distance. **Supposé** : Mairie choisie plutôt qu’un site communal de proximité. **Inconnu** : Occupant précis du 3 ; sept concordances par distance ne prouvent pas sept occupants de cette adresse.
- `3827` — **Mesuré** : CRM 1 allée du Vieux Berger ; label même allée et commune sans numéro ; complément ZAC des Bornes du Temps. **Supposé** : Positionnement d’adresse imprécis ; aucune autre adresse du siège n’est établie. **Inconnu** : Point fautif et numéro exact du label.
- `NEWCRM:a9ea8e8f260c573f5c932ac03ddbf382` — **Mesuré** : Siège rue de Fonton ; CRM 38 chemin Pavé ; écoles actives aux 38 B et 34 à 6 et 41 m ; halte-garderie fermée hors top100 à 465 m. **Supposé** : Label mairie au lieu d’un site communal. **Inconnu** : Localisation des bureaux municipaux exacts ; écoles proches ne certifient pas l’occupant fibre.
- `6573` — **Mesuré** : CRM La Lande de Donville sans numéro ; label non-siège actif 1890 route Américaine, ZI du Foirail, même commune et nom compatible. **Supposé** : Lieu-dit ou adresse CRM insuffisante ; label plausible pour le nom mais pas site exact démontré. **Inconnu** : Lien spatial entre lieu-dit et usine, numéro et raccordement ; aucune concordance par règle.
- `13608` — **Mesuré** : Label non-siège fermé au 3 rue d’Autriche ; 26440270200239 actif au 11 route de Clisson, concordance stricte et même enseigne CAMSP NANTES. **Supposé** : Ancien site CAMSP retenu au lieu de l’établissement courant. **Inconnu** : Date du déménagement et de la collecte CRM.
- `NEWCRM:12c75e845505aed3a1caeda0084af4a6` — **Mesuré** : Label non-siège actif au 359 boulevard Mireille Lauze ; 42505055607368 actif au 111 traverse de la Montre, concordance stricte. **Supposé** : Autre établissement de la même entreprise visé par le label. **Inconnu** : Occupation et configuration des sites à la date du raccordement.
- `9689` — **Mesuré** : Siège actif au 656 avenue de l’Industrie ; CRM au 666 de la même avenue ; aucun site concordant, BAN faible. **Supposé** : La différence déclarative de numéro peut venir du CRM, sans réel choix d’un autre site. **Inconnu** : Numéro correct et emplacement réel ; SIEGE suit la priorité mais ne prouve pas une cible hors site.
- `NEWCRM:6fcf2acdd8cb13aafa51156788396813` — **Mesuré** : CRM 1 ZAC du Beau Pré ; label siège fermé PARC DU BEAU PRE LE VILLAGE 2800, nom usuel identique ; qualité XY 22. **Supposé** : Même zone ou autre adresse, non départagés ; siège ne suffit pas à prouver une autre adresse. **Inconnu** : Plan du parc, numéro, occupation et historique de pharmacie ; concordance textuelle seulement partielle.
- `16478` — **Mesuré** : Label non-siège actif au 77 ; CRM au 12 de la même rue ; aucun établissement du SIREN ne valide la règle ; autres enseignes au 12. **Supposé** : Erreur de numéro CRM ou autre magasin ; aucune autre entreprise cohérente avec MONOPRIX au 12 n’est démontrée. **Inconnu** : Numéro du magasin raccordé et occupation à la date CRM ; ne pas déduire AUTRE_ENTREPRISE des seules absences.
- `1763` — **Mesuré** : CRM et label route de Trouville, même commune, sans numéro ; nom officiel CARPEDIEM compatible ; distance 692 m. **Supposé** : Position CRM de rue insuffisante ; pas de différence d’adresse établie. **Inconnu** : Numéro, entrée du site Intermarché et point fautif.
- `8637` — **Mesuré** : CRM 2 avenue de l’Europe ; label 2 allée de l’Europe, zone Coriolis, même commune ; nom officiel COBO FRANCE identique. **Supposé** : Variation du type de voie ou deux voies distinctes ; quasi-concordance retenue sans assouplir la mesure stricte. **Inconnu** : Tracé et type exact de voie, position du bâtiment ; une autre adresse du siège n’est pas démontrée.
- `NEWCRM:fd76dea1d399b9fa7b3c56744b253e67` — **Mesuré** : CRM 1 impasse du Moulin ; label même impasse et commune sans numéro ; qualité XY 12, nom officiel compatible. **Supposé** : Positionnement de rue imprécis plutôt que label d’un autre site. **Inconnu** : Numéro exact du label, point fautif et position fibre.
- `15303` — **Mesuré** : Même avenue Louis Renault et commune, sans numéro dans les deux sources ; nom officiel DISTRIBUTION SERVICES IKEA FRANCE plausible. **Supposé** : Adresse CRM trop peu précise pour distinguer géocodage et cible exacte ; label plausible pour le nom. **Inconnu** : Entrée et position du site de la ZAC du Val Bréon ; aucun établissement du SIREN ne valide la règle.

## Preuve et reproductibilité

- Distances des labels et booléens site top100 historiques reproduits : **30/30 et 30/30**.
- Folds du tirage : `{"2": 7, "3": 6, "4": 17}` ; qualités XY des labels : `{"11": 26, "12": 2, "21": 1, "22": 1}`.
- Lecture filtrée avant matérialisation du ledger, sources grandes vérifiées taille/mtime/inode avant/après.
- Stock unités légales daté du même 2026-08-01 : seules dénominations/sigles officiels, aucun nom personnel ajouté.
- JSON : annotations qualitatives réutilisables, empreinte de revue, adresses et points probants, inventaire des SIRET scannés.
- Durée du scan final : **48.86 s** ; RSS gardée maximale **1274.8 Mio**, DuckDB 512 Mio/un thread.
- Reproduction réelle : `.venv/bin/python scripts/retrieval_far_label_audit.py` (réutilise les annotations du JSON, rescane les sources).
- Inspection sans publication : même commande avec `--inspect` (preuves temporaires sous `/tmp`).
- Le rapport d’agent antérieur demandé était vide ; aucune information n’en a été déduite.
