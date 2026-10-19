# Familles d'erreurs G0 — protocole figé avant comptage

Source : banc `9a8ecdd6`, base85, prédictions OOF5 plis entreprise existantes,
17427 dossiers folds population2/3/4. Aucun fit, retrieval ou label corrigé.
L'absence de différence entre recettes ne prouve pas un plafond d'information :
ce diagnostic décrit des conflits observables, pas leurs causes certaines.

## Dénominateurs et référence

Erreurs étudiées = UNION présente parmi100 ET choix top1 OOF non UNION.
Les absences UNION restent séparées et dans les erreurs end-to-end totales.
La référence des familles est TOUJOURS le SIRET humain original, hydraté à
part même s'il est absent ; aucune injection dans les100. Le meilleur candidat
UNION selon le score OOF est aussi conservé comme oracle de reclassement.
Il peut différer du label exact ; les points du label ne deviennent pas features.

## Règles exclusives et ordonnées, premier prédicat vrai

1. Même SIREN, siège choisi/non-siège label (booléens officiels connus).
2. Même SIREN, états opposés A/F (deux sens distingués dans JSON).
3. Même SIREN, autre site proche : points tous deux numéro/voie et distance
   choix–label≤500m. C'est une proximité diagnostique, pas une preuve de même site.
4. Même SIREN, autre établissement restant (dont distances inconnues).
5. Autre SIREN, adresse stricte choix–label identique : INSEE, numéro positif,
   suffixe, voie normalisée ; « co-locataires » est un proxy, pas une occupation prouvée.
6. Autre SIREN, homonyme/même enseigne autre lieu : Jaro-Winkler nom CRM vers
   noms publics officiels≥0,92 pour choix ET label, ET communes officielles
   connues différentes OU distance choix–label typée>500m. Noms officiels construits
   comme V1, aucun alias CRM ; rapprochement lexical, pas identité juridique.
7. Label loin : distance CRM–label>500m, les deux points de précision numéro/voie.
8. Adresse CRM sans numéro positif (même parseur que le contrat site).
9. Nom générique : au moins un token de MAIRIE/COMMUNE/ECOLE/COLLEGE/LYCEE/
   AGENCE/CENTRE/SERVICE/SYNDICAT/ASSOCIATION/HOTEL, et tous ses tokens sont dans
   ce vocabulaire ou DE/DU/DES/LA/LE/LES/L/D/ET/EN/A/AU/AUX/SAINT/SAINTE/
   MUNICIPAL/MUNICIPALE/PUBLIC/PUBLIQUE/PRIMAIRE/MATERNELLE/NATIONALE/SAS/SARL/SA.
10. Autre.

L'ordre ci-dessus ne changera pas selon les effectifs. Les drapeaux non exclusifs
sont aussi comptés pour exposer les recouvrements. Distance absente/inconnue ≠0 ;
les distances finies mais trop imprécises sont décrites à part, jamais « loin »
par cette règle. Les booléens manquants ne valent pas faux connu.

## Incertitude et gains

Pour chaque famille : n, n/N_erreurs_récupérables, Wilson95 descriptif ; également
part du banc et plafond de gain UNION =100×n/17427 points, IC obtenu sur n/17427.
Le gain maximal exact compte seulement les labels exacts déjà présents dans100.
Ce sont des oracles conditionnels, pas des gains attendus ; pas de correction
causale ni d'inférence que tel signal résoudra tous les cas. Par pli et métriques
du banc conservés. Échantillon20 : parcours des familles dans l'ordre ci-dessus,
erreurs ordonnées SHA256(`20261019|family-example|query_id`), tour à tour jusqu'à20.

## Signaux et Gemma, calcul borné

Comparer les signaux aux85 features, puis vérifier les colonnes réellement locales
SIRENE (NAF, effectifs, dates), sans créer de nouvelles features ou superviser un modèle.
Inventorier les SIRET G0 uniques et les banques SITE officielles512, bindings et
tenseurs existants. Couverture potentielle de clé séparée de couverture validée
texte–ligne–encodeur ; les vecteurs parent/noms seuls ne remplacent pas un site.
Fiches CRM : distinguer noms seuls et nom+adresse, et le périmètre d'inventaire.
Un benchmark Metal `$H/runtime/bin/python`, modèle/adapter locaux inchangés,
1000 textes réels (500 CRM+500 vues SITE G0 déjà locales), pas de gradient,
chargement et encodage séparés, longueurs et pic mémoire RSS+driver MPS publiés.
Aucune installation/web/VM ni encodage massif ; extrapolations conditionnelles,
nombre de vues manquantes inconnu. Si4Gio ne peuvent être respectés, STOP explicite.

Namespace propre `H/step2_error_families_g0_20261019/`, source/script/protocole
hashés ; DuckDB512Mio/1thread, workers séquentiels, garde3,75Gio (<4), attente
worker V142/descendant>4Gio avant phase lourde, aucun signal à l'autre agent.
Vrai smoke puis lecture complète ; aucun test unitaire, aucun jeu fermé ni v174
lu/écrit, travaux étrangers et pending préservés, commits isolés sans push.

## Résultats

### Comptage exhaustif et plafonds conditionnels

**17427 dossiers,11824 exacts et12889 UNION en top1 OOF base ;4538 erreurs UNION (26,04%).**
**3436 erreurs récupérables** : UNION présente dans100 mais mauvais top1 ; **1102 absences** restent des pertes de génération, sans exclusion du banc.
Plafond UNION du pool :16325/17427=93.68% ; résoudre toutes les3436 erreurs ferait au maximum **+19.72pt**, pas26,04pt.
Référence humaine exacte absente pour190 de ces erreurs : un autre site admissible est dans100. La catégorie ne compare jamais au positif le plus avantageux pour sa définition.

| Famille exclusive, priorité figée | n | % des3436 [Wilson95] | Gain UNION max, pt [Wilson95] | Gain exact max, n |
|---|---:|---|---|---:|
| Même SIREN : siège choisi / label site | 267 | 7.77 [6.92–8.71] | +1.532 [1.36–1.73] | 267 |
| Même SIREN : fermé ↔ actif | 41 | 1.19 [0.88–1.61] | +0.235 [0.17–0.32] | 41 |
| Même SIREN : autre site proche | 37 | 1.08 [0.78–1.48] | +0.212 [0.15–0.29] | 36 |
| Même SIREN : autre établissement restant | 42 | 1.22 [0.91–1.65] | +0.241 [0.18–0.33] | 41 |
| Autre SIREN : même adresse stricte | 1046 | 30.44 [28.93–32.00] | +6.002 [5.66–6.36] | 1038 |
| Homonyme / enseigne, autre lieu | 23 | 0.67 [0.45–1.00] | +0.132 [0.09–0.20] | 23 |
| Label à plus de500m du CRM | 463 | 13.47 [12.37–14.66] | +2.657 [2.43–2.91] | 358 |
| Numéro CRM structuré non exploitable | 443 | 12.89 [11.81–14.06] | +2.542 [2.32–2.79] | 443 |
| Nom générique strict | 0 | 0.00 [0.00–0.11] | +0.000 [0.00–0.02] | 0 |
| Autre | 1074 | 31.26 [29.73–32.83] | +6.163 [5.82–6.53] | 999 |

Les gains sont des oracles sans régression sur les autres lignes, pas des effets mesurés de features. Wilson brut par requête, descriptif : dépendance entreprise non corrigée ; effectifs par CV et entreprises dans JSON. Les dix classes couvrent exactement3436/3436, une fois chacune.

### Recouvrements et limites des règles

- Même SIREN387 ; état A/F opposé195 mais41 exclusifs (154 déjà siège/site). Numéro structuré absent702 mais443 exclusifs. Label loin typé638 mais463 exclusifs. Les gains de drapeaux recouvrants ne se somment pas.
- Générique strict :3 drapeaux mais0 exclusif ; règle volontairement étroite, pas absence de noms génériques. Description complémentaire posthoc, non utilisée pour reclasser :383 noms contiennent un marqueur générique même accompagné d’autres mots.
- 190/702 numéros structurés inexploitables ont un chiffre dans la voie (pas forcément un numéro valide). Ne pas assimiler ce défaut de champ à une information irrémédiablement absente.
- Points label–CRM mesurables2717/3436, typés admissibles2698 ;10 distances>500m non typées exclues de la règle loin. L’incertitude et les dates CRM peuvent rester non connues ; aucune distance ne démontre que le label est faux.
- « Même adresse » ne prouve ni occupation simultanée ni succession entre SIREN. Les conflits d’objectif/site et les erreurs de représentation ne prouvent pas un plafond irréductible d’information.

### Signaux publics locaux par famille

| Famille | Signal possible et présence dans85 |
|---|---|
| Même SIREN : siège choisi / label site | Déjà : siège/statut, adresse et distance typée, noms de site. Manquants : création, effectifs, sens du nom CRM ; adresse seule ne prouve pas le rôle juridique. |
| Même SIREN : fermé ↔ actif | Déjà : actif/fermé, vraie fermeture historique et ancienneté, effectifs actifs du SIREN. Manquants : création ; date du CRM non attestée, donc ordre temporel non résolu. |
| Même SIREN : autre site proche | Déjà : enseignes/usuel du site, numéros/suffixe/voie, distance et précision. Manquants : NAF×vocabulaire et sémantique du rôle (primaire/maternelle, laboratoire/cabinet). |
| Même SIREN : autre établissement restant | Déjà : noms de site, géographie, statut/siège, comptages. Création/NAF/effectifs manquent ; vérifier la relation nom de site / fonction métier plutôt que le parent seul. |
| Autre SIREN : même adresse stricte | Adresse/distance sont déjà présentes et peuvent être identiques. NAF×vocabulaire, création/effectifs et sémantique sont absents. Deux SIREN à une même adresse peuvent aussi être des exploitants successifs, pas nécessairement co-locataires simultanés. |
| Homonyme / enseigne, autre lieu | Déjà : enseigne/usuel lexical, adresse/commune, distance typée, statut. La sémantique seule peut confondre les mêmes enseignes ; NAF/dates n’aident que si contrastés. |
| Label à plus de500m du CRM | Déjà : distance, qualité/incertitude, adresse, siège/statut et noms. Création/effectifs/NAF peuvent informer, sans prouver une erreur de label ni résoudre siège vs site. |
| Numéro CRM structuré non exploitable | Déjà : manque numéro, voie, géométrie/incertitude. Certaines voies contiennent un numéro : absence STRUCTURÉE n’est pas absence d’information. NAF/sémantique utiles possibles, aucun gain démontré. |
| Nom générique strict | Noms/enseigne/UL lexicaux déjà là. NAF×vocabulaire et sens du rôle absents ; zéro cas exclusif sous la règle stricte ne signifie pas zéro nom peu distinctif. |
| Autre | Mélange restant, pas une cause. NAF, création/effectifs et sémantique manquent aux85 ; noms historiques publics du cache512 ne sont pas tous dans les noms actuels85. |

NAF/nomenclature :3436/3436 choix et labels renseignés. Dans la famille même adresse, **668/1046 codes NAF diffèrent (63,86%)** : c’est un contraste observable, pas668 erreurs automatiquement résolubles ; aucun alignement code–vocabulaire CRM n’a été testé. Une table locale officielle code→libellé NAF n’a pas été vérifiée.
Création renseignée3426 choix/3432 labels. Effectifs :1481/2369 tranches horsNN connues ; année1502/2390. Une tranche datée n’est pas un nombre courant exact. `dateDebut` est le début de période, pas une preuve de fermeture ; les85 utilisent déjà la vraie transitionF de l’historique.

###20 exemples lisibles — sélection SHA par tour de familles

Une ligne : CRM → choix (nom, SIRET, adresse) / label humain (nom, SIRET, adresse). Tous restent des erreurs mesurées ; aucun verdict de label corrigé.

1. **Même SIREN : siège choisi / label site** — COMMUNE DE MEZIDON VALLEE DAUGE   MAIRIE DE COUPESARTE 14140 14431 → MAIRIE / COMMUNE DE MEZIDON VALLEE D'AUGE `20006499600016`, 2  ALLEE DU PARC DU BREUIL 14270 14431 / LABEL COMMUNE DELEGUEE DE COUPESARTE / COMMUNE DE MEZIDON VALLEE D'AUGE `20006499600032`, L'EGLISE 14140 14431.
2. **Même SIREN : fermé ↔ actif** — GROUPAMA SUPPORTS ET SERVICES CMP92000EICAMPUSWP5   AVENUE FREDERIC ET IRENE JOLIOT CURIE 92000 92050 → G2S / GROUPAMA SYSTEMES D'INFORMATION `41054695600260`, 184  AVENUE FREDERIC ET IRENE JOLIOT CURIE 92000 92050 / LABEL G2S / GROUPAMA SYSTEMES D'INFORMATION `41054695600237`, 7  BOULEVARD DES BOUVETS 92000 92050.
3. **Même SIREN : autre site proche** — 3CSUD 2  RUE DE L HOSTELLERIE 30900 30189 → 3CSUD `53806559000050`, 242  RUE CLAUDE NICOLAS LEDOUX 30900 30189 / LABEL 3CSUD `53806559000043`, 62  RUE DE L'HOSTELLERIE 30900 30189.
4. **Même SIREN : autre établissement restant** — FORTIL OUEST 1  IMPASSE DES JADES 44300 44109 → FORTIL NORD OUEST `82871667000033`, 1  IMPASSE DES TOURMALINES 44300 44109 / LABEL FORTIL NORD OUEST `82871667000025`, 11  RUE DU CHEMIN ROUGE 44300 44109.
5. **Autre SIREN : même adresse stricte** — METROPOLIS ARCHITECTES ASSOCIES 173  AVENUE CHARLES DE GAULLE 69160 69244 → METROPOLIS ARCHITECTES ASSOCIES `79908465200020`, 169  AVENUE CHARLES DE GAULLE 69160 69244 / LABEL METROPOLIS ARCHITECTES ASSOCIES `43994284800029`, 169  AV CHARLES DE GAULLE 69160 69244.
6. **Homonyme / enseigne, autre lieu** — KOOKAI 8  PLACE SAINTE OPPORTUNE 75001 75101 → FROM FUTURE / KOOKAI `33852356600267`, 320  RUE SAINT-HONORE 75001 75101 / LABEL KOOKAI / KOOKAI `39929299400049`, 2  RUE SAINTE-OPPORTUNE 75001 75101.
7. **Label à plus de500m du CRM** — DOMAINE ROLET PERE FILS 11  RUE DE L HOTEL DE VILLE 39600 39013 → DOMAINE ROLET `38829392000012`, 11  RUE DE L'HOTEL DE VILLE 39600 39013 / LABEL SCEA DOMAINE ROLET PERE ET FILS `48949525900017`, ROUTE DE DOLE 39600 39013.
8. **Numéro CRM structuré non exploitable** — MANULOC   0D ZONEINDUSTRIELLE DU PARC STOCKS 62820 62907 → SIMASTOCK / SIMASTOCK SOCIETE INDUSTRIELLE DE MANUTENTION ET DE STOCKAGE. `35181985900536`, ZI DU PARC A STOCK 62820 62907 / LABEL A1 LOCATION PHILIPPE MTP / PHILIPPE MANUTENTION `30341361100037`, ZI DU PARC A STOCK 62820 62907.
9. **Autre** — FIRST INN HOTEL 14  RUE DE LA REUNION 91940 91692 → FRANCOI DIZEINI `10399674000017`, 14  RUE DE LA REUNION 91940 91692 / LABEL HR EX LES ULIS `49271764000030`, RUE DE LA REUNION 91940 91692.
10. **Même SIREN : siège choisi / label site** — COMMUNE DE LANGON 8  COURS DES CARMES 33210 33227 → MAIRIE / COMMUNE DE LANGON `21330227600018`, ALLEES JEAN JAURES 33210 33227 / LABEL CENTRE CULTUREL DES CARMES / COMMUNE DE LANGON `21330227600117`, 8  PL DES CARMES 33210 33227.
11. **Même SIREN : fermé ↔ actif** — KUEHNE NAGEL   RUE DE L ESPACE SCHENGEN 77127 77251 → KUEHNE+NAGEL / KUEHNE+NAGEL `33358346601761`, BOULEVARD JEAN MONNET 77127 77251 / LABEL KUEHNE+NAGEL / KUEHNE+NAGEL `33358346600912`, DISTRIPOLE 77127 77251.
12. **Même SIREN : autre site proche** — PRIMAIRE LES BLEUETS   SQUARE AVERROES 69009 69389 → ECOLE PRIMAIRE LES BLEUETS / COMMUNE DE LYON `21690123102470`, SQUARE AVERROES 69009 69389 / LABEL ECOLE MATERNELLE BLEUETS LES CAPUCINES / COMMUNE DE LYON `21690123100839`, RUE MARCEL CERDAN 69009 69389.
13. **Même SIREN : autre établissement restant** — EQIOM FR70   ROUTE DU MOLE 4 59140 59183 → EQIOM `37791706700318`, ROUTE DU FOSSE DEFENSIF 59140 59183 / LABEL EQIOM `37791706700300`, BASSIN MARITIME 59140 59183.
14. **Autre SIREN : même adresse stricte** — LABORATOIRE BIO VSM BUSSY ST GEORGES 7  RUE KONRAD ADENAUER 77600 77058 → SCM CABINET DENTAIRE DU CLOS ST GEORGES `38230783300022`, 7  RUE KONRAD ADENAUER 77600 77058 / LABEL BIO VSM LAB `39998309700037`, 7  RUE KONRAD ADENAUER 77600 77058.
15. **Homonyme / enseigne, autre lieu** — SILIB 253  BOULEVARD ROBERT KOCH 34500 34032 → SILIB / SILIB `53076924900059`, 253  BOULEVARD ROBERT KOCH 34500 34032 / LABEL SILIB / PRESTIGE PROPRIETES `49206270800036`, 11  IMPASSE JOSEPH ROUMANILLE 34500 34032.
16. **Label à plus de500m du CRM** — MAS L ALMARRE INSTIT REED FONCT POMPONIANA OLBIA 2314  AVENUE DE LA FONT DES HORTS 83400 83069 → MAISON D'ACCUEIL SPECIALISEE MAS / SERVICES ADMINISTRATIFS `77557120100190`, 2314  AVENUE DE LA FONT DES HORTS 83400 83069 / LABEL CE INST REEDUC FONCT POMPONIANA OLBIA `31184500200014`, L ALMANARRE 83400 83069.
17. **Numéro CRM structuré non exploitable** — MAITRE BEATRICE DESAGNEAUX   23B RUE DE CONSTANTINOPLE 75008 75108 → MAITRE DATTIER `88795387500028`, 13  BOULEVARD DE COURCELLES 75008 75108 / LABEL BEATRICE DESAGNEAUX HUISSIER DE JUSTICE / BEATRICE DESAGNEAUX HUISSIER DE JUSTICE `83778638300013`, 23 B RUE DE CONSTANTINOPLE 75008 75108.
18. **Autre** — HOTEL RADISSON BLU BLAGNAC 2  RUE DIEUDONNE COSTES 31700 31069 → HOTEL TOULOUSE BLAGNAC `98880707900021`, 92  RUE DE BORDEBASSE 31700 31069 / LABEL 2 RUE DIEUDONNE COSTES `48350070800035`, 2  RUE DIEUDONNE COSTES 31700 31069.
19. **Même SIREN : siège choisi / label site** — BIOAMIENS 59  RUE ALEXANDRE DUMAS 80000 80021 → LABORATOIRE ALBERT 1 ER / BIOAMIENS `49108880300011`, 51   MAIL ALBERT 1ER 80000 80021 / LABEL LABORATOIRE DE LA VALLEE DES VIGNES / BIOAMIENS `49108880300029`, 59  RUE ALEXANDRE DUMAS 80090 80021.
20. **Même SIREN : fermé ↔ actif** — FNAC DARTY PARTICIPATIONS ET SERVIC 1  IMPASSE JACQUELINE AURIOL 91380 91161 → DARTY / ETABLISSEMENTS DARTY ET FILS `54208661601769`, 1  RUE JACQUELINE AURIOL 91380 91161 / LABEL ETABLISSEMENTS DARTY ET FILS `54208661601405`, 53  ROUTE DE LONGJUMEAU 91380 91161.

### Gemma : existence, compatibilité et coût réel

- **1348644 SIRET G0 uniques**,159306 clés dans les caches SITE512 existants, soit**11,8123%** ; **1189338 absents (88,1877%)** de cette banque.231 zones : documents SHA liés au producteur, adapter/step et tous les en-têtes768/shards vérifiés ; pas un remplacement par les2126387 vecteurs parent/noms seuls.
- Ordre texte→ligne et SHA/valeurs de tous les tenseurs rejoués sur UNE zone (02408) :12373 sites/11636 lignes texte, dont **2500 SIRET G0**, soit0.1854% du banc de sites. Les11,81% attestent les clés et leur matériel lié au producteur ; **pas une revalidation indépendante du mapping de toutes les zones**.
- **CRM nom+adresse :0/17427 vecteurs persistés identifiés (0%)**. Inventaire des noms de fichiers sur H horsv174/runtime/base/test/control :8 artefacts query-vector/binding, tousv33/v36 ; leurs banques noms seuls526 sont disjointes du banc,0 couverture même en noms seuls. Le producteurv14 encode les fiches CRM en mémoire mais ne les persiste pas. Absence dans ce périmètre audité, pas preuve contre toute archive non identifiée.
- **1000 textes réels uniques mesurés sur Metal :15.272s**,500 fiches CRM+500 premières vues SITE d’une zone G0. Chargement4.308s, warmup4 textes0.157s séparés. Batch4, max512 sans troncature, médiane58,5/p95169/max336 tokens ; Torch2.14.1/Transformers5.19.0, modèle/adapter local step1000 inchangé, aucun gradient.
- Détail MESURÉ :500 CRM en3.861s,500 SITE en11.411s. Débits1000 extrapolés :7,722s CRM et22,821s SITE. Seuls1000 items+4 warmup encodés ; aucun vecteur ajouté aux caches ni sauvegardé, aucune masse lancée.
- Scénario **une vue/site, sans déduplication, même débit que cet échantillon** : compléter les clés absentes≈7.54h +134.6s CRM ; si toute la banque SITE devait être réencodée≈8.55h +CRM. Ce ne sont PAS des temps mesurés ni budgets garantis : nombres de vues, longueur des absents, représentativité d’une zone et déduplication inconnus.
- Pic lecture CPU1.96Gio ; probe Gemma RSS1.14Gio, driver1.25Gio, somme simultanée max**2.25Gio** (<4 ; somme conservatrice en mémoire unifiée). Adapter caché non entièrement OOF, donc aucun score de validation ne lui est attribué.

### Une seule expérience recommandée

Une expérience : pilote de200 entreprises tirées par SHA sans sélectionner les erreurs, ajout aux85 d’une seule cosine Gemma BASE non adaptée (CRM nom+adresse / site noms publics+adresse actuels), mêmes5 plis entreprise et R3 fixes, delta UNION par bootstrap entreprise ; garder les100 et toutes les requêtes. Pas de nouvel objectif de précision ni promotion, pas d’encodage national.
Ce pilote utiliserait la base non fine-tunée pour éviter la contamination potentielle de l’adapter des caches, et des textes actuels limités aux informations85 pour ne pas confondre sémantique et ajout d’historique. Ces nouveaux vecteurs ne seraient pas compatibles avec les anciens caches adaptés. Expérience seulement proposée, non exécutée ; aucune promesse de gain.

### Preuve et reproduction

- Smoke40 erreurs puis3436 :16,82s lecture complète,40 affectations/faits identiques,0 fit,3436 classes uniques ;20 exemples. Protocole commit`c244c0d2` figé avant comptes, scripts hashés dans le reçu, SHA sources/banc/labels avant-après, aucune source modifiée.
- Namespace `H/step2_error_families_g0_20261019/` : `analyze.py`, `gemma_probe.py`, `publish.py`, `preregistered.md`, `smoke.json`, `analysis.json`, `gemma_cost.json`, `verification.json` ; logs `smoke.log`,`full.log`,`gemma.log`.
- Commandes réellement lancées : `.venv/bin/python -u H/step2_error_families_g0_20261019/analyze.py --smoke`, puis sans`--smoke` ; `$H/runtime/bin/python -u H/step2_error_families_g0_20261019/gemma_probe.py` ; publication et vérification `publish.py`. Tous avec`PYTHONDONTWRITEBYTECODE=1`, workers séquentiels, pas de suite unitaire/web/VM/jeu fermé/v174/positif injecté/push. Travaux tiers/pending préservés.
- Pré-déclaration commit`c244c0d2` ; résultat auto-référencé par `git log -1 --format=%H -- docs/step2_error_families_g0.json`.
