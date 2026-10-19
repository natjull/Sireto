# Audit des faux « sûr » — optionD, volume17427

Source figée : optionD `7b0005a8`, `content/margin` et `stack/margin`.
**5 occurrences,4 dossiers uniques. Tous restent des erreurs UNION ; aucun
label, dénominateur, seuil ni statut publié n'est corrigé par cet audit.**
Les sélections sont celles des modèles de confiance et de leurs seuils
internes, pas celles des full-fits du tableau top1. Les5 choix étaient déjà
au rang1 de V142 : il s'agit ici d'échecs de rejet de choix natifs ambigus.

## Rejeu et sources

`scripts/step2_g0_bench.py _audit` rejoue les modèles sérialisés, le rang,
la marge et la comparaison au seuil pour CHAQUE faux SUR union sur les526.
Faits structurés : `H/step2_g0_bench_20261017/smoke/audit/facts.json`, SHA256
`892166e52c6e979829af2664443c21362875088fd577540de175d6e7999d1daf`.
Le run complet reproduit ces mêmes faits dans son namespace `full/audit/`.
CRM/pool/géométrie V142 immuables ; stock SIRENE août2026, points typés étude
juillet2026 ; aucun web, recherche de successeur, géocodage ou correction.
Distance = euclidienne Lambert93. Une distance sans point CRM est **non
mesurable**, jamais zéro. Les observations de juillet/août ne datent pas le CRM.

| Politique | SUR | Corrects union | Précision union, Wilson95 | Couverture |
|---|---:|---:|---|---|
| content/margin |56|54|96,43% [87,88;99,02]|10,65%|
| stack/margin |61|58|95,08% [86,51;98,31]|11,60%|

La cible98% n'est pas atteinte en test, même si elle l'était empiriquement
sur de très petites calibrations : content folds2/3/4=6/6,5/5,15/15 ;
stack=3/3,12/12,17/17. Aucune borne Wilson n'était imposée au seuil.

| Politique | Seuil fold2 | Seuil fold3 | Seuil fold4 |
|---|---:|---:|---:|
| content/margin |3,3782331943511963|3,581026077270508|2,535097360610962|
| stack/margin |5,090987682342529|4,092787265777588|3,337373733520508|

##1. MON VETO — faux pour les deux politiques

- Dossier `NEWCRM:15c976dd6d8de9da0360b5afa9fe00e4`, fold population4.
- CRM : **MON VETO**,2 RUE RAYMOND BARBET,92000, INSEE92050.
- Choix : **43198246101688**, SIREN431982461, MON VETO ; même adresse au2,
  **actif, établissement non siège** ; concordance adresse stricte vraie.
- Label : **51354012000013**, SIREN513540120, BARBET PLAINCHAMP ;5 RUE
  RAYMOND BARBET,92000/92050 ; **fermé, siège** ; adresse stricte fausse.
- Même SIREN ? **Non**. CRM→choix **2,89m**, CRM→label **27,07m**,
  choix→label **29,02m**. CRM bâtiment/points étude11 : seuil distance200m.
  Le faible éloignement ne compense jamais la différence de SIREN dans UNION.
- Marge content **3,2581281661987305 ≥2,535097360610962** ; marge stack
  **5,191051959991455 ≥3,337373733520508**. Les deux acceptations sont rejouées.
- **Verdict — label discutable :** le CRM correspond au nom et au numéro2 du
  site actif plutôt qu'au siège fermé labellisé au5, mais aucune succession
  juridique entre ces deux SIREN n'est prouvée et le label reste inchangé.

##2. Hôtel du Golf — faux content/margin

- Dossier `NEWCRM:db5e4874e021371c69011789c42a1afa`, fold population2.
- CRM : **SAS HOTEL DU GOLF ST OMER HOTEL DU GOLF**, sans numéro,
  LE BOIS DE LA HAUTE DREVE,62380, INSEE62008 ; **point CRM absent**.
- Choix : **48850820100016**, SIREN488508201, HOTEL DU GOLF DE SAINT OMER,
  enseigne LE RISTANDEL ; CHEMIN DES BOIS sans numéro,62380/62008 ;
  **actif, siège**, point étude11 (numéro équivalent).
- Label : **39924578600021**, SIREN399245786, NAJETI ; même CHEMIN DES BOIS
  sans numéro,62380/62008 ; **actif, siège**, point étude12 (rue).
- Même SIREN ? **Non**. Distances CRM→choix/label et seuil : **non mesurables**.
  Choix→label **322,00m** ; ce n'est pas une preuve de distance au CRM.
  Aucune adresse stricte ne peut être validée sans numéro CRM.
- Marge **3,4282712936401367 ≥3,3782331943511963** ; acceptation rejouée.
- **Verdict — indéterminé :** sans point CRM ni preuve du lien entre l'hôtel
  nommé et NAJETI, le nom et la même voie ne suffisent pas à trancher l'entité
  juridique à rattacher.

##3. SAS5EME ETAGE — faux stack/margin

- Dossier `NEWCRM:1c6805a5f39956c1909201e3e901b453`, fold population3.
- CRM : **SAS5EME ETAGE**,22 RUE LA TOUR D AUVERGNE,44000, INSEE44109.
- Choix : **84287677300016**, SIREN842876773,5EME ETAGE ; **27** RUE
  LA TOUR D'AUVERGNE,44200/44109 ; **actif, siège** ; adresse stricte fausse.
- Label : **53535805500049**, SIREN535358055, LE MUR DU SONGE ; **22** même
  voie,44200/44109 ; **actif, siège** ; concordance adresse stricte vraie.
- Même SIREN ? **Non**. CRM→choix **17,96m**, CRM→label **10,00m**,
  choix→label **27,58m**. CRM parcelle = rue/imprécis, points étude11 :500m.
  La proximité des deux sociétés ne démontre pas leur identité.
- Marge **4,921479225158691 ≥4,092787265777588** ; acceptation rejouée.
- **Verdict — indéterminé :** le nom soutient le choix au27 tandis que le
  numéro soutient le label au22, et les seules preuves disponibles ne résolvent
  pas cette contradiction CRM ni un éventuel lien commercial entre sociétés.

##4. FOURNIER — faux stack/margin

- Dossier `NEWCRM:803b3525a250be994e57f391f41b6b69`, fold population3.
- CRM : **FOURNIER**,350 RUE DES CLAUWIERS,59113, INSEE59560.
- Choix : **32305708301200**, SIREN323057083, FOURNIER RETAIL,
  enseigne SOCOO'C ; même adresse350,59113/59560 ; **actif, siège**.
- Label : **42857705000183**, SIREN428577050, SODIMOB ; même adresse350,
 59113/59560 ; **fermé, siège**. Les deux adresses strictes concordent.
- Même SIREN ? **Non**. CRM→choix et CRM→label **790,62m** ; choix→label
  **0,00m**. CRM déclaré faible = rue/imprécis, points étude11 :500m ; la voie
  distance échoue pour les deux, mais la voie adresse reste valide sous réserve
  du SIREN. `distance_precision=0` n'est pas un certificat de point parfait.
- Marge **4,573137283325195 ≥4,092787265777588** ; acceptation rejouée.
- **Verdict — label discutable :** le nom FOURNIER et le site actif coïncident
  avec l'adresse et le point du label SODIMOB fermé, ce qui questionne son
  actualité sans prouver une succession juridique ni autoriser une substitution.

## Synthèse prudente

- Par dossier unique : **2 labels discutables,2 indéterminés** ; aucune erreur
  de label démontrée et aucune cause « erreur modèle seule » incontestable.
- Par occurrence : **3 labels discutables,2 indéterminés** ; **les5 restent
  des erreurs mesurées du modèle** dans la définition UNION, inchangée.
- Les4 dossiers opposent des SIREN différents, même quand noms/adresses/points
  sont très proches. Pas de nouveau statut, de tolérance inter-SIREN ou de
  règle spécifique à ces cas : l'audit n'alimente pas les features/cibles.
- Une vérification officielle de l'entité contractante et de la date du CRM
  pourrait départager ces cas ; elle n'a pas été exécutée, aucun gain de
  précision corrigée n'est annoncé et aucun SUR n'est validé pour production.
