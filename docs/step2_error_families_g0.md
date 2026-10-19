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

EN_ATTENTE — aucune famille comptée ni coût Gemma annoncé à ce stade.
