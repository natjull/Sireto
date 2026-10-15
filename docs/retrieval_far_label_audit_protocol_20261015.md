# Audit des labels éloignés — protocole figé du 15 octobre 2026

## Objet et population

Audit descriptif de qualité de vérité, sans modifier un label, une coordonnée,
un candidat ou une métrique historique. Le CRM vise le site raccordé à la fibre ;
un écart de coordonnées n'établit pas à lui seul une erreur humaine.
Population : les **86** lignes `exact=true` et
`label_site_diagnostic.label_far_from_crm=true` du résultat
`docs/retrieval_site_operational_metric_result_20261014.json`
(SHA256 `54988149ccfe156c176780c930b0d608b64317024ae7f45022d6aede47836f16`).
Elles appartiennent aux 440 succès exacts mesurables sur 507 succès exacts,
dans les 526 requêtes de développement, folds 2/3/4 uniquement.
Aucune lecture de fold0, fold1, test4323 ou control962.

## Tirage, avant toute lecture individuelle des cas

Trier les `query_id` comme chaînes Python (ordre lexicographique), puis
`random.Random(20261015).sample(ids, 30)` : sondage aléatoire simple sans remise.
Ne jamais retirer, remplacer ou redessiner une ligne, même sans preuve disponible.
Ordre du tirage immuable :

```json
["5010", "4360", "NEWCRM:58e9008a310b509ce47da72b1aaffd1a", "NEWCRM:7d92ccf89904477aad39be1ab49ecdf9", "11185", "12944", "1151", "NEWCRM:de2cddee252d0afb4bc1883f8734802d", "389", "NEWCRM:9a36e53253268cef68169de0db776a9d", "NEWCRM:3ac8233f7524db85a30659a5bd58194c", "NEWCRM:2836cc0263fe9ef7540bed3a8a0d3266", "NEWCRM:ba537927a8ded41fa84d977502486fbe", "NEWCRM:bead59a4ba32bffc4c244e2a144802ce", "5121", "NEWCRM:803b3525a250be994e57f391f41b6b69", "13566", "NEWCRM:b38f9f7fc040bc393760ce781998bcbd", "3827", "NEWCRM:a9ea8e8f260c573f5c932ac03ddbf382", "6573", "13608", "NEWCRM:12c75e845505aed3a1caeda0084af4a6", "9689", "NEWCRM:6fcf2acdd8cb13aafa51156788396813", "16478", "1763", "8637", "NEWCRM:fd76dea1d399b9fa7b3c56744b253e67", "15303"]
```

Le présent fichier seul est commité avant hydratation et examen des cas.
Le rapport d'agent antérieur demandé est un fichier vide (0 octet), donc sans
information supplémentaire exploitable.

## Sources locales et méthode

Réutiliser les bindings scellés du résultat du 14 octobre : panel CRM original,
géométrie BAN v17, sorties `learned_native_ooc` du run
`v142/bano_native_20261013_night/all_new_sources_max_bano`, ledger officiel
filtré en amont sur les 30 identifiants ET folds 2/3/4. Les autres lignes ne
sont pas matérialisées pour l'audit. Réutiliser le stock SIRENE du 2026-08-01
(adresses, noms officiels/enseignes, siège, état administratif), et l'étude
géolocalisation SIRENE du 2026-07-01 (XY, EPSG, qualité, précision).
BAN/BANO : seulement les preuves déjà présentes dans la géométrie locale,
pas de géocodage supplémentaire ni sélection opportuniste de coordonnées.

Scanner **tous** les établissements des SIREN des 30 labels, pas uniquement les
candidats, et si nécessaire les occupants SIRENE à l'adresse CRM pour documenter
une autre entreprise. Scans DuckDB/Arrow projetés et filtrés, un thread,
512 Mio pour DuckDB, garde RSS <3,5 Gio et limite processus <4 Gio si disponible.
Petites sources hashées ; gros stocks : références SHA héritées, taille/mtime/inode
vérifiés avant/après, sans nouveau hash intégral. Sources ouvertes en lecture seule.
Aucun accès web/API, aucun run/evaluate/retrieval, aucun modèle entraîné ou rejoué.

Pour chaque établissement du même SIREN : règle adresse stricte OU distance<=T
du contrat du 14 octobre, fonctions du script existant réutilisées sans changement.
Numéro/suffixe/voie normalisée et INSEE, ou T=200/500 m avec qualités admissibles.
Publier les concordances, distance, qualité, adresse/distance comme preuve,
actif (`A`)/fermé (`F`)/inconnu et rang émis à base 1, ou hors top100.
Un résultat négatif signifie « aucune concordance selon cette règle dans ces
sources », non preuve d'absence réelle d'occupant. Le label exact reste distant
selon sa source de point, même si une concordance textuelle existe.

## Catégorie unique, priorité obligatoire

Examiner dans l'ordre suivant, et retenir la première catégorie étayée :

1. **SIEGE** : label officiellement siège et adresse CRM différente. Exiger une
   différence textuelle (numéro/voie/commune), pas seulement une distance XY ;
   signaler si le site CRM de la même entreprise n'est pas établi.
2. **AUTRE_SITE** : label non-siège ; un autre établissement du même SIREN
   concorde à l'adresse CRM selon la règle, avec adresses textuelles différentes.
3. **AUTRE_ENTREPRISE** : aucune concordance du SIREN après scan complet et
   preuve officielle locale d'une autre entreprise cohérente avec le nom CRM
   à son adresse. Sans preuve d'occupant, ne pas confondre absence de concordance
   et entreprise différente ; rester INDETERMINE.
4. **ADRESSE_CRM_DOUTEUSE** : adresse incomplète, mal géocodée ou incohérente
   (voie/CP/commune), avec adresse label plausible pour le nom CRM. Expliciter
   le défaut observé et distinguer mauvaise géolocalisation supposée/mesurée.
5. **COORD_IMPRECISES** : adresses textuelles concordantes ou quasi (préciser
   l'écart textuel, ne pas assouplir la règle principale), mais points éloignés.
6. **INDETERMINE** : preuve insuffisante ou contradictoire ; dire ce qui manque.

Aucune règle ad hoc par entreprise ; nom officiel compatible n'est pas une preuve
de site fibre. Chaque décision conserve ses faits sources et son niveau
« mesuré / supposé / inconnu ». Priorité SIEGE peut masquer un problème de XY :
la catégorie décrit le constat administratif, pas une cause certaine de label erroné.

## Restitution et extrapolations préenregistrées

`docs/retrieval_far_label_audit_20261015.{json,md}` : les six comptes sur 30,
pourcentages et Wilson bilatéral 95 % (z=1,959963984540054), y compris les zéros.
Pas de correction de population finie du Wilson demandé ; pas de pondération.
Pour chaque catégorie de proportion p : nombre indicatif attendu **86*p**
dans les 86 ; même nombre dans la strate éloignée incluse dans les 440,
et part **86*p/440** des 440. Intervalles indicatifs obtenus par les mêmes
transformations du Wilson, explicitement non des observations exhaustives.
Ne pas calculer 440*p : les 30 ne sont pas tirées parmi les 440. Rien n'est
inféré sur les 354 autres mesurables ou les 67 non mesurables, ni sur la nation.
Séparer constats administratifs, soupçons de cible hors site et vérité erronée
non démontrée ; ne pas annoncer automatiquement SIEGE+AUTRE_SITE comme taux
d'erreur certifié. Tableau par query_id, catégorie, même SIREN au CRM/rang/état,
phrase française <=20 mots ; détails probants dans le JSON et annexe locale.

Script éventuel `scripts/retrieval_far_label_audit.py`, smoke réel sur le tirage
figé (pas de tests unitaires). Commit isolé script/résultats et une ligne de
`handover.md` référençant protocole et commit résultats de façon résoluble.
Aucun autre fichier ajouté au commit, aucun push, sources et travaux étrangers préservés.
