# BODACC administration : ce qui est utilisé et ce qui ne l'est pas

Audit statique du 14 octobre 2026, pendant le mandat nocturne. Aucun nouveau
scan de BODACC, nom personnel, adresse, label, calcul natif ou pipeline modifié.
Ce document décrit une complétude de code et de sources, pas un gain de recall.

## Conclusion

Les **sociétés dirigeantes sont déjà une source de contexte de retrieval via
le RNE**. Les noms des dirigeants personnes physiques ne le sont pas.
Le champ libre **BODACC `personne.administration` n'entre pas dans ce contexte** :
il est exclu de la construction canonique et seulement lu dans des diagnostics
post-scellés. Un apport historique complémentaire est donc possible, mais
ni sa couverture supplémentaire ni son utilité ne sont encore mesurées.

## Lineage concret

| Chemin | Rôle conservé | Usage réel |
|---|---|---|
| BODACC `listepersonnes → personne.administration` | Aucun dans le builder canonique | Ignoré ; pas alias, pas graphe, pas feature |
| RNE `content.*.composition.pouvoirs → entreprise` | Sujet enregistré → société représentante, code de rôle | Table distincte des noms propres du sujet |
| V30 → V56/V57 → V71 | Noms de sociétés représentantes, codes 75/73/99/71/OTHER | Canal lexical corporate et interactions avec les preuves propres du site |
| V72 → V88/V99 → V115 | Acteur corporate direct et pairs distincts partageant un acteur | Contexte national avant le même collecteur 100, pas une équivalence d'identité |
| BODACC `acte.descriptif` → V65 → V71/V88 | Clause stricte de société mère annoncée | Contexte publié distinct ; ce n'est pas `administration` |
| V49/V52 `administration` | Diagnostic post-scellé | Ne construit pas de source utilisée par V115 |

Preuves de code :

- `src/xgb_matcher/official_evidence_builder.py`, `_canonicalize_bodacc`
  (1057), inspection d'identité explicitement limitée (1080), exclusions de
  manifeste (1824/2004) : `directors`, `bodacc_full_text`,
  `relations_inferred_from_text`. Aucune lecture d'`administration` dans ce fichier.
- `src/xgb_matcher/rne_legal_entity_roles.py`, `extract_legal_entity_roles`
  (179) lit uniquement les pouvoirs ayant `typeDePersonne=ENTREPRISE` **ou un
  objet `entreprise` explicite**. Il ne lit pas les objets de personnes physiques,
  leurs noms, bénéficiaires effectifs ou adresses. Il retient dénomination,
  identifiant corporate lorsqu'il existe, code/libellé de rôle, dates et
  rattachements SIRET explicitement structurés. Ces rattachements ne sont pas
  des preuves d'exploitation ou de propriété.
- `scripts/embeddinggemma2_v56_corporate_site.py`, `compile_context` (114)
  exige deux SIREN de neuf chiffres distincts et filtre les dates connues après
  le 1er août 2026. Les dates d'effet inconnues restent inconnues. La portion
  `représenté par …` est supprimée de la dénomination corporate, pour ne pas
  transformer le représentant naturel en contexte lexical.
- V56 `corporate_component` (205) utilise couverture de tokens/IDF et maximum
  par propriétaire ; champ lexical corporate 6. Les interactions numériques
  sont rôle × géographie/commune/voie/numéro de voie **propres au candidat**.
  Son composant lexical direct n'impose pas à lui seul une géographie valide ;
  ne pas confondre ce composant avec la branche V88/V99 ci-dessous.
- `scripts/embeddinggemma2_v72_corporate_incidence.py` garde comme nœuds liés
  les sujets et acteurs corporate publiés `O`, hors EI, à dénomination présente.
  Les incidences distinctes ne multiplient pas le soutien quand un acteur a
  plusieurs rôles. Les noms d'acteurs non résolus sont stockés séparément,
  sans relation légale résolue ; cela ne garantit pas leur usage dans V88/V99.
- V88 `prepare`/V99 `BeforeOnlyCollector` : acteur direct et pairs distincts,
  atténuation `.5/sqrt(degré)` ; MAX des contextes et soutien administratif
  INSEE/CP de la branche originale ; poids corporate gelé, quantification
  20 bits, documents SIRET et un seul collecteur initial 100.
- `scripts/embeddinggemma2_v115_streamed_activity_ablation.py` importe V99
  comme baseline : ce n'est pas seulement un ancien prototype RNE isolé.

La table RNE déjà locale déclare **1 754 075 rôles** issus de **27 348 473
enregistrements**, dont **1 640 743 rôles avec un SIREN lié structuré**.
Ces comptes viennent de son manifeste, pas d'un nouveau scan et pas d'une
mesure de couverture CRM. SHA de table :
`e626cdf17395693411375d41c6848fcf6085d62ce3c66d08cf1388a0d42457b1`.
Les filtres consommateurs sont plus restrictifs que la table source ; le
nombre restant n'est pas recompté ici. V56 ne vérifie pas lui-même la diffusion
des deux unités légales dans `compile_context`, contrairement au graphe V72 ;
sa préparation des interactions lit les établissements publics. Ne pas attribuer
automatiquement à toute la chaîne le masque corporate bilatéral de V72.

## Pourquoi ne pas simplement indexer toute la chaîne administration

Le [schéma officiel DILA](https://echanges.dila.gouv.fr/schemabook/bodacc/sor/Bodacc_rediff_all.xsd)
déclare `Personne_Type.administration` comme une **chaîne optionnelle**, pas
une liste structurée séparant sociétés et personnes physiques, ni un jeu
d'arêtes corporate. Copie de schéma déjà vérifiée par V152, SHA
`bb29490b5096908d563c772623bfee6b6cf51d26aeb510310d0b08c8dd63b95b`.

Un token de nom commercial présent dans une chaîne de dirigeants ne prouve
pas qu'il désigne une société : il peut appartenir au nom d'une personne.
Faire un simple match texte contre des noms corporate ne résout pas cette
ambiguïté. De même, « président » ne signifie pas « propriétaire », une
mention publiée ne garantit pas une relation actuelle, et la relation porte
sur la société annoncée, pas sur tous ses sites comme identité locale.

## Une seule direction proposée, source-only et non lancée

Construire, si autorisé, un audit whole-source des 19 années avant cutoff
sur le **sujet PM directement immatriculé**, corporate actuellement public
et hors EI, sans croisement d'acteurs. Extraire uniquement les clauses qui
portent **explicitement une personne morale administratrice**, en conservant
le rôle annoncé, la date de publication et l'incertitude de date d'effet.
Les clauses ambiguës, personnelles ou sans attribution sûre restent exclues
et comptées ; aucun texte administratif brut ou nom naturel exporté.

Le nom corporate ainsi attesté serait un **contexte de société dirigeante**,
jamais alias propre, identité de site, opérateur supposé ou propriété.
Une mention sans identifiant corporate explicite ne devient pas une arête
vers une entreprise homonyme. Comparer cette source au contexte RNE déjà
admis **avant** de parler de nouveauté, puis décrire la couverture query-only
ALL526 sans sélection des ratés ni labels dans la construction. Une future
ablation, ses poids et son admission relèvent d'une décision distincte.

Cette direction ne demande pas d'utiliser les dirigeants naturels. Elle
cherche uniquement des relations corporate historiques publiées que le RNE
actuel ne représenterait plus. Leur présence dans le champ, leur caractère
réellement corporate et leur apport incrémental restent à prouver.
