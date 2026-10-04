# V167 — durée de vie des métadonnées, reprise des seuls lots PASS

Contrat avant mesure nationale, 4 octobre 2026. Le top100 initial V38 et la
politique autorisée V153 restent actifs. Aucun modèle aval modifié, aucun label
ou fold0/1 ouvert, aucun pool candidat créé. Ce travail ne mesure pas le recall.

## Arrêt historique conservé

V166 remaining session75068 est TERMINAL1 au lot49. Worker STOP RSS1121075200
octets,4,346172s, minfree76, pression[1], swap global0. La trace termine dans
le deuxième chargement des plans/audits à la fin du worker, après SQL ; elle
ne constitue pas une instrumentation RSS de chaque instruction SQL.
Les49 lots0..48 ont été relus avec phase_record166 et exits parents confirmés :
2929166 alias,206832290 postings, RSS worker max1072152576 (<1Gio), max6,801799s,
minfree74. Le lot49 arrêté et tous artefacts166 sont immuables, jamais relancés.

## Changement d'implémentation limité

V167 réutilise exactement le kernel166 (4 lecteurs/writers), physical_plan152,
pack134, toutes les sources et les contrôles de scalar/export d'origine.
À l'ouverture, validation complète des plans, code, sources, audits et exits.
En fin de phase SQL, revalidation des mêmes hashes cryptographiques déjà
validés et de CHAQUE epoch source, sans reparsing des grands JSON en doublon.
Les STOP apparus pendant le traitement sont aussi contrôlés. Ce n'est pas un
cache non validé, une réduction des sources ni un changement de score.

La reprise optionnelle épingle launch/plan166 et uniquement des manifests de
lots individuellement PASS, protégés, mêmes slices et paramètres, exit réel0
et budget1Gio/80s. Tout STOP est exclu. Chaque import conserve son chemin et
ses titres historiques ; les fichiers ne sont ni copiés ni réétiquetés.
À chaque utilisation du lot, tous ses artefacts sont rehashés. L'intégrité des
certificats est revérifiée aux phases suivantes, même si les artefacts du lot
ne sont pas des entrées de la nouvelle phase. Aucun index partiel ne devient
complet : les576 lots,256 packs et audits/seals intégraux restent nécessaires.

## Ordre de coût et limites

1. Tests fixtures et corruption/exits/STOP, sans prétention de coût national.
2. INIT frais, import163 SOURCE complet et seuls lots166 PASS dans le budget.
3. Lot49 frais sous supervision164 et garde153 identiques :1Gio,80s,
   pression macOS normale, free>=50%, swap global seulement télémétrie.
4. Seulement si PASS avec exit confirmé : remaining-parts sur tous les lots
   manquants. Arrêt au premier échec ; pas de hausse de budget ou de retry.
5. Les seals/packs ne sont pas lancés automatiquement par remaining-parts.
   Leur coût national doit être vérifié séparément, sans transformer une
   validation d'index en résultat top100 ou en SLO de recherche.

La dernière mesure exacte reste V101114/128=89,0625% sur pilote réutilisé,
pas une validation nationale. Objectif99,2% NON atteint ; noREADY. Les noms
doivent encore tenir sous1Gio avant tout essai du moteur national complet.
