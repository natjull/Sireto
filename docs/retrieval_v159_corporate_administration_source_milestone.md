# V159 — qualificatif juridique parenthétique, source scellée avant labels

Nouvelle grammaire scientifique pré-déclarée au commit `3104884a`, avant tout
label V158/V159 : une seule parenthèse finale dont tout le contenu matche les
formes juridiques corporate déjà figées. Les autres parenthèses, listes,
mentions naturelles, contacts/adresses et noms non résolus restent exclus.
Même rôle-colon, résolution exacte unique dans les unités corporate publiques,
cutoff, RNE-pair dedup conservateur, degré global, IDF et MAX que V158 B.

Root : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v159/parenthetical_corporate_administration_20261014_night`.
Session `94528`, chunk final `434cc7`, exit tool réel 0 ; enfant `68515`, owned
exit 0. Durée 695,541 s ; pic RSS 1 369 899 008 octets. Tous les 19 fichiers
canoniques lus et hashés ; 50 380 026 lignes, dont 50 141 539 avant cutoff de
publication 2026-08-01. Aucun nouveau retrieval, modèle, index ou serving.

Generation SHA :
`48b4e796b76c03663338e5a52d04988062609575332cb8df5a63a78aea560e96`.

## Résultats source-only

| Mesure | V158 B strict | V159 qualificatif légal seul |
|---|---:|---:|
| Lignes distinctes sujet/acteur/nom/rôle/origine applicable | 25 076 | 110 784 |
| Sujets corporate publics | 23 362 | 82 653 |
| Acteurs corporate publics résolus | 10 492 | 25 168 |
| Lignes dont paire sujet/acteur déjà dans RNE | 14 118 | 53 004 |
| Lignes dont paire sujet/acteur non présente dans RNE | 10 958 | 57 780 |
| Noms avec résolution exacte ambiguë exclus | 8 911 | 17 164 |
| Requêtes ALL526 avec contexte brut | 290 | 367 |
| Requêtes ALL526 avec contexte de paires absentes RNE | 248 | 320 |

V159 a identifié avant résolution 269 832 clauses avec uniquement un
qualificatif juridique parenthétique final, 65 499 préfixes légaux et
43 378 suffixes légaux simples. Ce sont des occurrences de clauses, pas des
noms, entreprises, requêtes ou liens uniques. Les compteurs de rejet portent
aussi sur les clauses : ils peuvent dépasser le nombre d'annonces.

Le champ `qualification_origin` est conservé dans les relations et la
provenance. Date de publication, édition et type/famille disponibles sont
conservés, sans inventer la date d'effet ni les types absents des exports.

## Limites inchangées

Cette grammaire reste volontairement non exhaustive du texte administration.
Une mention textuelle corporate ne certifie pas la propriété, une fonction
actuelle ou l'identité/opérateur d'un site. Les identifiants servent seulement
aux joins techniques. Les noms personnels et textes administratifs bruts ne
sont pas des exports de preuve ni des features.

La dédup RNE est par paire, tous rôles confondus : « nouveau » signifie absent
de cette table de relations, pas nécessairement absent du contexte lexical
global. L'égalité de nom exacte est évaluée parmi les corporate publics actuels
et leurs alias officiels admis ; ce pool d'alias n'est pas certifié as-of cutoff.
Le degré global V159 peut modifier le niveau de confiance d'un acteur déjà
présent dans V158 : comparer les sources, pas soustraire leurs confidences
comme si elles partageaient le même univers de relations.

Les deux sources sont désormais complètes et scellées avec leurs exits réels
0, **avant toute lecture label-side**. Un diagnostic ALL526 séparé pourra
mesurer la couverture du propriétaire humain seulement après accord racine.
Aucun gain de recall ni aucune généralisation de performance n'est établi ici.
