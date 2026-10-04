# V226 — contrôler la diversité géographique des négatifs appris

Uniquement APRES V225 complete actualexit0/PASS/admission. Aucun nouveau
fit, candidat, label courant ni recall. Les sources TRAIN-only V56/V77
originales restent immuables.512 scènes complètes, IDs disjoints du pilote128,
folds234 seulement, registry officiel original et query IDs de chaque modèle
gelé. Filtrer exactement les parents protégés du held fold dans les négatifs.

Compter la table positive.place / négatif.place par paire, les requêtes sans
contraste place, et le poids place appris. Place signifie INSEE OU CP, pas XY
ni proximité. Cette statistique ne prouve pas à elle seule une causalité,
un mauvais entraînement ou une géographie totalement absente des features.
Il s'agit de tester une hypothèse, pas de la déclarer vraie à l'avance.

Worker propre quiet30/2GiB/80s/pression1/free>=50/un thread sous supervision214,
actualexit0 requis, sources utilisées FULL SHA et epochs pré/post. Lectures
de JSON d'apprentissage seulement, pas moteur ni embeddings ni target parquet.
Refuser mélange fold0/1, overlap128, modèle/registry/scènes contradictoires,
sources modifiées et sortie partielle. Aucun source/poids modifié, aucune
dépense externe. L'objectif initial100 exact99,2% reste non validé par cet audit.
