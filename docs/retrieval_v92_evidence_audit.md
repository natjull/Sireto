# V92 — même audit label-side, lots physiques bornés

Le premier audit V91 a STOP_RESOURCE_POLICY à80s pendant une requête
ponctuelle native, sans rapport128 scellé. Aucune contradiction ni diagnostic
de pertes ne peut être déduit de cet arrêt. L'ancien code et son statut
restent immuables.

V92 conserve tous les contrôles et les128 requêtes du contrat d'audit V91,
ainsi que la même requête et les mêmes scores serving V91. Seul le découpage
physique change : huit lots16 successifs, chacun80s/6Gio/zéro swapout,
supervisé90s, contrôleur900s. Pas d'augmentation du budget d'un processus.
Ce n'est pas un corpus16, ni une mesure partielle : le rapport n'est scellé
qu'après concaténation128 alignée exactement, intégrité des références,
ressources et hits identiques à l'évaluation précédente. Les sorties label-side
ne sont jamais utilisées par la génération. Aucun fold0/1 ni fit ni nouveau
choix de poids, aucune dépense externe. Les logs permettent de situer un
éventuel autre arrêt sans devoir supposer le processus encore actif.
