# V80 — Faisabilité des chemins de succession liés au site

Objectif initial100 exact≥99,2% inchangé, modèles aval gelés. V79 n'a
sélectionné aucun ajout géographique ; ne pas refaire un pilote identique.
Les relations SIRENE sont un apport de preuves distinct, pas une nouvelle
grille de poids sur les mêmes alias. V39 a validé les adaptateurs mais pas
la compatibilité de site ni le moteur initial des successions.

Auditer le stock brut national, non les seules projections par préfixe.
Comptages exhaustifs des lignes, continuité économique explicite, identifiants
valides, dates présentes. Une ligne n'est pas un événement dédupliqué. Pour
le rattachement au site, échantillon FIXE de64 paires à continuité vraie,
sélection par SHA256 des deux SIRET, indépendant de tout CRM et label.
Tous les records/dates de chaque paire choisie conservés dans les comptes.
Le diagnostic de64 paires n'est PAS un corpus de recherche, une couverture
nationale des noms, ni un estimateur du recall CRM.

Joindre les deux endpoints au catalogue officiel courant, sans lookup humain.
Identifiants uniquement jointure/groupement/tie. Mesurer présence, même SIREN,
même adresse normalisée nonvide avec accord INSEE ou CP nonvides, et sièges
transférés. Géographie et adresse COURANTES ne prouvent PAS un site historique
à la date de succession. Une adresse différente n'autorise pas la propagation
d'un nom de site vers tout le SIREN, ni une exclusion de requête. La continuité
économique seule ne suffit pas à assimiler deux sites ou identités.

Fichiers complets et manifeste entièrement SHA-vérifiés avant/après ; source
canonique43,9m conservée, pas de sélection par erreurs V78. DuckDB512MB,
2threads, pas de spill, worker80s/2Gio/zéro swapout, supervision90s. Aucun
index national construit, modèle appris, candidat produit, label ouvert,
résultat après reranker, service externe, fold0 ou fold1. Le prochain build
devra garder noms d'ancrage, direction, relation, date et adresse/portée
distincts, puis une recherche native à100 sans seed/pool parent fixe.
