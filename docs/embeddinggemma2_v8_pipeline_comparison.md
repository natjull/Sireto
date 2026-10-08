# V8 — Comparaison de deux pipelines, pas attribution fine-tuning seule

V7 a été REFUSÉ par son préflight : erreur absolue .005123790353536606,
erreur cosinus .000349581241607666, supérieures aux tolérances préfixées.
Aucun seuil n'est assoupli, aucune génération optimisée admise ni label lu.
Les différences de calcul BF16/padding ne sont pas assimilées à une identité.

L'exécution V8 utilise exactement l'encodeur original lots8, sans changement
de poids ou de textes. Elle teste une nouvelle hypothèse, figée avant labels :
le canal SITE seul (maximum cosinus des vues officielles nom+adresse), plutôt
que le mélange .75site+.25identité. La loss site est devenue très faible,
alors que la loss identité reste élevée ; tester cette preuve indépendamment
est pertinent. Cela économise les encodages d'identité, sans supprimer aucun
site, requête ou variante SITE. Le canal identité est explicitement non mesuré.
Le champ technique v3_official512 désigne les poids/textes de ce modèle, MAIS
le score V8 est déclaré MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND dans le launch,
binding, génération et report ; ne pas le confondre avec le score dual V4/V6.
Ce pipeline officiel512 (checkpoint V6 best1000) est comparé à la baseline
native160 déjà entièrement scellée. Même80requêtes/119064établissements, aucun
filtre de dénominateur, aucune injection. L'ablation intermédiaire
v1_official512 n'est PAS mesurée : les apports des textes enrichis et du
fine-tuning ne pourront PAS être isolés. Ce choix est figé avant labels.

Deux sorties complètes avec ownedexit0, mêmes queries et cap100, ainsi que
lexical national, sont scellées avant labels. Les métriques exact/SIREN/
opérationnel, Wilson, gains/pertes et segments sont publiées. La géographie
V5 scellée est réutilisée sans rescan, même RRF60 et poids BAN faible.5.

Échéance08:43 inchangée. Pas de training supplémentaire, ni de modification
des namespaces V4–V7 historiques. Test4323/folds0/1 et aval gelés. Ce diagnostic
de80requêtes n'est ni national dense, ni OOF complet, ni gate99,2%, ni SLO.
