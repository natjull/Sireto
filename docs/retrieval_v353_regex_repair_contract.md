# V353–V355 — correction exacte de compilation, pas nouvelle recette métier

Le run V352/V349 du 6 octobre 2026 s'est arrêté sur la position 389 après
97 checkpoints complets (388 sorties). Le lot 97 incomplet n'est pas admis.
Erreur inconnue/fatale conservée, aucun redémarrage de son STOP terminal.

La règle nominale V256 « SESSAD » contient 31 partitions. La dernière excède
les 1 000 états du compilateur DFA natif tantivy 0.25.1. Les ressources ont
passé ; ni RAM ni limite de temps n'expliquent cette exception.

V353 tente d'abord la construction RegexQuery d'origine. UNIQUEMENT après
cette erreur de taille exacte, les groupes d'alternatives littérales ASCII
sont factorisés par suffixe : `(XS|YS) = (X|Y)S`. Chaque groupe d'origine reste
englobé pour conserver la portée des quantificateurs, espaces compris.
Pas de suppression de partition, raccourcissement, nouveau dictionnaire,
feature, filtre géographique ou identifiant de similarité. Le langage reconnu,
le RegexQuery natif, le score constant, les preuves et le collecteur unique
direct100 restent identiques. Les expressions qui compilent déjà ne changent
pas. Échec de la réparation/autre erreur : fatal, jamais score nul artificiel.
Limite DFA native non augmentée, aucune dépendance installée/index reconstruit.

V354 crée une fonction privée avec le code du worker V344 inchangé et deux
liaisons explicites : validation du lancement neuf et recherche V353. Aucun
global/module/fichier historique n'est modifié. Le lancement gèle TOUS les
anciens et nouveaux fichiers. Les producteurs conservent les mêmes preuves
exit0 propre/PASS/epochs/hashes pré-post, ordre et 100 SIRET maximum.

V355 vérifie sans labels la compilation des deux grammaires sur les 2 000
requêtes, puis exécute un SEUL lot réel dans un dossier neuf. Il ne lit/importe
aucun ancien payload/checkpoint, ne réécrit aucun STOP et ne permet aucune
évaluation. Son résultat n'est ni le run 2 000 complet ni un recall/SLO.
Les tests synthétiques prouvent des sémantiques de moteur, pas le recall métier.

Ressources inchangées V342 : 3 Gio RSS, 80 s worker, 20 s requête incluant
l'ouverture index, quiet30, libre≥50%, pression1, un worker/thread. Modèles
aval gelés, folds0/1 fermés, panel2/3/4 query-only inchangé. Cible métier
99,2% recall SIRET humain exact dans les 100 AVANT aval, exact/op séparés.

Une nouvelle orchestration complète doit geler V354 comme worker, reprendre
les contrôles V349/V352 et revalider TOUS les producteurs avant le GT. V355
ne modifie pas l'interdiction actuelle d'importer des fragments d'un STOP ;
une éventuelle migration des checkpoints complets nécessite un protocole
explicite avec provenance et preuve de parité, jamais une suppression de STOP.
