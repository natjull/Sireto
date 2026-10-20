# Jugement des labels — avenant v2 préenregistré avant tirage

## Statut et périmètre

Le criblage `c8b9e405` reste inchangé : 17427 G0, folds population2/3/4, FORT95/FAIBLE50/AUCUN17282, SUR4459. La fiche v1 est archivée sur place, NON UTILISÉE ; aucun de ses fichiers n'est modifié. Sources locales officielles et artefacts G0 scellés uniquement, aucun apprentissage/web/jeu fermé/VM. Écritures : ce nouvel avenant et son JSON, une ligne append-only handover, namespace neuf `$H/label_noise_review_20261020/v2/`. Aucun fichier tiers/pending touché. Mémoire gardée à3,75Gio, DuckDB1thread/512Mio sans spill, longs runs détachés avec journal.

## Tirage et options préenregistrés

100 dossiers sans remise : **50 FORT,25 FAIBLE,25 AUCUN**. Listes d'indices G0 croissants ; `random.Random(20261023).shuffle` successivement FORT/FAIBLE/AUCUN. Parcourir chaque permutation jusqu'au quota, en sautant uniquement les dossiers sans alternative publique nommée non UNION ou indiscernables sur tous les champs affichés. Publier tous les effectifs éligibles/exclus et le nombre de remplacements rencontrés dans les préfixes tirés ; pas de transfert de quota. Strate insuffisante : STOP.

FORT/FAIBLE : concurrent figé du criblage. AUCUN : parmi les100 candidats G0 publics disposant d'un nom officiel, choisir le **maximum Jaro-Winkler** du nom CRM, sous masque UNION=False et SIRET différent du label ; égalités départagées par SIRET lexical uniquement. Aucun score/rang modèle ni préférence actif n'intervient dans ce choix. Pas de substitution du deuxième concurrent si la paire choisie est indiscernable : passer au dossier suivant de la strate.

Champs montrés identiquement dans CSV et HTML : dénomination, enseignes/autres noms officiels, état, siège/secondaire, création, fermeture administrative, numéro/suffixe et voie séparés, complément, code postal/commune, distance typée au point CRM/qualité, NAF. Date de fermeture = dernier début de période historique F avec changement d'état attesté, uniquement si le stock courant est F ; inconnue sinon, sans assimiler arbitrairement dateDebut à une fermeture. Pour un actif : sans objet. Données absentes explicitement inconnues, jamais zéro. Distances affichées à0,01m. La comparaison d'indiscernabilité porte sur ces valeurs réellement affichées après harmonisation casse/espaces, pas les identifiants cachés.

Ordre des fiches : graine20261024 ; ordre des options : graine20261025 ; IDs opaques : graine20261026. Aucun identifiant SIREN/SIRET/query, strate, rôle, score ni provenance modèle dans les supports juge. Clé séparée `v2/private/review_key.json`, dossier0700/fichier0600, SHA256 public. Un autre compte ou le partage des seuls HTML/CSV est nécessaire : chmod ne protège pas du compte propriétaire.

## Poids et estimation inchangée, exclusions explicites

Population N=17427 et poids de strates N_h/N inchangés :95/17427,50/17427,17282/17427. Recenser l'éligibilité sur toute la population AVANT tirage. Si exclusions, ne pas extrapoler silencieusement les répondants éligibles aux exclus : E_h/n_h est le poids d'expansion des éligibles ; les N_h−E_h exclus restent inconnus dans le dénominateur17427, avec bornes0/1. Publier N_h,E_h,n_h, poids d'expansion et probabilités d'inclusion, ainsi que SUR éligibles/exclus. Sans exclusion, cela reproduit exactement N_h/n_h.

Réponses : Option 1 / Option 2 / Les deux / Aucune / Je ne sais pas. Primaire Y=1 si concurrent seul choisi,0 si label choisi/Les deux/Aucune ; Aucune est publiée séparément comme rejet des deux, pas confondue avec un concurrent choisi. Non-réponse/JNS = inconnus, jamais suppression de dénominateur. Le taux primaire est un proxy de label non-occupant par choix concurrent, pas une mesure exhaustive. Publier aussi le rejet explicite du label (concurrent seul OU Aucune), clairement secondaire. Pour la précision SUR occupant : Horvitz–Thompson sur les4459 SUR, Z=1 si top1 est une option reconnue occupante,0 si montré mais rejeté ; inconnu si non montré/JNS/manquant. Laisser la précision hors[0,1] éventuelle de l'estimateur HT brute signalée plutôt que la masquer. Bornes avec inconnus, pas de point fabriqué si non résolu.

Bootstrap stratifié10000, graine20261023, rééchantillonnage de dossiers dans chaque strate, poids fixes ; strate entièrement recensée fixée. Percentiles95%, pour les deux extrémités si inconnus. Aucune modification de label/statut/pool. Toujours publier les références officielles à côté : UNION4360/4459=97,7798%, exact4151/4459=93,0926%. `score_review.py` refuse IDs/réponses invalides, doublons, clés incohérentes et données sans aucune réponse ; les réponses manquantes deviennent inconnues. Valider le script avec réponses fictives aléatoires dans un répertoire temporaire privé puis supprimer uniquement ces fichiers de simulation, jamais les publier comme résultats.

## Application locale et mesure de temps

`jugement.html` : un seul fichier autonome sans ressource réseau, CSP bloquant les connexions ; lecture d'un dossier à la fois, gros caractères,5 boutons, commentaire, précédent/suivant, progression, sauvegarde locale avec avertissement si indisponible, export CSV UTF-8 des100 IDs (réponses vides si non jugées). Gestion des champs comme texte, échappement CSV/formules, pas d'injection HTML. Le navigateur ne reçoit jamais la clé. Aucun télémétrage/envoi ; localStorage contient des données locales sensibles, ne pas utiliser une session partagée. Tester navigation, sauvegarde/rechargement et export sur la page réelle avec WebKit local si disponible, sans dépendance réseau.

Lecture chronométrée de5 fiches choisies au hasard, graine20261027, depuis le support juge uniquement, sans ouvrir la clé pendant cette lecture. Rapporter les secondes réellement écoulées et l'extrapolation100 fiches, en distinguant **lecture agent** de **jugement humain Nathan**, non mesuré ; aucune durée humaine inventée. Les lectures de l'agent ne remplissent pas les réponses réelles.

## Exécution et résultats

### Livraison et archivage

Avenant commité AVANT tirage : **`b7379b48cc18e11bb21ff3126d1d0317416b94e0`** ; SHA256 protocole `ef62d1541a1bb79df1af01032429d227e0a84cbf76304f5eea9c8c0880deca7c`. V1 archivée sur place, non utilisée et SHA inchangés. Aucun label, pool, statut, modèle, jeu fermé, fichier tiers/pending ou VM modifié.

**100 dossiers :50 FORT,25 FAIBLE,25 AUCUN ;100 paires discernables ;0 remplacement dans les préfixes tirés.** Une paire indiscernable existe dans toute la population AUCUN, mais ne figurait pas dans le préfixe tiré. Elle reste inconnue dans le dénominateur17427 ; pas d'extrapolation silencieuse des éligibles à cette ligne.

| Strate | N population | E éligibles | n revue | Poids N/17427 | Expansion E/n | Inclusion n/E |
|---|---:|---:|---:|---:|---:|---:|
| FORT |95|95|50|0,0054513112|1,9|0,5263157895|
| FAIBLE |50|50|25|0,0028691111|2|0,5|
| AUCUN |17282|17281|25|0,9916795777|691,24|0,0014466755|

Aucun dossier exclu faute de concurrent public nommé ; une seule exclusion d'indiscernabilité, hors SUR. L'incertitude de cette ligne reste dans les sorties du scorer, même si les100 fiches sont entièrement renseignées. La strate AUCUN domine le poids ;25 témoins n'autorisent aucune garantie de précision élevée.

### Ouvrir et remplir

**HTML autonome** : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/label_noise_review_20261020/v2/jugement.html`.

CSV alternatif : même dossier, `review_sheet.csv`. Le HTML embarque les100 fiches, sans ressource réseau ; aucune dépendance à côté n'est nécessaire pour le juge. Gros caractères système20px, un dossier à la fois, cinq choix, commentaire facultatif, précédent/suivant, progression, sauvegarde locale et avertissement en cas d'échec. L'export produit `reponses.csv`,100 IDs opaques, réponses vides conservées. Les100 lignes livrées restent **sans aucune réponse humaine**.

**Partager uniquement le HTML avec Nathan**, jamais `private/`, la clé, les codes de construction ou les documents méthodologiques permettant une reconstruction des rôles. Le CSV peut servir d'alternative. La clé est protégée0700/0600 contre les autres comptes, pas contre son propriétaire. Les réponses du navigateur restent locales : exporter avant de changer de session ou d'effacer ses données.

- SHA256 CSV : `f17dd1414792bf2d29e3ec182bc27d3ec70013f95c77520b86a762bbaef70a01`.
- SHA256 HTML : `3d998cd2fdcdd13f4b77cca4d4ae7e50438e69f0dd3b763036998253aad70404`.
- Clé : `v2/private/review_key.json`, SHA256 **`a691b688c377b20d593b79ed15039b7148078703315adf7ade17daf4853157b7`**. Aucune correspondance option/label publiée dans Git.

### Scorer le retour humain

```bash
H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$H/label_noise_review_20261020/v2/score_review.py" \
  "$HOME/Downloads/reponses.csv" \
  "$H/label_noise_review_20261020/v2/private/review_key.json" \
  --out "$H/label_noise_review_20261020/v2/score_humain.json"
```

Le scorer vérifie la clé contre le SHA du manifeste adjacent, valide IDs/réponses/doublons, refuse un retour vide et refuse d'écraser une sortie existante. Il conserve les manquants dans les100 dossiers du plan, utilise E/n et garde l'exclu inconnu. Sortie en fractions : proxy concurrent choisi, rejet explicite du label secondaire incluant Aucune, rejet des deux, vue précision SUR occupant HT avec inconnus, IC bootstrap10000 et références officielles côte à côte. Aucun point si des inconnus restent ; bornes/IC des extrémités à la place. Les extrémités HT SUR peuvent sortir de[0,1] par variabilité du domaine : elles sont signalées brutes, sans masquer le problème. Ce ne sont ni des corrections de vérité ni de nouvelles précisions officielles.

Les références restent4360/4459 UNION et4151/4459 exact. Aucun taux de bruit humain ou gain de précision n'est calculé avant réponse. Les sources du scorer, constructeur et gabarit sont archivées dans le JSON de cet avenant ; la clé et les fiches individuelles ne le sont pas.

### Vérifications réelles et ressources

- `build_review.py` : recensement d'éligibilité des17427 dossiers ; génération finale57,06s, pic RSS1,890Gio (<4). Source de fermeture : historique officiel d'état, pas simple date de dernière modification.
- `verify_pairs.py` :100 lignes/options/IDs/ordre de tirage reproduits,100 paires distinctes,25 meilleurs concurrents AUCUN rejoués sur2500 candidats,200 NAF/dates contrôlés,18 références SHA recontrôlées, clé0700/0600, égalité des données HTML/CSV et v1 inchangée.
- `smoke_browser.py` : **WebKit macOS réel**, clics sur choix/navigation, sauvegarde et rechargement, export réel du Blob `reponses.csv` contenant100 lignes, guillemets/retours ligne/formules neutralisés vérifiés. Magasin navigateur éphémère nettoyé ; aucune réponse de simulation dans le navigateur de Nathan.
- `smoke_score.py` :100 réponses fictives aléatoires,8 scénarios CLI (inconnus, déterminés, partiel, vide, doublon, ID invalide, réponse invalide, mauvais SHA), formule pondérée recomposée indépendamment,10000 bootstraps, dénominateurs et métriques officielles inchangés. Fichiers de simulation supprimés automatiquement ; **aucune estimation fictive conservée ou publiée comme résultat**.
- Une première génération omettait la colonne NAF dans l'hydratation : correction de plomberie et génération complète refaite, mêmes règles et mêmes100 dossiers. Ancienne génération conservée dans `v2/private/attempt1/`, jamais utilisée. Le smoke WebKit a aussi nécessité un correctif de compilation du helper et l'acceptation du BOM décodé ; pas de défaut d'export dans la page.

Reçus dans `v2/manifest.json`, `pair_verification.json`, `browser_smoke.json`, `score_smoke.json` ; logs détachés dans le même namespace. Aucune suite de tests du dépôt lancée.

### Temps effectivement observé, sans clé

Cinq fiches tirées avec graine20261027, lues depuis le CSV juge seulement, sans ouvrir la clé pendant la lecture ; aucune réponse réelle remplie. Première lecture29,74s, soit5,95s/fiche ; relecture de la version finale enrichie NAF26,56s, soit**5,31s/fiche**, extrapolation mécanique **8,85min/100**. Ce sont des temps muraux de **lecture agent**, avec latence d'outil et familiarité à la relecture, PAS des temps de jugement humain. Le temps Nathan n'est pas mesuré : ne pas lui promettre9minutes. Détails et horodatages dans `v2/reading_time.json`.

Commit résultats auto-référencé par `git log -1 --format=%H -- docs/label_noise_review_v2.json`. Aucun push. Reste : retour humain des100 fiches, puis scorer sans changer aucun label ni résultat officiel.
