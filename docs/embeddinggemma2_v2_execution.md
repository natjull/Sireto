# V2 — données préparées et paramètres d'exécution

8 octobre2026 (travail commencé7octobre). Milestone données `8f841fb0`,
contrat immuable `docs/embeddinggemma2_v2_contract.md`. Namespace données :
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v2/data_20261007_a`.

Préparation réellement complète exit0 en168,24s :176 497 sites natifs officiels
des SIREN TRAIN/validation,208 031documents dont négatifs BM25 officiels
historiques,208 385vues site,42 064vues identité sur tout le dictionnaireV130
64buckets hashés. Ce n'est pas le corpus d'une évaluation retrieval et ne
réduit pas le dénominateur d'un futur test. TRAIN15 035positifs exploitables,
1positif absent séparément ; validation2 212.1 964TRAIN/281validation avec
un négatif de mêmeSIREN strictement prouvé autre site. Aucun sidecar d'évaluation
consommé pour obtenir ces preuves, aucun label corrigé.

Les documents conservent les textes officiels complets ; chaque vue encodable
<=512tokens avec validation tokenizer avant fit. Aucun truncation=True.
Nom légal du parent et nom de sibling ne sont pas confondus avec une enseigne
du SIRET cible. Les champs marque/succession manquants ne sont pas inventés.
BAN/XY non injectés en texte ; intégration numérique géographique encore due.

Plan pondéré :22 127exemples par epoch pour15 035queries. Répétitions autant
que nécessaire pour couvrir les vues site et entreprise, poids inverse du
nombre de vues afin de ne pas surpondérer les entreprises riches en alias.
Max3vues site positives,213vues identité pour un parent exceptionnel. La
couverture réellement exposée, y compris en cas budget, est publiée. Arrêt
anticipé seulement après au moins un epoch complet puis3 contrôles stagnants ;
le checkpoint sélectionné peut précéder l'exposition complète : son payload
exposed garde la trace, ne pas prétendre toutes vues apprises sans vérification.

Minage statique avec adapter V1 choisi7200, sur banque TRAIN uniquement et
première vue complète de chaque site, dans commune CRM.117 538sites sont
géographiquement atteignables par cette recette (179 825dans la banque complète).
Les lignes hors communes interrogées et les vieux négatifs sans INSEE ne
pouvaient jamais être choisis : ne pas dépenser le GPU pour les encoder.
Pas ANN/oracle ni vérité ajoutée à une évaluation. Max4négatifs explicites :
deuxlexicaux, un dense, un sibling prouvé lorsque disponible ; compléter
avec lexical lorsque canal indisponible. Relations natives incertaines masquées,
jamais positives d'identité.

Paramètres choisis sans grid : reprise rank8/alpha16 V1, base geléeBF16,
vrai batch16/micro2, loss75%site/25%identité, température.05, LR3e-5,
warmup3% puis cosine10%LR, decay.01/clip1,2epochs/8h max. MémoireGPU
cap.25/RSS5Gio, pression libre65→50→45% et retriesresource-only12.
Pas d'augmentation arbitraire rank/contexte8k/epochs : priorité données,
négatifs et objectif. Pas de claim optimum ou 99,2% garanti.

Pipeline séquentiel owned : minage complet exit0 → prepare → preflight
réel deuxpas incluant les vues les plus longues et deuxobjectifs → training.
Le preflight ne fournit PAS de checkpoint utilisé par training. Stop si
un stage échoue. Runtime/CATNAT/Mac seuls, pas cloud, test4 323/folds0/1 fermés,
aval inchangé. Septtests PASS dont gradient FP64 pondéré cached/direct,
protection mêmesite/suffixe/histoire, couverture de tokens/vues et IDs Unicode.
Les futures comparaisons devront séparer gain des données et gain des poids :
V1 anciens textes /V1 nouveaux textes /V2 nouveaux textes, vrais distracteurs,
initial<=100 avant aval, jamais métrique du corpus TRAIN comme recall.
