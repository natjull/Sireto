# Résultat technique Perplexity 0,6B — 8 octobre 2026

Essai `probe_20261008_b` terminé, processus own exec session8483 exit0.
Modèle public révision `8fc2de24534aa3610d85fa59c463313a5f096455`, aucun training.
Code protocole `c0cd459b`, reprise dépendances isolées `3255a207`.

MPS natif, fallback CPU désactivé, float32/eager/batch4, torch2.14.1,
transformers5.19.0, sentence-transformers6.1.0. Dépendances image exigées
même pour texte ajoutées UNIQUEMENT à l'overlay Perplexity : Pillow12.3.0,
Torchvision0.29.1. Le runtime Gemma n'a pas été modifié. Premier essai A
STOP au chargement, artefacts conservés. Téléchargement HTTP public (~2,4Go),
transport Xet échoué puis désactivé dans le seul processus de téléchargement.

Sur 4 requêtes query-only et 128 documents officiels de 4 petites communes
(304/1726/937/3992 sites complets dans leurs sources ; seulement 32 textes
hashés par commune utilisés pour cette mesure MATÉRIELLE) :

- Chargement : 5,417s ; warmup 1 document : 1,789s.
- Encodage chaud 128 documents : 17,047s, soit 7,509 textes/s.
- Encodage 4 requêtes : 0,261s, soit 65ms/requête moyen sur ce petit batch.
- MaxSim CPU 512 paires : 20,419ms, oracle scalaire écart maximum0.
- Embeddings finis 128 dimensions/token ; aucune entrée tronquée ; maximum
  longueur brute253tokens, prompts et limites natifs du modèle.
- Moyenne69,836 vecteurs/document ; 4 576 768octets de vecteurs float32,
  soit35 756octets/document avant compression/dédup/index.
- Mémoire driver MPS observée max3 672 686 592octets (~3,42Gio), allocation
  courante MPS max2 389 551 616octets ; RSS~739Mo. vmmap physique4,4Gio,
  pic4,5Gio : RSS seul sous-estime nettement les buffers GPU.

La bibliothèque emploie les implémentations PyTorch de référence de la
convolution et gated-delta : correctes mais moins optimisées que les kernels
CUDA spécialisés. Aucun fallback d'opération MPS vers CPU autorisé. Scoring
MaxSim délibérément sur CPU. Pas d'installation de kernels CUDA sur Mac.

Débit mesuré EN CONCURRENCE avec Gemma, batch4, textes courts ; ce n'est ni un
débit isolé, ni une comparaison équitable des deux encodeurs, ni un SLO, ni
une ETA valable pour tous les2,55Msites. Score/latence d'une commune dense
complète et passage à l'échelle ANN/compression restent non mesurés.

AUCUN recall évalué, aucun label ouvert, aucun positif injecté, aucun modèle
aval/serving modifié. Les4requêtes/128distracteurs de cet essai ne peuvent
servir à revendiquer une qualité de retrieval. Prochaine étude utile : canal
Perplexity initial100 exact sur des communes complètes et fusion avec les
canaux lexical/géo sur mêmes requêtes ; protocole/corpus scellés avant labels.
Aucune passe Perplexity2,55Msites ni fine-tuning lancé implicitement.

Artefacts : `/Volumes/CATNAT_DATA/SIRETO_PPLX/probe_20261008_b/`
(`inputs.json`, `state.json`, `report.json`, `memory.json`).
Après arrêt Perplexity, Gemma V15 toujours vivant : controller27102,
worker59799 zone33013, progression horodatée1791478106.127674, aucun signal
envoyé au run ni code/config hashé modifié. Deux tests unitaires PASS.
