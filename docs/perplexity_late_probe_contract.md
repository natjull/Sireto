# Perplexity late-interaction — essai technique Mac du 8 octobre 2026

Autorisation utilisateur « Tente le » : tester le modèle public 0,6B sur le
Mac, en parallèle du pilote Gemma, sans mutation de ce pilote ou de son runtime.
Ce premier essai n'est PAS un benchmark de recall ni un corpus représentatif.

Modèle `perplexity-ai/pplx-embed-v2-late-0.6b`, révision immuable
`8fc2de24534aa3610d85fa59c463313a5f096455`, poids publics float32, attention eager,
MPS sans fallback CPU, prompts query/document natifs du modèle et MaxSim natif.
Aucun remote code, installation/mise à jour du runtime ACTIF, fine-tuning,
quantification ni changement du serving ou des modèles aval.

Sources : panel query-only V15 folds2/3/4 et documents officiels entièrement
préparés/scellés avec owned exit0. Quatre communes déjà préparées de 1 à 10000
établissements choisies par hash fixe, jusqu'à 4 requêtes et 32 documents par
commune, sélection documentaire hashée indépendamment des labels. Textes :
noms légaux parent, propres noms/enseignes historiques du site et ses adresses.
Identifiants uniquement pour sélection technique reproductible, jamais dans
les textes. Aucune lecture des sorties candidats, labels, folds0/1 ou test4323.
Le petit nombre de distracteurs ne permet AUCUNE affirmation de recall.

Mesurer chargement, warmup, encodage chaud batch4 (128 documents au plus),
nombre de vecteurs/token et octets d'index float32, requêtes et MaxSim CPU.
Vérifier finitude/dimension128 et concordance MaxSim avec un oracle scalaire.
Vérifier absence de troncature avant encodage. RSS, compteurs MPS et empreinte
physique vmmap séparés ; ce sont observations locales, pas garantie de pic.

Admission mémoire réelle free>=30/pression0–2, retrait de ce seul essai si
free<20, pression anormale ou mémoire driver MPS>6Gio. Essai technique borné
180s entre opérations ; pas un budget du run Gemma ni un watchdog de ce run.
Téléchargement public sur CATNAT (~2,4Go), pas de dépenses externes. Aucun signal
envoyé au run Gemma, aucun fichier actif modifié. Erreurs/artefacts conservés.

Si faisable, prochaine étape à décider : benchmark apparié contre les mêmes
communes COMPLÈTES et mêmes requêtes que Gemma, recherche initiale<=100, avec
lexical/géo puis fusion scellée avant labels. Pas de rerank d'un pool Gemma.

Source primaire : https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b

Premier essai `probe_20261008_a` STOP avant inférence : processeur multimodal
exige Pillow/Torchvision absents du runtime. Reprise `probe_20261008_b` avec
overlay CATNAT séparé `vision_overlay`, Pillow12.3.0/Torchvision0.29.1 installés
sans dépendances ; torch2.14.1/transformers/ST originaux inchangés. PYTHONPATH
de CE processus uniquement. Aucune étape de modèle/corpus/labels changée,
premiers artefacts conservés. Résultat à vérifier, pas présomption MPS.
