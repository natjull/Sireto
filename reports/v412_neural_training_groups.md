# V4.12-N — scènes d'apprentissage neuronales

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_neural_training_groups/55b5fa545d29fd26`

- 8 192 scènes issues exclusivement des folds 2, 3 et 4 ;
- 16 candidats par scène : un positif réellement retrouvé et 15 négatifs ;
- 131 072 couples CRM–candidat ;
- négatifs prioritaires : autres établissements du même SIREN, puis premiers
  candidats du retrieval gelé ;
- aucun minage ni choix par XGBoost ;
- aucune injection du positif ;
- folds 0 et 1 absents de l'apprentissage.

Chaque mise à jour modèle comparera simultanément le score du positif aux 15
concurrents de sa scène. Le SIRET ne figure pas dans le texte modèle.
