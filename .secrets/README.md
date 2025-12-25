# Secrets (local only)

Ce dossier sert à stocker **localement** des secrets (clés API, tokens) sans jamais les committer.

- Fichier conseillé : `.secrets/keys.env` (non suivi par git)
- Charger les variables d’environnement depuis ce fichier avant exécution (ex. `source .secrets/keys.env`)

Variables attendues :

- `SIRETO_GOOGLE_API_KEY` : clé API Google (Custom Search)
- `SIRETO_GOOGLE_CSE_ID` : identifiant du moteur (cx)
- `SIRETO_BRAVE_API_KEY` : clé API Brave Search

Important :

- Ne jamais committer de secrets.
- Si une clé a été partagée en clair, la révoquer/rotater.
