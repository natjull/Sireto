# Consultation Claude demandée — avis reçu après reconnexion

10 septembre 2026. L'utilisateur demande un avis via `claude -p`, modèle
Claude Opus 5, effort extra high, avant un nouveau run de retrieval, avec
reprise explicite de la piste anciens/nouveaux candidats.

**Mise à jour :** la seconde tentative après reconnexion a réussi, avec
`claude-opus-5` confirmé dans `modelUsage`, effort demandé `xhigh`.
Synthèse et examen critique : `docs/retrieval_claude_opus_review_received.md`.
Aucun nouveau run de retrieval. Le blocage ci-dessous est conservé comme
historique de la première tentative, il n'est plus actif.

## Première tentative — historique

- CLI installé : Claude Code2.1.143. `--effort xhigh` accepté par son aide.
- Modèle demandé explicitement : `claude-opus-5`. Aucun fallback demandé.
- Prompt : `docs/retrieval_claude_opus_review_request.md`, chiffres agrégés,
  contexte technique, contraintes et limites; aucune donnée CRM brute ni
  identifiant d'entreprise/personne transmis dans ce dossier.
- Outils désactivés (`--tools ""`), MCP vide et strict, hooks désactivés,
  aucune persistance de session demandée; exécution dans un répertoire
  temporaire distinct du projet. Aucun accès au dépôt délégué.
- Authentification locale déclarée Claude Max; l'inférence échoue avec401,
  **jeton OAuth expiré**, avant appel du modèle. `modelUsage` vide,
  tokens entrée/sortie0, coût déclaré0, temps total192995 ms.

**Aucun avis de Claude obtenu.** La disponibilité effective d'Opus5 n'a
donc pas été validée par une réponse du service. Ne pas attribuer à Claude
une recommandation ou un accord avec notre plan. Aucun modèle alternatif
appelé, aucun nouveau test de retrieval lancé.

Reprise nécessaire : authentification utilisateur via `claude auth login`,
puis relancer la même demande avec modèle et effort explicitement fixés.
Ne pas chercher, afficher ou modifier manuellement les jetons de connexion.

## Point historique rétabli avant la revue

Ancien/nouveau = provenance de génération, pas âge de l'entreprise ni label.
La suppression de la protection1750 anciens+250 rescue a bien produit
**+45 exacts Master2000 nets en v22** (728→773), puis le parent appris v23
**+49 nets** (773→822), sur le panel de développement biaisé1034.
La fusion v24 n'ajoute qu'un net (+1,6 gains/5 pertes), non promue.
La formulation conversationnelle « aucun gain démontré au Master2000 »
était donc trop catégorique. Ces progrès expérimentaux ne sont pas une
validation nationale, une amélioration produit intégrée ni un Recall100.
Les rapports ne permettent pas à eux seuls d'attribuer chacun de ces gains
au changement de modèle de l'assistant.

Le dossier destiné à Claude demande précisément de ne pas réinventer ces
expériences et de critiquer le critère trop local de rejet d'un canal : les
cinq contrôles perdus par v28 ont chacun leur SIRET exact apporté également
par un canal textuel. Cela reste une preuve de présence dans les sources,
pas une preuve de survie au nouveau Master2000.
