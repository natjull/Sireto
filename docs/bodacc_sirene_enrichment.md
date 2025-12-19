# Enrichissement d’une base SIRENE locale avec les annonces du BODACC

## Objectif

Construire, à partir des annonces du BODACC (jeu `annonces-commerciales` sur Opendatasoft), un enrichissement local de la base SIRENE permettant de :

- Cartographier les liens entre unités légales (SIREN) : rachats de fonds, fusions, TUP, apports partiels d’actif, etc.
- Améliorer la recherche par “groupe / maison mère” (ex. retrouver SERF à Décines quand on cherche STRYKER).
- Suivre des événements structurants (ventes d’établissements, transferts, procédures, dissolutions) en les rattachant aux SIREN SIRENE.

Cas emblématique : annonce du 18/02/2025 où STRYKER FRANCE (SIREN `333710275`) achète un établissement secondaire de SERF (SIREN `973504509`) à Décines-Charpieu, avec tous les champs structurés (`listepersonnes`, `listeprecedentproprietaire`, `listeetablissements`, `acte`).

---

## Champs BODACC pertinents

Dataset : `annonces-commerciales` (API Opendatasoft).

Champs particulièrement utiles pour le chaînage avec SIRENE :

- **Identification des parties**
  - `listepersonnes` : JSON sérialisé décrivant le déclarant (souvent l’acheteur)  
    - `personne.numeroImmatriculation.numeroIdentification` → SIREN
    - `typePersonne` (`pp`/`pm`), `denomination` ou `nom`/`prenom`
  - `listeprecedentproprietaire` : ancien propriétaire du fonds (cédant)
  - `listeprecedentexploitant` : ancien exploitant (location-gérance, etc.)
  - `registre` : liste/mélange des SIREN/inscriptions RCS (souvent plusieurs SIREN)  

- **Description du fonds / établissement**
  - `listeetablissements.etablissement` :
    - `origineFonds` → texte incluant le prix / nature de l’opération
    - `qualiteEtablissement` → principal / secondaire, etc.
    - `activite` → description libre de l’activité
    - `adresse` → `numeroVoie`, `typeVoie`, `nomVoie`, `codePostal`, `ville`

- **Typage et contexte de l’opération**
  - `familleavis_lib` : ex. *Ventes et cessions*, *Modifications diverses*, *Procédures collectives*, *Créations*, *Radiations*.
  - `acte` :
    - `vente.categorieVente` (ex. “Achat d’un fonds par une personne morale”, “Mise en activité d’une société suite à achat”)
    - `descriptif` : texte libre (parfois “fusion”, “TUP”, “apport partiel d’actif”)
    - `vente.publiciteLegale.date`, `dateCommencementActivite`
  - `modificationsgenerales` : texte libre sur les modifications diverses (fusons/TUP, changement d’actionnaire, etc.)
  - `jugement` : pour les procédures collectives

- **Métadonnées**
  - `dateparution` : date officielle BODACC
  - `id`, `url_complete` : identifiant et URL pour traçabilité

---

## Stratégie d’échantillonnage

Pour ne pas concevoir le parseur “à l’aveugle”, un échantillonnage systématique a été réalisé :

- **Années analysées** : 2010, 2013, 2016, 2019, 2022, 2024, 2025
- **Familles d’avis** :
  - `Ventes et cessions`
  - `Modifications diverses`
  - `Procédures collectives`
  - `Radiations`
  - `Créations`
- **Taille** : 200 annonces par combinaison (année × famille) ≈ 7 × 5 × 200 = 7 000 annonces.
- **Indicateurs calculés par famille/année** :
  - `% acq` : % d’annonces avec SIREN acquéreur dans `listepersonnes`
  - `% counter` : % avec SIREN contrepartie structurée (`listeprecedentproprietaire` ou `listeprecedentexploitant`)
  - `% two_struct` : % avec 2 SIREN structurés (acheteur + cédant)
  - `% two_any` : % avec ≥ 2 SIREN au total (structuré + `registre` + texte)
  - `% text_second` : % sans contrepartie structurée, mais avec 2 SIREN détectés grâce au registre/texte
  - `% addr` : % avec `listeetablissements` présent
  - `% amount` : % avec une mention d’`origineFonds` (montant) dans `listeetablissements`

---

## Constats principaux

### 1. Ventes et cessions : or massif pour les liens inter‑SIREN

Sur les années échantillonnées :

- **SIREN acquéreur structuré (`listepersonnes`)** : 90–98% des annonces.
- **SIREN cédant structuré (`listeprecedentproprietaire` / `listeprecedentexploitant`)** : ~80–92%.
- **Deux SIREN structurés (acheteur + cédant)** (`two_struct`) :  
  ≈ 77–83% selon l’année (2010–2025).
- **Deux SIREN au total (en incluant `registre` + texte)** (`two_any`) :  
  ≈ 83–93%, avec un pic >90% à partir de 2019.
- **Adresse & montant** (`listeetablissements`) :
  - Adresse présente dans 80–99% des cas.
  - Montant (via `origineFonds`) présent dès lors que l’adresse l’est, avec un léger décrochage en 2024–2025 (environ 2/3 des cas).

> Conclusion : les avis *Ventes et cessions* permettent, dans ~80% des cas, de créer un lien inter‑SIREN “acheteur ↔ cédant” purement avec les champs structurés, et jusqu’à ~90% en exploitant aussi `registre` et le texte.

### 2. Modifications diverses : essentiel pour fusions/TUP, mais peu structuré

- Toujours 1 SIREN structuré (l’entité concernée, via `listepersonnes`).
- Quasi jamais de second SIREN structuré (`counter` ≈ 0%).
- Second SIREN détectable dans le texte :
  - `two_any` ≈ 0–2.5% selon les années, via `modificationsgenerales` / `acte` / `registre`.
- Le texte contient cependant les mots-clés clés : “fusion-absorption”, “transmission universelle de patrimoine”, “apport partiel d’actif”, “cession de parts”, “changement d’actionnaire unique”, etc.

> Conclusion : pour les fusions/TUP/apports, il faudra filtrer par mots‑clés et faire de l’extraction par regex ou LLM sur le texte libre pour récupérer le second SIREN (quand il est mentionné). Pas de champ structuré dédié au “partenaire de fusion”.

### 3. Procédures collectives, radiations, créations

- **Procédures collectives** :
  - Un seul SIREN structuré (l’entité en difficulté).
  - Peu ou pas de second SIREN (plans de cession rarement structurés).
  - Utile pour enrichir le statut de l’entité, pas pour les liens inter‑SIREN (sauf extraction texte lourde).

- **Radiations** :
  - 100% avec SIREN de l’entité concernée.
  - Pas de second SIREN.

- **Créations** :
  - Toujours 1 SIREN (nouvelle entité).
  - `listeetablissements` très bien rempli (adresse + activité, souvent montant si création sur achat).
  - Certains cas “Mise en activité d’une société suite à achat” : le cédant est parfois structuré dans `listeprecedentproprietaire` → à exploiter pour un lien “nouvelle entité ↔ vendeur”.

> Conclusion : ces familles servent surtout à enrichir les attributs (statut, activité, création sur achat, etc.), moins les relations entre SIREN, sauf cas spécifiques (création suite à achat).

---

## Approche d’extraction automatisée

### Étape 1 – Pipeline déterministe sur « Ventes et cessions »

1. **Extraction des SIREN structurés**
   - `acquéreur` : `listepersonnes.personne.numeroImmatriculation.numeroIdentification`
   - `cédant` :
     - `listeprecedentproprietaire.personne.numeroImmatriculation.numeroIdentification`
     - sinon `listeprecedentexploitant`
   - Normalisation : supprimer espaces/punct, garder 9 chiffres (SIREN).

2. **Complément via `registre` et texte si cédant manquant**
   - `registre` : liste de SIREN bruts (parfois `acheteur` + `cédant`).
   - Texte : regex sur `modificationsgenerales`, `acte`, `jugement`, `divers` pour repérer des suites de 9 chiffres.
   - Si `counter` structuré absent mais `two_any` ≥ 2 SIREN → considérer qu’il existe une contrepartie textuelle ; selon le cas, stocker l’edge avec confiance plus faible.

3. **Typage de la relation**
   - À partir de `acte.vente.categorieVente` :
     - Achat d’un fonds / établissement principal / secondaire / complémentaire.
     - Mise en activité suite à achat.
     - Apport / attribution / autres.
   - Montant : via `listeetablissements.etablissement.origineFonds` (parser valeur si possible).
   - Date : `dateparution` (et éventuellement `acte.vente.publiciteLegale.date` / `dateCommencementActivite`).

4. **Enregistrement dans la base SIRENE locale**
   - Tables possibles :
     - `sirene_unites_legales` (déjà existant, import INSEE).
     - `bodacc_evenements` (id BODACC, date, type, texte brut, source).
     - `sirene_liens` (siren_src, siren_dst, type_lien, date_effet, montant, adresse_fonds, id_bodacc, confiance).
   - Chaque annonce *Ventes et cessions* devient un ou plusieurs liens `(acquéreur ↔ cédant)` attachés à un ou plusieurs établissements (via adresse).

### Étape 2 – Modéliser les événements à 1 SIREN (statut / attributs)

- **Créations**
  - Enrichir : date de création (BODACC vs SIRENE), adresse et activité détaillée, mention “création sur achat” si cession.

- **Procédures collectives**
  - Ajouter des événements : sauvegarde, RJ, LJ, plan de continuation/cession (même sans second SIREN).

- **Radiations / TUP**
  - Marquer l’unité légale comme dissoute/radiée, et si TUP, récupérer dans le texte la société bénéficiaire (via regex/LLM).

### Étape 3 – Exploiter « Modifications diverses » pour fusions/TUP/apports

1. **Filtrage par mots-clés** dans `modificationsgenerales.descriptif` / `acte.descriptif` :
   - “fusion”, “fusion-absorption”
   - “transmission universelle de patrimoine”, “TUP”
   - “apport partiel d’actif”
   - “cession de parts”, “changement d’actionnaire unique”, etc.

2. **Extraction semi‑automatique** :
   - SIREN de l’entité principale : via `listepersonnes` (structuré).
   - Second SIREN :
     - d’abord via regex sur le texte + sur `registre`,
     - si ambigu, marquer l’événement comme “relation non résolue” (à traiter plus tard).

3. **Typage du lien** :
   - `fusion_absorption`, `TUP`, `apport_partiel`, `cession_de_parts`, `changement_actionnaire`, etc.
   - Ces liens n’auront pas toujours 2 SIREN ; dans ce cas, on stocke au minimum un événement structurant sur l’unité légale.

### Étape 4 – Usage optionnel d’un petit LLM (secours)

- **Périmètre** :
  - Avis où :
    - `familleavis_lib = Ventes et cessions` ou `Modifications diverses`,
    - 2 SIREN détectés mais rôles ambigus,
    - ou pas de second SIREN structuré mais texte très riche (fusion/TUP).

- **Rôle du LLM** :
  - Synthétiser l’événement en JSON structuré :  
    `{ event_type, siren_a, role_a, siren_b?, role_b?, date, montant?, adresse? }`
  - Ne pas le mettre sur le chemin critique (coût + opacité) ; il sert à augmenter la couverture “long tail”.

---

## Prochaines étapes possibles

1. **Formaliser le schéma d’enrichissement**
   - Définir précisément :
     - `sirene_liens` : types de lien (`achat_fonds_principal`, `achat_fonds_secondaire`, `fusion_absorption`, `TUP`, `apport_partiel`, `location_gerance`, …).
     - `bodacc_evenements` : stockage brut des JSON BODACC + métadonnées.

2. **Implémenter un premier extracteur “Ventes et cessions”**
   - Script Python :
     - Récupération périodique des annonces (fenêtre glissante, ex. 2 ans).
     - Extraction structurée + regex simple.
     - Insertion dans la base locale (fichiers SQLite / PostgreSQL / autre).
   - Mesurer sur un vrai volume (ex. toutes les ventes 2024–2025) :
     - taux de liens inter‑SIREN créés,
     - taux d’échecs ou d’ambiguïtés.

3. **Étendre à « Modifications diverses » ciblées**
   - Ajouter un module de détection d’événements de fusion/TUP/apport par mots‑clés.
   - Prototyper extraction regex du second SIREN sur un échantillon annoté.

4. **Intégrer dans la stack SIRETO / pipe V6**
   - Exposer une API interne ou des vues SQL permettant :
     - recherche par groupe (ex. “STRYKER” → récupérer tous les SIREN liés),
     - enrichissement des matchings SIRET (ajouter les “liens BODACC” dans le contexte donné au LLM).

5. **Optionnel : ajouter une couche LLM**
   - Sur les cas les plus complexes (fusions multi‑entités, textes très verbeux) :
     - valider/compléter la sortie des règles,
     - tagger la nature de la relation quand le texte est ambigu.

---

## Résumé

- La famille **“Ventes et cessions”** fournit déjà, avec les champs structurés existants, une base très solide pour construire un graphe de liens entre SIREN (≈ 80% des annonces donnent directement un couple acquéreur/cédant).
- Les **“Modifications diverses”** sont indispensables pour capturer fusions/TUP/apports, mais nécessitent un parsing texte (mots‑clés + regex, éventuellement LLM).
- Les autres familles enrichissent surtout les **attributs** des unités légales (statut, activité, création sur achat).
- La démarche recommandée est itérative : commencer par un extracteur déterministe sur “Ventes et cessions”, mesurer, puis étendre progressivement aux cas plus complexes (fusions/TUP, procédures, etc.) en ajoutant du parsing texte et, si besoin, un LLM de secours.

