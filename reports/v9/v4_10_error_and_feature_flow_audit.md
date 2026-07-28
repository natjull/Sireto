# V4.10 — audit des erreurs restantes et du flux d'information

## Verdict

La prochaine expérience doit porter sur un **accepteur structuré unique**.
Elle ne doit ni enrichir le garde-fou V4.9, ni ajouter une quatrième couche,
ni relancer le retrieval ou le ranker.

Le problème principal n'est pas la puissance du modèle. Entre le ranker et
l'accepteur, 47 des 64 features candidat disparaissent. L'activité principale
SIRENE n'est même pas chargée dans le magasin de candidats. L'accepteur décide
donc de l'AUTO avec une représentation amputée de l'identité et de la fonction
du site.

## Ne pas confondre deux populations

Les 31 dossiers étudiés ici sont les 31 labels fiables faux ou ambigus que le
garde lexical V4.9 ne détecte pas. Ce ne sont pas 31 faux AUTO de `HARD_W1`.

| Sous-population | Cas | Faux AUTO `HARD_W1` |
|---|---:|---:|
| Prédictions hors pli `hard_oof` | 26 | 2 |
| Random V4.8 consommé | 3 | 0 |
| Descriptif non-OOF | 2 | non mesurable |

Les trois faux AUTO random qui ont invalidé `HARD_W1` sont précisément les
trois cas interceptés par V4.9. Ils sont donc hors de ces 31.

Les deux faux AUTO hors pli restants sont :

- lycée Jean-Bart remplacé par l'amicale du lycée ;
- centre hospitalier/USLD remplacé par une autre unité hospitalière du même
  SIREN.

## Familles descriptives

Une lecture exhaustive des preuves des 31 dossiers donne quatre familles.
Elles servent à définir les informations nécessaires, pas à mesurer une
performance.

### 1. Mauvais site au sein du même SIREN — 14 cas

Le système reconnaît l'organisation ou fait confiance au SIRET CRM, mais
sélectionne le mauvais établissement géographique.

- 12/14 ont plusieurs établissements du SIREN dans le pool ;
- 9/14 ont un top-2 du même SIREN ;
- 13/14 ont un numéro de voie différent ou absent ;
- 9/14 ont un code postal différent ;
- 10/14 ont une ville différente.

Il faut représenter la concurrence entre tous les sites du même SIREN, et non
la réduire à un compte et un écart de score.

### 2. Autre personne morale à la même adresse — 14 cas

Le top-1 est un colocataire, une SCI, un propriétaire, un associé ou un autre
exploitant :

- adresse textuelle parfaite ;
- nom faible ;
- SIREN différent.

L'accepteur voit séparément le nom et l'adresse, mais sa régression logistique
ne reçoit pas l'interaction explicite « adresse très forte sans preuve
d'identité ». Cette interaction existait parmi les sept features V8 calculées
mais exclues.

### 3. Entité affiliée ou acteur support — 2 cas

Le collège est remplacé par l'APEL et le lycée par son amicale. Le ranker
dispose de `is_association`, `is_crm_school`, de la forme juridique et de la
source du meilleur nom. L'accepteur perd ces informations.

### 4. CRM composite — 1 cas

`SOFRAT DEPAUL` concatène deux identités légitimes. Aucun signal de pluralité
d'identités CRM n'est transmis à l'accepteur.

## Flux exact des features

### Avant le ranker

Le pipeline possède :

- le nom et l'adresse CRM, la commune, le CP, l'INSEE et le SIRET suspect ;
- l'état du SIRET d'entrée et les frères actifs du même SIREN ;
- les noms établissement/UL/sigle/dirigeants du candidat ;
- l'adresse, la forme juridique, le siège, l'état et les provenances de
  retrieval.

Le magasin candidat ne charge pas
`activitePrincipaleEtablissement`. Il faut aujourd'hui relire le stock maître
pour obtenir l'activité du site.

### Ranker

Le ranker A consomme 64 features :

- similarités nom/adresse ;
- forme juridique, siège, association et école ;
- scores et rangs de retrieval ;
- relation au SIRET/SIREN d'entrée ;
- état et provenance du candidat.

Les sept interactions V8 sont calculables mais exclues. Les features
sémantiques sont désactivées.

### Accepteur

L'accepteur consomme 80 colonnes : 20 agrégats de scène et
top-1/top-2/delta pour seulement 20 familles. Seules 17 des 64 features
candidat survivent.

Sont notamment perdus :

- égalité avec le SIRET ou le SIREN d'entrée ;
- état et provenance du candidat ;
- forme juridique, siège, association et école ;
- exactitudes géographiques et nominales ;
- source du meilleur nom ;
- rangs détaillés des canaux ;
- activité/fonction du site ;
- description de tous les frères du même SIREN.

Les neuf colonnes sémantiques de scène sont égales à zéro sur les 172 cas :
elles occupent de la place mais n'apportent aucun signal.

## Orientation retenue

La V4.10 doit reconstruire une seule matrice de décision query-level :

```text
top-100 et scores du ranker gelé
  → identité du top-1/top-2
  → relation au SIRET CRM
  → activité et rôle du site
  → concurrence complète au sein du SIREN
  → interactions identité/adresse
  → un accepteur
  → AUTO ou REVIEW
```

Les règles déterministes deviennent des features explicables. Elles ne
forment pas un veto indépendant. La régression logistique et un XGBoost peu
profond seront comparés avec les mêmes entrées ; le modèle le plus simple
gagne en cas de parité.

Les 172 cas consommés peuvent servir au design et au fit, jamais à valider la
V4.10. Toute conclusion de performance exigera une population CRM nouvelle,
disjointe et adjudiquée avec des preuves indépendantes.
