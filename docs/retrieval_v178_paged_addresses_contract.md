# V178 — lecteur complet et preuves d'adresses sur fichiers

Protocole avant mesure, 4 octobre 2026. V177 compile encore ses 256 packs ;
V178 refuse d'ouvrir une source sans son manifest complet et sa vraie sortie
de seal supervisé. Ce code est une intégration non mesurée nationalement,
pas un résultat de recall ou de latence.

À l'ouverture : hash du manifest, autorité source complète V177/V175/V168,
epochs globaux, vraie sortie parent de seal et de chaque phase, ressources,
hash de chaque tableau CSR, dtype/shape/boundaries, compteurs et égalité
exacte avec toutes les 256 phases. Code/config et autorités restent figés.
Tous les tokens/postings/aliases restent accessibles ; aucun pool/cap de
candidats, pas de sélection préalable selon la requête ou un label.

La requête réutilise coverage170/138 INCHANGÉ : compte exact des tokens
distincts sur chaque whole-alias, max des fractions des canaux, float64,
égalité normalisée non vide, aucun overflow/saturation/token cap. Les
vecteurs sont propres à une requête fraîche sur CATNAT, lus/mis à jour par
morceaux ; seules les cartes readonly de ce lecteur sont fermées/rouvertes,
après copie ou après consommation complète. Pas de purge globale.

PagedAliasSnapshot respecte l'interface AliasSnapshot147, vérifie TOUS les
aliases par morceaux (fini, [0,1], exact implique couverture1), garde des
gathers readonly copiés et refuse toute dérive après scellement. La propriété
des fichiers est transférée à un lease readonly : le producteur mutable est
révoqué. Les epochs seuls ne détectent pas toutes les écritures mmap ; les
deux vecteurs sont aussi hashés au scellement et doivent être rehashés par
snapshot.verify_after avant publication des candidats, après consommation
des facts. L'effacement des seuls fichiers propres au snapshot vérifie ces
hashes, jamais d'effacement d'une source ni d'un fichier changé. Le reader
ReversePart147 inchangé conserve le numéro/CP/INSEE/current et la provenance
de CHAQUE fact d'adresse. Pas de Frankenstein proof ou propagation aux
sites frères. Aucun identifiant comme signal de similarité.

Une deadline totale englobe les contrôles de source, calcul et validation
de snapshot ; pas de temps caché en dehors du coût de requête. Tout échec
garde ses fichiers et ne retourne aucun snapshot partiel. verify_after
rehash tous les tableaux propres à cet accès et recontrôle les autorités
et epochs des sources originales ; les gros corpus bruts ne sont pas à
nouveau rehashés par requête (audits source complets déjà scellés V175).

Le national ne pourra être lancé que sous superviseur V153/V164 inchangé :
1 Gio RSS/80 secondes, pression réelle normale/free >=50%, Mac/CATNAT seuls,
swap global télémétrie non attribuée. Pas de téléchargement, dépense,
CRM/GT/oldhit/fold0/1/modèle aval ouvert. Aucun candidat initial100 ni
recall/SLO/READY produit par ce seul lecteur. Intégration au collecteur
direct initial100 et mesure all-human restent nécessaires.
