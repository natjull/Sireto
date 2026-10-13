# Fiabilité des labels humains sur les 22 ratés restants

Les labels ne sont pas tous incontestables. **Deux désignent un autre site
que l'adresse CRM et quatre autres sont fortement suspects.** Cela ne permet
pas de déclarer six erreurs humaines certaines : un rattachement de
facturation ou une fiche ancienne peut expliquer certains écarts.

Le score historique reste **504/526, soit 95,82 % exact**. Aucun label,
candidat, modèle, dénominateur ou résultat opérationnel n'a été modifié.

## Ce qui est confirmé et ce qui ne l'est pas

Le périmètre comprend les 22 erreurs exactes du dernier run complet, pas les
seuls cas faciles à vérifier. Trois questions sont séparées :

- Le SIRET labellisé est-il un établissement réel ?
- Désigne-t-il l'entreprise ou le lieu décrit par le CRM ?
- Est-il réellement le contractant ou le destinataire de facturation ?

L'existence d'un établissement n'atteste pas la deuxième ou la troisième
question. Les données utilisées par le retrieval portent le nom et l'adresse
CRM ; elles ne contiennent ni facture, ni contrat, ni date métier de la fiche.
Le contrat actif privilégie le rattachement à la bonne entreprise et au site
physique. Une cible de facturation différente constitue donc une ambiguïté
de définition à rendre explicite, pas une nouvelle règle de score à inventer.

| Conclusion documentaire | Nombre | Signification |
|---|---:|---|
| Site CRM différent du site labellisé | 2 | Écart de site établi ; intention administrative et date non connues. |
| Label fortement suspect | 4 | Autre entreprise nommée à l'adresse, structure manifestement différente ou service public opposé au label. Pas de correction certifiée. |
| Identité de l'entreprise corroborée | 3 | Le nom commercial ou le sigle correspond à l'entreprise labellisée ; l'adresse exacte reste à résoudre. |
| Rattachement relationnel ou historique plausible | 6 | Le nom CRM peut désigner un prestataire, membre ou organisme lié ; pas un alias de la cible. |
| Preuves insuffisantes pour trancher | 5 | Ni validation exacte ni erreur certaine. |
| Informations publiques masquées | 2 | Indécidable sans information autorisée ; aucune reconstruction. |
| Total | **22** | Toutes les erreurs exactes sont conservées. |

**Nombre de mauvaises saisies humaines définitivement prouvées : zéro.**
**Nombre de contradictions de site établies : deux.** Les deux nombres
répondent à des questions différentes. Une contradiction de site signifie
que le label ne désigne pas le lieu CRM, pas que le commercial ne pouvait
avoir aucune raison contractuelle de le choisir.

## Les deux écarts de site

### Foyer Ty Breiz

Le CRM désigne le foyer au **87 rue des Chalâtres à Nantes**. SIRENE donne
à cette adresse le site `20000554400024`, dont l'enseigne est FOYER TY BREIZ
SERVICE HAJI. FINESS nomme aussi ce site et publie ce SIRET. Le label
`20000554400073` est le siège d'ENVOL LOIRE ATLANTIQUE au **20 rue du Port
Boyer**. Les deux établissements sont actifs dans le stock du 1er août 2026.

**Conclusion : pour identifier le lieu demandé, le label vise le mauvais
site.** Le retrieval trouve le foyer en premier et le résultat est déjà
accepté comme opérationnel dans l'évaluation existante. Pour identifier une
entité administrative, le siège peut être intentionnel. Il faut donc valider
la destination métier, pas chercher à faire ressembler le siège au foyer.

### Atol à Châlons en Champagne

Le CRM indique **6 rue de la Marne**. Le label `53989139000020` est un autre
établissement de MODERNE 66, au centre commercial Croix Dampierre, **avenue
du Président Roosevelt**. Il est fermé dans le stock du 1er août 2026.

La même entreprise avait le site `53989139000012`, au **4 rue de la Marne**,
avec le complément **4-6** et l'enseigne OPTIQUE GUILLAUME. Un lien de
succession INSEE vers le site Roosevelt existe au 1er avril 2018, avec
transfert de siège et continuité économique. Le réseau ATOL publie toujours
un magasin au 6 rue de la Marne, mais cette page actuelle ne suffit pas à
identifier son exploitant exact à la date de la fiche CRM.
[Magasin ATOL](https://magasins.atol.fr/grand-est/marne).

**Conclusion : le label ne correspond pas à l'adresse CRM.** Il peut s'agir
d'un rattachement au nouvel établissement après déménagement, d'une ancienne
adresse CRM ou du mauvais magasin d'un exploitant multisite. Le bon SIRET à
retenir selon la date n'est pas définitivement arbitré.

## Les quatre labels fortement suspects

### Convergence Medias

Le CRM indique CONVERGENCE MEDIAS au **199 rue Hélène Boucher à Castelnau-le-Lez**.
Le label `39900947100018` appartient à **INOX**, au **184 chemin des Cigales**.

Une autre entreprise, `481904753`, porte le nom actuel CONVERGENCE. Son site
public `48190475300029` est exactement au 199 rue Hélène Boucher. BODACC
publiait déjà CONVERGENCE MEDIAS sous ce SIREN à cette adresse en 2008.
[Annonce BODACC numéro 1035](https://www.bodacc.fr/telechargements/COMMERCIALES/PDF/B/BODACC-B_20080101_0001_p000.pdf).

**Conclusion : forte présomption de mauvais SIREN.** Nom et adresse
correspondent à une autre entreprise documentée. Aucun lien public examiné
n'établit que le SIRET INOX était le contractant de cette fiche. Une facture
ou le justificatif de la saisie permettrait de trancher définitivement.

### SAS 5eme Etage

Le CRM indique **SAS 5EME ETAGE, 22 rue de la Tour d'Auvergne à Nantes**.
Le label `53535805500049` appartient à **LE MUR DU SONGE**, à cette même
adresse. Le fait de partager une adresse n'établit pas l'identité.

Une personne morale distincte, **5EME ETAGE**, SIREN `842876773`, existe.
L'annonce officielle BODACC `C20260108381`, parue le **10 juin 2026**, donne
explicitement ce nom et le siège au **22 rue de la Tour d'Auvergne**. Le stock
du 1er août donne son site `84287677300016` au **27** de cette rue : la
discordance temporelle est conservée, pas gommée.
[Annonce BODACC](https://www.bodacc.fr/pages/annonces-commerciales-detail/?q.id=C20260108381).

**Conclusion : forte présomption de confusion entre deux entreprises
co-localisées ou proches.** Le nom CRM n'est pas un alias prouvé du MUR DU
SONGE. L'hypothèse propriétaire/occupant ou contractant reste à confirmer
dans le dossier commercial.

### Maison des projets de Stains

Le CRM désigne **SIP PLAINE COMMUNE MAISON DES PROJETS**, au **10 avenue
George Sand à Stains**. Le label `30120992000034` appartient à **ABAQUEPLAST**,
au **41 avenue Gaston Monmousseau**, une entreprise de commerce de fournitures
industrielles.

La ville publie la Maison des projets à l'adresse CRM dans un article daté
du **10 septembre 2021**. Il s'agit d'un lieu d'information sur les projets
urbains ; cette source ne fournit pas son SIRET exact.
[Ville de Stains](https://stains.fr/actualites/pretresse-clos-saint-lazare-infos-sur-les-projets-urbains/).

**Conclusion : label fortement incohérent avec le lieu décrit.** Le bon
opérateur public et son établissement ne sont pas identifiés avec une preuve
SIRET suffisante. Ne pas remplacer le label par un SIRET communal au hasard,
ni interpréter automatiquement SIP comme un service des impôts.

### Trescal à Bollène

Le label `31381896500029` appartient à une entreprise individuelle, pas à la
personne morale **TRESCAL**, SIREN `562047050`. Le code d'activité historique
de l'unité labellisée est `93.0E` ; ce code seul ne prouve pas une erreur.
Les noms de personnes physiques n'ont pas été lus ou exportés pour ce contrôle.

Le lien INSEE du SIRET labellisé vers `31381896500045` date du **24 octobre
1996**. TRESCAL publie bien un laboratoire de Bollène et son identité juridique
est corroborée par des sources publiques. L'adresse CRM au 63 rue Henri
Pellegrin n'a toutefois pas été reliée avec certitude à un établissement
TRESCAL daté.
[Laboratoire TRESCAL Bollène](https://www.trescal.dk/calibration-lab/bollene-france).

**Conclusion : forte suspicion de mauvais SIREN, mais pas de correction
SIRET certifiée.** Il faut retrouver l'adresse historique ou le document
commercial. L'activité différente et la forme juridique ne suffisent pas
à prouver à elles seules qui facturait.

## Les identités corroborées et les rattachements plausibles

| Cas | Ce qui est établi | Verdict sur le label exact |
|---|---|---|
| GEDIMAT Cambrai | Le réseau nomme GEDIMAT Bracq au Marais Cantimpré ; la CCI associe GEDIMAT au SIRET LEON BRACQ `68552038900030`. | Entreprise corroborée. Rue du Champ de Tir et numéro CRM 9012 à vérifier ; pas d'erreur exacte démontrée. |
| EBSN Nevers | Les mentions de l'association publient explicitement `43266299700019` et le nom Espace Bernadette Soubirous Nevers. | Identité corroborée. L'adresse CRM boulevard Victor Hugo versus rue Saint-Gildard n'est pas arbitrée ; ne pas supposer une entrée adjacente. |
| Cuisines Schmidt Hauconcourt | Une page du magasin indexée et un jugement du 17 décembre 2025 relient CUISINES SCHMIDT Hauconcourt à SARL SALVY. Le domaine du magasin redirige maintenant ; sa publication passée n'est pas une archive de production. | Entreprise corroborée, adresse exacte encore à rapprocher. Pas d'erreur de SIRET démontrée. |
| OPHLM Angoulême | Le SIRET label `13001281800220` porte l'enseigne SIRENE **TRÉSORERIE D'ANGOULÊME OPH**. L'office HLM a, lui, son SIRET `40278771700022` au 42 rue du Docteur Duroselle. | Rattachement au comptable public plausible. Le label est différent de l'office et du lieu, mais pas un choix arbitraire prouvé faux. |
| NAOS Marseille | NAOS fournit des services à Aubert ; le magasin Aubert La Valentine correspond au secteur CRM. Le label Aubert fermé possède un successeur dans un autre SIREN au 1er janvier 2025. | Ancien exploitant plausible. Date et rôle CRM nécessaires ; pas une erreur prouvée. |
| NAOS Arçonnay | Le site labellisé de l'entreprise AUBERT FRANCE porte l'enseigne AUBERT, ZC Leclerc. NAOS et Aubert ont une relation publiée. | Entreprise cliente plausible, SIRET historique fermé ; adresse et date à arbitrer. |
| NAOS Mondeville | Site Aubert fermé au centre commercial Mondeville 2, succession INSEE au 1er janvier 2025 vers un autre SIREN. | Même distinction prestataire/site-client et ancien/nouvel exploitant. |
| NAOS Villeneuve d'Ascq | Site Aubert fermé boulevard de Valmy, succession INSEE au 1er janvier 2025 vers un autre SIREN. | Même distinction ; avenue de l'Avenir et site exact à corroborer. |
| ALPI Amou | BANATIC publie l'adhésion de la communauté Coteaux et Vallées des Luys à ALPI. La fiche observée est mise à jour après le cutoff. | Rattachement à un membre plausible ; ALPI et la communauté ne sont pas la même personne morale. Site d'Amou et date non arbitrés. |

Sources complémentaires :
[GEDIMAT](https://www.gedimat.fr/magasins),
[CCI Hauts-de-France](https://preprodevent.hauts-de-france.cci.fr/rst/entreprise/leon-bracq-et-cie/),
[mentions Espace Bernadette](https://www.sainte-bernadette-soubirous-nevers.com/fr/pages/mentions-legales/),
[jugement SALVY](https://www.doctrine.fr/d/TJ/Metz/2025/TJP496288783B4A133C1889),
[relation NAOS et Aubert publiée par SAP en 2020](https://news.sap.com/france/2020/03/allgeier-es-france-met-en-place-pour-naos-une-plateforme-e-commerce-mutualisee-pour-deux-distributeurs-specialises/),
[Aubert La Valentine](https://magasins.aubert.com/fr/france-FR/M78/aubert-la-valentine/details),
[BANATIC ALPI](https://www.banatic.interieur.gouv.fr/intercommunalite/254003304-sm-agence-landaise-pour-l-informatique--alpi).

Une relation publiée ne valide ni la cible de facturation du CRM ni l'exactitude
du SIRET. Une fermeture aujourd'hui ne rend pas faux un label historique.

## Les sept dossiers restant indécidables

| Cas | Point encore non résolu |
|---|---|
| CEGEDIMCLOUD SMARTRX Villeurbanne | Label SMART GESTION, enseigne APTEO, 51 rue Emile Decorps ; CRM 18 rue de Cyprian. Aucun pont démontré vers ce SIRET. SMART RX a aussi un ancien site au 13 rue Emile Decorps : proximité de noms/rues insuffisante. |
| INFODESCA | Bonne entreprise retrouvée, mais label ZONE INDUSTRIELLE versus CRM 320 avenue Georges Guignard et autre site au 830. Historique de numérotation ou site distinct non tranché. |
| CERGY PONTOISE RUE DES VERGERS | Label siège de l'agglomération, CRM boulevard de l'Hautil. Aucun établissement précis du lieu confirmé ; siège administratif possible. |
| ADOS LOISIRS | Label mairie rue de Fonton, CRM 20 rue Jean Mermoz. Le site de cette rue dans SIRENE porte HALTE GARDERIE et est fermé : il ne prouve pas à lui seul le lieu des adolescents. |
| CAMPHES Corbie | Label NDS/KANDY au 40 rue Gambetta, CRM 4 chemin du Bastion. Une relation réseau Kandy/Camphes est plausible, mais ni cette adresse ni l'exploitant exact à la date CRM ne sont démontrés. |
| GROUPAMA SUPPORTS ET SERVICES | Champs publics protégés : aucun historique utilisé pour restaurer nom/adresse masqués. |
| CAPITAL SANTE INSTITUT | Même limite de diffusion ; la source publique ne permet pas de certifier le label. |

## La clarification métier nécessaire

Le jeu mélange potentiellement trois cibles : **le lieu où fournir le
service**, **son exploitant**, et **l'entité qui contracte ou paie**. Ce sont
parfois trois établissements ou entreprises différents. L'audit ne doit pas
transformer cette ambiguïté en six corrections certaines.

Pour terminer l'arbitrage, il faut seulement le justificatif de rattachement
et la date métier des dossiers les plus suspects : Convergence Medias,
SAS 5eme Etage, Maison des projets, Trescal, puis Foyer Ty Breiz et Atol.
Une facture/commande portant le SIRET ou l'explication originale du commercial
peut répondre ; une nouvelle recherche web ne révèle pas ce contrat privé.

En attendant, les labels restent inchangés. Les vrais écarts documentaires
sont conservés dans un audit séparé ; les sources trouvées ne sont pas
converties en alias de retrieval. Le prochain objectif n'est pas d'apprendre
« OPHLM = DDFIP » ou « NAOS = AUBERT », mais de représenter explicitement
le rôle d'un comptable, d'un prestataire ou d'un exploitant historique.

## Traçabilité technique

Contrôle des 22 outcomes après génération complète possédée et scellée.
Script `scripts/embeddinggemma2_v122_label_clarification.py`, commits
`1f25cc3e` puis `36731c3b` pour sérialiser les dates officielles en ISO.
Le premier répertoire interrompu `_a` est conservé ; le contrôle complet
est dans `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v122/label_clarification_20261013_b`.
Le report relie les stocks publics du 1er août 2026, les successions et les
artefacts scellés. Aucun nom de personne physique n'est exporté.

Le contrôle en lecture seule de l'API DILA a confirmé l'annonce
`C20260108381`, son sujet 842876773, sa date du 10 juin 2026 et le siège
au 22 rue de la Tour d'Auvergne. Les pages actuelles examinées restent des
preuves de diagnostic : aucune admission rétroactive au benchmark.

Les folds humains 0/1, le test 4323 et le contrôle indépendant 962 restent
fermés ; aucune étape de training, aucun nouveau retrieval, aucune correction
de label, aucune modification des métriques gelées, aucun unit test.
