# V4.12 — audit métier de cinq REVIEW historiques supplémentaires

## Résultat

| ID | Décision métier | Fiabilité | Erreur du pipeline / du label historique |
|---:|---|---|---|
| 344 | `MATCH_EXACT` → `19771515400021` | Haute | Le ranker est exact, mais l'accepteur refuse le dossier. |
| 410 | `MATCH_EXACT` → `19772128500017` | Haute | Le ranker est exact ; l'ancien label `AMBIGUOUS` est faux et entraîne l'accepteur dans le mauvais sens. |
| 896 | `MATCH_EXACT` → `18770918300169` | Haute | Le ranker est exact ; l'ancien label `AMBIGUOUS` est faux et entraîne l'accepteur dans le mauvais sens. |
| 1073 | `MATCH_EXACT` → `49927250800015` | Haute | Le ranker est exact ; l'ancien label `AMBIGUOUS` est faux et entraîne l'accepteur dans le mauvais sens. |
| 1140 | `AMBIGUOUS` | Haute | Le nom CRM désigne l'ancien exploitant, tandis que le SIRET historique et l'activité à l'adresse désignent le repreneur. |

Bilan : **4 résolus fiables, 1 ambigu, 0 non résolu, 4 labels exacts
utilisables**. Les quatre SIRET exacts étaient déjà les top 1 du ranker. Le
problème principal de ce lot n'est donc pas le classement, mais la qualité des
anciennes cibles et l'excès d'abstention de l'accepteur.

## Preuves consultées

### 344 — Collège Lelorgne de Savigny

Le snapshot SIRENE local associe le collège actif, siège, au SIRET
`19771515400021`, 1 rue de Savigny à Provins. Le
[ministère de l'Éducation nationale](https://www.education.gouv.fr/ivac/2025/0771515d/college-lelorgne-de-savigny)
confirme l'établissement et son UAI `0771515D`; une fiche issue des données
Éducation/SIRENE publie le même
[SIRET et la même adresse](https://annuaire-education.fr/etablissement/provins/college-lelorgne-de-savigny/0771515D.html).

### 410 — Collège Saint-Louis de Lieusaint

Le snapshot SIRENE local associe sans concurrent le collège public actif,
siège, au SIRET `19772128500017`, 124 mail des Pépinières. L'
[Onisep](https://www.onisep.fr/ressources/structures-enseignement/ile-de-france/seine-et-marne/college-saint-louis)
confirme le nom, l'adresse et l'UAI `0772128V`; la fiche SIRENE recoupée publie
le [SIRET exact](https://www.societe.com/societe/college-saint-louis-197721285.html).

### 896 — CCI Avon / CFA UTEC Avon

Le snapshot SIRENE local donne `18770918300169`, actif, à 1 rue du Port de
Valvins, sous l'enseigne CFA UTEC Avon et l'unité légale CCI 77. La
[CCI Seine-et-Marne](https://www.seineetmarne.cci.fr/lieu/avon-fontainebleau-utec)
confirme son implantation d'Avon à cette adresse. Sa propre
[liste des SIRET](https://www.seineetmarne.cci.fr/sites/default/files/2025-08/liste_des_numeros_siret.pdf)
associe explicitement Avon, l'activité CFA UTEC et le SIRET
`18770918300169`.

### 1073 — LFB Biomanufacturing

Le snapshot SIRENE local identifie le siège actif `49927250800015`, avenue des
Chênes Rouges à Alès. Les
[mentions légales officielles de LFB](https://www.lfbbiomanufacturing.com/fr/informations-legales/)
confirment la dénomination, l'adresse et le SIREN `499272508`; le registre
recoupé publie le [SIRET du siège](https://www.pappers.fr/entreprise/lfb-biomanufacturing-499272508)
`49927250800015`. Un second établissement existe dans l'extension « Alès 4 »,
mais cette précision n'apparaît pas dans le CRM, qui désigne naturellement
l'entité et son siège.

### 1140 — LG Alès Automobiles / LG E-Motors

Le cas porte deux preuves incompatibles. Le nom CRM correspond à LG Alès
Automobiles, dont les [statuts](https://www.pappers.fr/entreprise/lg-ales-automobiles-454019704/documents/LG%20ALES%20AUTOMOBILES%20-%20Statuts%20mis%20%C3%A0%20jour%2022-03-2023.pdf)
et le [site du groupe](https://lggroupe.com/lg-contact/lg-ales-automobiles/)
placent l'activité au 157 chemin du Mas de la Bedosse. Mais une cession de la
branche MG Motor au 31 décembre 2024 attribue désormais cette activité et cette
adresse à LG E-Motors, SIRET `97770403000047`, comme le détaille l'
[avis BODACC repris sur la fiche établissement](https://www.pagesjaunes.fr/pros/00336656).
Sans date de référence ni marque/branche explicite dans le CRM, choisir entre
l'ancien exploitant `45401970400021` et le repreneur `97770403000047` serait
arbitraire : le dossier reste `AMBIGUOUS`.

## Conséquence pour l'apprentissage

Ce mini-lot inverse trois cibles `AMBIGUOUS` en `MATCH_EXACT` et une cible
`MATCH_EXACT` en `AMBIGUOUS`. Il explique directement pourquoi les tentatives
de pondération et de nouvelles variables ont échoué : elles réutilisaient
encore 188 labels historiques non relus, dont au moins quatre des cinq premiers
sont incorrects ou mal qualifiés. Aucun nouveau réentraînement ne doit avoir
lieu avant d'avoir mesuré cette contamination sur un lot plus large.
