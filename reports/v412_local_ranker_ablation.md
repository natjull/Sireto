# V4.12 — ablation ranker sur 241 labels localement identifiables

Le score OOF courant recalculé sur le même périmètre est `212/241`.
Le contrôle est l'ensemble figé de `1 127` IDs `scope=CONTROL`; les quatre vérités du contre-audit sont appliquées en mémoire. Le ranker courant vaut `1 123/1 127` après cette correction.

| Variante | Poids | OOF 241 | Delta | Régressions OOF | Contrôle 1 127 | Fixes contrôle | Régressions contrôle | 4 contre-audits | Base fit OOF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| targeted | 0.50 | 219/241 (90.87 %) | +7 | 5 | 1125/1127 (99.82 %) | 2 | 0 | 2/4 | 4647/4666 |
| source_relational | 0.25 | 218/241 (90.46 %) | +6 | 5 | 1124/1127 (99.73 %) | 1 | 0 | 1/4 | 4649/4666 |
| targeted | 0.25 | 217/241 (90.04 %) | +5 | 7 | 1124/1127 (99.73 %) | 1 | 0 | 1/4 | 4651/4666 |
| targeted | 0.75 | 215/241 (89.21 %) | +3 | 6 | 1126/1127 (99.91 %) | 3 | 0 | 3/4 | 4647/4666 |
| targeted | 1.00 | 215/241 (89.21 %) | +3 | 6 | 1126/1127 (99.91 %) | 3 | 0 | 3/4 | 4649/4666 |
| source_relational | 0.75 | 215/241 (89.21 %) | +3 | 7 | 1125/1127 (99.82 %) | 2 | 0 | 2/4 | 4646/4666 |
| source_relational | 0.50 | 215/241 (89.21 %) | +3 | 7 | 1124/1127 (99.73 %) | 1 | 0 | 1/4 | 4649/4666 |
| source_relational | 1.00 | 213/241 (88.38 %) | +1 | 9 | 1125/1127 (99.82 %) | 2 | 0 | 2/4 | 4648/4666 |

## Verdict

La meilleure variante sans régression contrôle est `targeted` au poids `0.50` : `219/241` OOF et `1125/1127` contrôles.
Elle effectue `12` corrections et `5` régressions à l'intérieur des 241 cas, soit un gain net de `+7`. Sur les 1 127 contrôles corrigés, elle effectue `2` corrections sans perdre aucun des 1 123 cas déjà corrects.
Le contrôle historique base-fit vaut `4647/4666` contre `4655/4666` pour le ranker courant : le gate métier est propre, mais il ne faut pas présenter le résultat comme une absence de régression universelle.

### Quatre contrôles contre-audités — meilleure variante

| Query | Ranker courant | Variante retenue | Vérité corrigée | Correct |
|---|---:|---:|---:|---|
| `10420` | `39187532500014` | `39187532500014` | `97150121800095` | non |
| `12633` | `51942934400020` | `82113394900015` | `82113394900015` | oui |
| `fresh:FR027494` | `92264921500014` | `92264921500014` | `33888018003730` | non |
| `fresh:FR037625` | `39155877200029` | `32671645300021` | `32671645300021` | oui |

Les sorties `non_trusted_dev` natives des runs portent sur 1 135 cas et ne sont pas utilisées pour ce gate. Chaque répertoire de run contient désormais une sortie candidats recalculée exactement sur les 1 127 contrôles figés.
