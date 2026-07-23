# SIRETO neural architecture spikes

Generated: 2026-07-23T13:33:32.677674+02:00

## Protocol

- Shared SIREN-disjoint holdout: 400 queries.
- Maximum full INSEE partition: 2500 candidates.
- Candidate SIRENs belonging to dev/test ground-truth entities were purged from training negatives.
- The cross-encoder reranks the same V7 decider scenes as XGBoost.
- The dual-encoder retrieves from complete INSEE partitions.

## Cross-encoder

| Metric | XGBoost | Cross-encoder |
|---|---:|---:|
| Exact SIRET Hit@1 | 85.25% | 51.75% |
| Exact SIRET Hit@3 | 96.50% | 75.75% |
| Same SIREN @1 | 88.75% | 56.25% |

Paired Hit@1 delta: -33.50%
95% bootstrap CI: [-38.25%, -28.50%]

When restricted to the six highest-scoring XGBoost candidates, the
cross-encoder reaches 67.75%
Hit@1; XGBoost top-6 candidate coverage is
98.50%.

## Dual-encoder retrieval

| Metric | TF-IDF fusion | Dense before fine-tuning | Dense after fine-tuning |
|---|---:|---:|---:|
| Exact SIRET Recall@1 | 59.25% | 65.00% | 74.50% |
| Exact SIRET Recall@10 | 92.75% | 86.25% | 91.25% |
| Exact SIRET Recall@50 | 97.25% | 94.00% | 96.00% |
| SIREN Recall@50 | 97.50% | 94.25% | 96.25% |

Ground-truth coverage in the complete INSEE partitions: 100.00%.
The union of TF-IDF top-50 and dense top-50 reaches
99.25% exact-SIRET recall with a candidate
budget of at most 100, rescuing
8 TF-IDF misses.

## Tokenizer audit

The exported model declares `BertTokenizer` but
contains the multilingual fast SentencePiece tokenizer. Loading it through the
declared class maps ordinary French terms to `<unk>`. These spikes force
`PreTrainedTokenizerFast`. Existing semantic-feature
benchmarks must be treated as suspect until the production loader/export is
corrected and re-evaluated.

## Interpretation

These are bounded architecture probes. They estimate the representation and
retrieval ceilings on entity-disjoint data; they are not production AUTO-rate
or open-set precision measurements. The short cross-encoder is a no-go as a
drop-in XGBoost replacement. The learned dense retriever is a go as a
complementary retrieval channel, not as a replacement for sparse retrieval.
