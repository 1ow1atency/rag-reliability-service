# RAG Reliability Service

Work in progress.

Documentation corpus from [github.com/kubernetes/website](https://github.com/kubernetes/website) at commit ba4362b3dcf7089084e809fc1c8fd4c84831fc8d, licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Eval set

`data/eval/questions.jsonl` holds 49 questions (40 answerable, 9 unanswerable with answer `NOT_IN_DOCS`), split into dev (24) and test (25).

- 5 questions (q001–q005) were written by hand.
- 44 were drafted by an AI model and then reviewed by the project author. Each drafted answerable question was evidence-checked: its supporting sentence was confirmed at the recorded line of its source page at the pinned docs commit above. The drafts, including the evidence sentences and line numbers, are kept in `data/eval/candidates.jsonl`.
- `scripts/validate_eval.py` checks the format, the dev/test split and that each answerable question's source file exists.

## Retrieval results

The retrieval setup was chosen on dev and locked in `configs/final_retrieval.json` (commit `27b08b8`) before a single run on test: fixed 512-token chunks (510-token cap), hybrid RRF (k=60) over dense and BM25 top 50, top 20 reranked with `BAAI/bge-reranker-base`, top 3 returned. Chunk-level hit = a top-k chunk from the right page contains the question's evidence sentence; answerable questions with an evidence sentence only (17 dev, 19 test).

| Setup | Dev R@1 | Dev R@3 | Dev R@5 | Dev MRR@10 | Test R@1 | Test R@3 | Test R@5 | Test MRR@10 | Top-3 tokens (test) |
|---|---|---|---|---|---|---|---|---|---|
| **Final:** fixed_512 + hybrid + bge-reranker | 12/17 | 16/17 | 17/17 | 0.819 | 14/19 | 17/19 | 18/19 | 0.829 | 1,481 |
| fixed_512 dense | 11/17 | 11/17 | 13/17 | 0.692 | 10/19 | 14/19 | 14/19 | 0.647 | 1,418 |
| heading + hybrid + bge-reranker | 9/17 | 15/17 | 16/17 | 0.700 | 13/19 | 18/19 | 18/19 | 0.825 | 959 |

The samples are small (17 dev / 19 test questions, so one question moves recall by 5–6 points), and on test the heading-chunk setup matches the final config within one question while handing the model about 35% fewer tokens, so the 512-token chunks' gain does not clearly pay for their token cost. Full results: `results/metrics/` (`chunker_comparison_dev.json`, `retrieval_test_final.json`).

## Limitations

- Glossary definitions pulled in by Hugo shortcodes (`glossary_definition`) are not in the corpus, so the opening definition is missing on the 25 pages that use them.
- Some questions have more than one valid page (for example q007, which `concepts/workloads/pods/pod-hostname.md` also answers), so page-level recall slightly undercounts.
