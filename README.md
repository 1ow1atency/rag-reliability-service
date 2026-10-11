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

## Generation results

Answers come from `openai/gpt-oss-20b` at temperature 0, either closed-book (question only) or with RAG (the top 3 chunks from the locked retrieval config, labelled with their page paths, and an instruction to answer only from them or reply `NOT_IN_DOCS`). `openai/gpt-oss-120b` at temperature 0 judges each answer: CORRECT / INCORRECT / NOT_ATTEMPTED against the gold answer, or ABSTAINED / ANSWERED for unanswerable questions. **Dev was used to tune the prompts**; they were locked in commit `73e6cf1` before the single test run.

Test split (25 questions: 21 answerable, 4 unanswerable):

| Mode | Correct | Incorrect | Not attempted | Unanswerable abstained | Unanswerable answered |
|---|---|---|---|---|---|
| Closed-book | 11/21 (52%) | 10/21 (48%) | 0/21 | 0/4 | 4/4 |
| RAG | 17/21 (81%) | 3/21 (14%) | 1/21 (5%) | 4/4 | 0/4 |

With RAG, 16 of the 17 answerable test questions whose evidence chunk was in the top 3 were answered correctly. All four closed-book answers to unanswerable questions were made-up specific values (for example "110 containers" and "$300"). Results: `results/metrics/generation_test.json`, per-question rows in `results/generation/`.

**Paired comparison (test, answerable):** both correct 10, RAG only 7, closed-book only 1, both wrong 3. RAG − closed-book accuracy is +28.6 points (17/21 vs 11/21), 95% bootstrap CI [+4.8, +52.4] (10,000 question resamples, seed 42). The interval is wide because there are only 21 questions. Results: `results/metrics/generation_test_paired.json`.

**q004's wording may be ambiguous:** "how often will the node controller start evicting Pods" can be read as a delay ("5 minutes", the answer both modes gave) rather than the eviction rate in the gold answer. Results are therefore reported both ways. Without the two hand-written test questions (q002, q004), both of which both modes got wrong: RAG 17/19 (89.5%), closed-book 11/19 (57.9%), difference +31.6 points, CI [+5.3, +57.9].

**What closed-book accuracy measures:** the questions ask for docs-specific details (exact defaults, version numbers, file names, limits), so closed-book accuracy reflects recall of those details, not general Kubernetes knowledge.

**Borderline judging rule:** for unanswerable questions, a confident claim that no such limit, value or guarantee exists (for example "There is no hard limit") counts as ANSWERED, not ABSTAINED, in both modes. On dev this applied to two closed-book answers (q003, q049); on test it did not occur.

**Known failure cases (dev):** RAG replied `NOT_IN_DOCS` to three answerable dev questions. Only q035 is a true over-abstention: its evidence sentence was in the first retrieved chunk ("Pod Security Admission is a non-mutating admission controller"), but the model did not infer the "no" answer. For q037 (the evidence sentence was not in the top 3) and q005 (the stated default was in a chunk that was not retrieved), abstaining was the grounded behaviour, and closed-book answered both correctly from memory.

## Limitations

- Glossary definitions pulled in by Hugo shortcodes (`glossary_definition`) are not in the corpus, so the opening definition is missing on the 25 pages that use them.
- Some questions have more than one valid page (for example q007, which `concepts/workloads/pods/pod-hostname.md` also answers), so page-level recall slightly undercounts.
