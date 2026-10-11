# RAG Reliability Service

Work in progress.

Documentation corpus from [github.com/kubernetes/website](https://github.com/kubernetes/website) at commit ba4362b3dcf7089084e809fc1c8fd4c84831fc8d, licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Eval set

`data/eval/questions.jsonl` holds 49 questions (40 answerable, 9 unanswerable with answer `NOT_IN_DOCS`), split into dev (24) and test (25).

- 5 questions (q001–q005) were written by hand.
- 44 were drafted by an AI model and then reviewed by the project author. Each drafted answerable question was evidence-checked: its supporting sentence was confirmed at the recorded line of its source page at the pinned docs commit above. The drafts, including the evidence sentences and line numbers, are kept in `data/eval/candidates.jsonl`.
- `scripts/validate_eval.py` checks the format, the dev/test split and that each answerable question's source file exists.

## Run the service

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_docs.py          # pinned docs snapshot -> data/raw
.venv/bin/python -m src.ingest.chunking         # chunks -> data/processed
cp .env.example .env                            # then set GROQ_API_KEY in .env (or export it)
.venv/bin/uvicorn src.service.app:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How long are kubeadm join tokens valid by default?"}'
```

## Run with Docker

```bash
cp .env.example .env                 # then set GROQ_API_KEY in .env (or export it in your shell)
docker compose build
docker compose up -d
docker compose ps                    # wait for (healthy)
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How long are kubeadm join tokens valid by default?"}'
docker compose down
```

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

## Reliability signals (exploratory, 5 samples)

Does sampling uncertainty flag closed-book answers that are wrong? For each of the 49 questions, `openai/gpt-oss-20b` answered 5 more times at temperature 1.0 (same locked prompt). The samples were clustered by meaning with `microsoft/deberta-large-mnli` bidirectional entailment, and three scores were computed: semantic entropy, naive entropy (over exact strings) and 1 − agreement rate. The clustering and scoring code is copied from the semantic-entropy-hallucination project. A "bad" answer is the temperature-0 answer judged INCORRECT (answerable) or ANSWERED (unanswerable), using the locked prompt version only; there were no NOT_ATTEMPTED answers to exclude. The run used 50,771 tokens over 245 calls.

AUROC for flagging bad answers (higher score = more likely bad), 95% bootstrap CI (1,000 question resamples, seed 0):

| Score | Test (n=25: 14 bad, 11 good) | Dev + test pooled (n=49: 21 bad, 28 good) |
|---|---|---|
| Semantic entropy | 0.571 [0.333, 0.841] | 0.620 [0.459, 0.784] |
| Naive entropy | 0.435 [0.223, 0.647] | 0.509 [0.373, 0.648] |
| 1 − agreement rate | 0.549 [0.320, 0.828] | 0.604 [0.452, 0.766] |

Paired bootstrap, semantic entropy minus baseline (pooled): naive entropy +0.111 [−0.042, +0.264], 1 − agreement +0.016 [−0.016, +0.054]. Every interval includes zero.

- **Ties dominate:** with 5 samples, semantic entropy takes only 7 distinct values (pooled: 13 of 49 questions at 1.332, 8 at 0). Naive entropy puts 31 of 49 questions at its maximum (1.609), because almost every sample is worded differently.
- **Selective answering by semantic entropy (pooled):** accuracy is 57.1% answering all, 60.5% at 87.8% coverage, 66.7% at 49.0–61.2% coverage, and 72.2% at 36.7% coverage. The coverage floor is 16.3%: 8 questions tie at entropy 0, and only 50% of those are right.
- **Confident errors:** 4 of those 8 zero-entropy answers are wrong the same way every time (q002 "409 Conflict", q034 "Yes", q048 "5", q053 "1 replica"). Consistent misconceptions are exactly what sampling-based uncertainty cannot catch.

**Manual check (by reading the samples, not automated; no labels or scores changed):** of the 13 questions at semantic entropy 1.332, 9 had samples with the same meaning to a human reader placed in different clusters. In 4 of them (q007, q015, q022, q042) all 5 samples meant the same thing; in 5 (q009, q013, q014, q024, q046) there was real disagreement but agreeing samples were still split. Only 4 (q006, q010, q030, q047) truly disagreed. None of the 8 zero-entropy questions had samples with different meanings merged. Finding: the NLI clustering over-splits long, explanatory answers that make the same claim, which inflates semantic entropy mostly on correct answers (1 of the 4 entirely split questions is bad, against 3 of the 4 truly disagreeing ones). This blunts its ability to flag bad answers, so the AUROC above may understate (untested) what better clustering could achieve. The manual check was done after seeing the labels, so it may be biased on borderline cases.

**Reclustering with first sentences only (post-hoc, exploratory):** after the manual check, one variant was defined and committed before running (`025a639`): each saved sample is cut to its first sentence (at the first ". ", "? ", "! " or newline) and reclustered with the same NLI model and question prefix. No new model answers were generated, and labels are unchanged. Same bootstrap (1,000 resamples, seed 0).

| | Test (n=25) | Dev + test pooled (n=49) |
|---|---|---|
| Original semantic entropy | 0.571 [0.333, 0.841] | 0.620 [0.459, 0.784] |
| First-sentence semantic entropy | 0.675 [0.451, 0.901] | 0.700 [0.540, 0.843] |
| Paired difference | +0.104 [+0.013, +0.215] | +0.080 [+0.015, +0.161] |

- **Ties (pooled, first-sentence):** 0.000: 9, 0.500: 11, 0.673: 2, 0.950: 5, 1.055: 2, 1.332: 15, 1.609: 5.
- **The gain is not where the manual check expected:** only 1 of the 13 former 1.332 questions changed (q009, down to 0.500). The four questions where all samples meant the same thing (q007, q015, q022, q042) are still split. Overall 12 of 49 scores changed: 8 correct answers moved down, mostly questions whose samples begin with "No."/"Yes.", and 2 bad answers moved up (q034, q041).
- **Part of the gain is an artifact:** q034 (bad) rose from 0 to 0.673 because two samples reduced to "Yes." were no longer clustered with the longer "Yes – …" samples that mean the same thing.
- **No merging errors:** the four questions that truly disagree and the eight original zero-entropy questions show no different answers merged.

Although the paired intervals exclude zero, this variant was chosen after inspecting the data, the sample is small, and part of the improvement comes from a clustering artifact, so it is not evidence that first-sentence clustering is better. No other variant was tried. Results: `results/metrics/reliability_closed_book_recluster_first_sentence.json`.

Exploratory only: 5 samples and 25–49 questions are too few for firm conclusions, and every AUROC interval includes 0.5. Results: `results/metrics/reliability_closed_book.json`, samples in `results/reliability/closed_book_samples.jsonl`.

## Limitations

- Glossary definitions pulled in by Hugo shortcodes (`glossary_definition`) are not in the corpus, so the opening definition is missing on the 25 pages that use them.
- Some questions have more than one valid page (for example q007, which `concepts/workloads/pods/pod-hostname.md` also answers), so page-level recall slightly undercounts.
