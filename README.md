# RAG Reliability Service

Work in progress.

Documentation corpus from [github.com/kubernetes/website](https://github.com/kubernetes/website) at commit ba4362b3dcf7089084e809fc1c8fd4c84831fc8d, licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Eval set

`data/eval/questions.jsonl` holds 49 questions (40 answerable, 9 unanswerable with answer `NOT_IN_DOCS`), split into dev (24) and test (25).

- 5 questions (q001–q005) were written by hand.
- 44 were drafted by an AI model and then reviewed by the project author. Each drafted answerable question was evidence-checked: its supporting sentence was confirmed at the recorded line of its source page at the pinned docs commit above. The drafts, including the evidence sentences and line numbers, are kept in `data/eval/candidates.jsonl`.
- `scripts/validate_eval.py` checks the format, the dev/test split and that each answerable question's source file exists.

## Limitations

- Glossary definitions pulled in by Hugo shortcodes (`glossary_definition`) are not in the corpus, so the opening definition is missing on the 25 pages that use them.
- Some questions have more than one valid page (for example q007, which `concepts/workloads/pods/pod-hostname.md` also answers), so page-level recall slightly undercounts.
