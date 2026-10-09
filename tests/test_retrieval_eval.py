from src.evaluation.retrieval_eval import first_hit_rank, normalize_doc_path


def test_normalize_doc_path_matches_source_file_form():
    expected = "concepts/workloads/pods/pod-lifecycle.md"
    for path in [
        "concepts/workloads/pods/pod-lifecycle.md",
        "data/raw/k8s-website/content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
        "/Users/me/repo/data/raw/k8s-website/content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
        "./concepts/workloads/pods/pod-lifecycle.md",
        "/concepts/workloads/pods/pod-lifecycle.md",
        "concepts\\workloads\\pods\\pod-lifecycle.md",
    ]:
        assert normalize_doc_path(path) == expected, path

    # Page-level hit: first chunk from the right page, whichever path form it uses.
    question = {"source_file": expected}
    retrieved = [
        {"doc_path": "concepts/workloads/pods/pod-qos.md"},
        {"doc_path": "data/raw/k8s-website/content/en/docs/" + expected},
    ]
    assert first_hit_rank(question, retrieved) == 2
    assert first_hit_rank(question, retrieved[:1]) is None


def test_evidence_match_ignores_whitespace_and_requires_right_page():
    from src.evaluation.retrieval_eval import clean_evidence, first_evidence_rank

    raw = ('If you delete the `default` ServiceAccount object in a namespace, the\n'
           '{{< glossary_tooltip text="control plane" term_id="control-plane" >}}\n'
           'replaces it with a new one.')
    evidence = clean_evidence(raw)
    assert evidence == ("If you delete the `default` ServiceAccount object in a namespace, "
                        "the control plane replaces it with a new one.")

    question = {"source_file": "concepts/security/service-accounts.md"}
    wrapped = "Intro.\nIf you delete the `default` ServiceAccount object in a namespace,\n  the control plane\nreplaces it with a new one.\n"
    retrieved = [
        {"doc_path": "concepts/security/other.md", "text": wrapped},           # right text, wrong page
        {"doc_path": "concepts/security/service-accounts.md", "text": "Unrelated."},  # right page, no evidence
        {"doc_path": "concepts/security/service-accounts.md", "text": wrapped},
    ]
    assert first_evidence_rank(question, evidence, retrieved) == 3
    assert first_evidence_rank(question, evidence, retrieved[:2]) is None
