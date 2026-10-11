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


def test_evidence_is_whole_detects_split_across_chunks():
    from src.evaluation.retrieval_eval import evidence_is_whole

    page = "tasks/administer-cluster/kubeadm/adding-linux-nodes.md"
    evidence = "By default, node join tokens expire after 24 hours."
    split = [
        {"doc_path": page, "text": "Some intro. By default, node join tokens"},
        {"doc_path": page, "text": "expire after 24 hours. If you are joining a node"},
    ]
    assert not evidence_is_whole(evidence, page, split)

    # Whole in one (wrapped) chunk of the right page counts; the same text on another page does not.
    whole = split + [{"doc_path": page, "text": "tokens.\nBy default, node join tokens\nexpire after 24 hours."}]
    assert evidence_is_whole(evidence, page, whole)
    elsewhere = split + [{"doc_path": "tasks/other.md", "text": evidence}]
    assert not evidence_is_whole(evidence, page, elsewhere)
    assert evidence_is_whole(evidence, "data/raw/k8s-website/content/en/docs/" + page, whole)
