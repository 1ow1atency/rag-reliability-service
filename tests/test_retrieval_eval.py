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
