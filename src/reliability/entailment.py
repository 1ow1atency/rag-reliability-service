"""Cluster sampled answers by meaning using bidirectional entailment from an NLI model.

Copied from the semantic-entropy-hallucination project (src/clustering/entailment.py),
without its demo script. Logic is unchanged.
"""

import string

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL = "microsoft/deberta-large-mnli"

_tokenizer = None
_model = None
_entailment_id = None


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model():
    global _tokenizer, _model, _entailment_id
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(_device()).eval()
        label_to_id = {label.lower(): i for i, label in _model.config.id2label.items()}
        _entailment_id = label_to_id["entailment"]
    return _tokenizer, _model


def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _entails(premise: str, hypothesis: str) -> bool:
    tokenizer, model = _load_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).item() == _entailment_id


def _same_meaning(question: str, a: str, b: str) -> bool:
    if _normalize(a) == _normalize(b):
        return True
    text_a = f"Question: {question} Answer: {a}"
    text_b = f"Question: {question} Answer: {b}"
    return _entails(text_a, text_b) and _entails(text_b, text_a)


def cluster_answers(question: str, answers: list[str]) -> list[int]:
    """Return a cluster id for each answer; answers in the same cluster share a meaning.

    Each answer is compared with the first answer (the representative) of each existing
    cluster, in order. It joins the first cluster where entailment holds in both
    directions, and otherwise starts a new cluster.
    """
    representatives = []  # representatives[c] is the first answer placed in cluster c
    cluster_ids = []
    for answer in answers:
        for c, rep in enumerate(representatives):
            if _same_meaning(question, answer, rep):
                cluster_ids.append(c)
                break
        else:
            representatives.append(answer)
            cluster_ids.append(len(representatives) - 1)
    return cluster_ids
