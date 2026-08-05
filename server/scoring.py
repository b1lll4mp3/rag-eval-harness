"""Deterministic fact scoring — verbatim port of eval/run_eval.py.

Any change here MUST be mirrored in eval/run_eval.py and re-baselined:
two scorers that drift apart make runs incomparable.
"""
import re

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\bno (?:relevant )?(?:context|information|documents?)\b",
    r"\bnot (?:enough|sufficient) (?:context|information)\b",
    r"\bcannot (?:find|answer|determine)\b",
    r"\bunable to (?:find|answer|determine)\b",
    r"\bthe (?:provided )?context does not\b",
]


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower())


def fact_present(fact, normalized_answer):
    fact = normalize(fact)
    if re.fullmatch(r"[\w.:@/-]+", fact):
        return re.search(rf"(?<!\w){re.escape(fact)}(?!\w)", normalized_answer) is not None
    return fact in normalized_answer


def score_facts(key_facts, answer):
    if not key_facts:
        return None, [], []
    norm = normalize(answer)
    matched, missed = [], []
    for alternatives in key_facts:
        hit = next((a for a in alternatives if fact_present(a, norm)), None)
        (matched if hit else missed).append(hit or alternatives[0])
    return len(matched) / len(key_facts), matched, missed


def is_refusal(answer):
    norm = normalize(answer)
    if len(norm.strip()) < 15:
        return True
    return any(re.search(p, norm) for p in REFUSAL_PATTERNS)
