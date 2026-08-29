"""Scoring for memory evaluation.

Two paths:

* ``deterministic_match`` -- a fast, cheap, lenient proxy for development
  iteration (exact / containment on normalized text). NOT the official
  metric; use the judge for published numbers.

* ``judge_results`` -- an LLM judge using the official LongMemEval answer-
  check prompts (replicated verbatim so scores are comparable to published
  runs). Judge is model-agnostic via ``ReaderClient``.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass

from .protocol import ReaderClient
from .runner import ReplayResult

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "to", "of", "in", "on", "at", "for", "with",
    "and", "or", "but", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "what", "when",
    "where", "who", "how", "that", "this", "there", "as", "by", "from",
    "not", "no", "yes", "me", "us", "them", "be", "been", "being",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def deterministic_match(hypothesis: str, answer: str) -> bool:
    """Lenient deterministic correctness proxy.

    True if normalized hypothesis equals normalized answer, contains the
    answer as a substring, or contains the answer's content words.
    Suitable for development iteration only.
    """
    hyp = _normalize(hypothesis)
    ans = _normalize(answer)
    if not ans:
        return True
    if hyp == ans:
        return True
    if ans in hyp:
        return True
    ans_words = [w for w in ans.split() if w not in _STOPWORDS]
    return bool(ans_words) and all(w in hyp.split() for w in ans_words)


def build_anscheck_prompt(
    question_type: str,
    question: str,
    answer: str,
    hypothesis: str,
    abstention: bool,
) -> str:
    """Replicate the official LongMemEval answer-check prompts."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {hypothesis}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    common = (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no."
    )
    if question_type == "temporal-reasoning":
        common += (
            " In addition, do not penalize off-by-one errors for the number of "
            "days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still correct."
        )
    elif question_type == "knowledge-update":
        common += (
            " If the response contains some previous information along with an "
            "updated answer, the response should be considered as correct as "
            "long as the updated answer is the required answer."
        )
    elif question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\n"
            f"Model Response: {hypothesis}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return (
        f"{common}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
        f"Model Response: {hypothesis}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


@dataclass
class ScoreReport:
    overall: float
    per_type: dict[str, float]
    counts: dict[str, int]
    n: int


def _aggregate(flags: list[tuple[str, bool]]) -> ScoreReport:
    per_type: dict[str, list[bool]] = defaultdict(list)
    for qtype, correct in flags:
        per_type[qtype].append(correct)
    n = len(flags)
    overall = sum(c for _, c in flags) / n if n else 0.0
    return ScoreReport(
        overall=overall,
        per_type={t: sum(v) / len(v) for t, v in per_type.items()},
        counts={t: len(v) for t, v in per_type.items()},
        n=n,
    )


def score_deterministic(results: list[ReplayResult]) -> ScoreReport:
    """Score with the deterministic proxy."""
    flags = [
        (r.question_type, deterministic_match(r.hypothesis, r.answer))
        for r in results
    ]
    return _aggregate(flags)


def judge_results(
    results: list[ReplayResult],
    judge: ReaderClient,
    temperature: float = 0.0,
) -> tuple[ScoreReport, list[ReplayResult]]:
    """Score with an LLM judge using official LongMemEval prompts.

    Returns (report, results) where each result carries its judged label.
    """
    labeled: list[ReplayResult] = []
    for r in results:
        prompt = build_anscheck_prompt(
            r.question_type, r.question, r.answer, r.hypothesis, r.is_abstention
        )
        response = judge.complete(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        label = "yes" in response.lower()
        labeled.append(
            ReplayResult(
                question_id=r.question_id,
                question_type=r.question_type,
                question=r.question,
                answer=r.answer,
                hypothesis=r.hypothesis,
                is_abstention=r.is_abstention,
                timing=r.timing,
                judged=label,
            )
        )
    flags = [(r.question_type, r.judged) for r in labeled]
    assert all(f is not None for _, f in flags)
    return _aggregate(flags), labeled