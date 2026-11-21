
import re
from typing import Any


STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "to", "of", "in", "on", "for", "with", "it",
    "this", "that", "uses", "use", "has", "have", "be",
    "by", "as", "from"
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {word for word in words if word not in STOP_WORDS}


def extract_claims(answer: str) -> list[str]:
    if not answer or not answer.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 15
    ]


def calculate_evidence_score(claim: str, evidence: str) -> float:
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)

    if not claim_tokens or not evidence_tokens:
        return 0.0

    matched = claim_tokens.intersection(evidence_tokens)

    return round(len(matched) / len(claim_tokens), 3)


def verify_claims(
    claims: list[str],
    sources: list[dict[str, Any]],
    threshold: float = 0.35
) -> list[dict[str, Any]]:

    results = []

    for index, claim in enumerate(claims, start=1):
        evidence_matches = []

        for source_index, source in enumerate(sources, start=1):
            content = source.get("content", "")
            score = calculate_evidence_score(claim, content)

            if score > 0:
                evidence_matches.append({
                    "source_id": source.get(
                        "source_id", f"S{source_index}"
                    ),
                    "source_name": source.get(
                        "source_name", "Unknown source"
                    ),
                    "relevance_score": score,
                    "evidence": content[:500]
                })

        evidence_matches.sort(
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        best_score = (
            evidence_matches[0]["relevance_score"]
            if evidence_matches else 0.0
        )

        status = (
            "candidate_support"
            if best_score >= threshold
            else "insufficient_evidence"
        )

        results.append({
            "claim_id": f"C{index}",
            "claim": claim,
            "verification_status": status,
            "best_evidence_score": best_score,
            "evidence_matches": evidence_matches
        })

    return results


def build_citation_report(
    answer: str,
    sources: list[dict[str, Any]]
) -> dict:

    claims = extract_claims(answer)
    verified_claims = verify_claims(claims, sources)

    return {
        "total_claims": len(claims),
        "total_sources": len(sources),
        "claims": verified_claims
    }
