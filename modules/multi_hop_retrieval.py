
from langchain_openai import ChatOpenAI

from modules.hybrid_retrieval import (
    hybrid_search,
    reciprocal_rank_fusion,
)
from modules.multi_query_retrieval import multi_query_search


def _response_text(response):
    """Convert common LLM response formats into plain text."""
    content = response.content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


def generate_followup_query(
    question,
    documents,
    api_key,
    model_name,
):
    """Generate one evidence-informed follow-up retrieval query."""

    if not question or not question.strip() or not documents:
        return None

    evidence_parts = []

    for index, document in enumerate(documents[:5], start=1):
        content = (document.page_content or "").strip()

        if not content:
            continue

        evidence_parts.append(
            f"[Evidence {index}]\n{content[:1000]}"
        )

    if not evidence_parts:
        return None

    evidence = "\n\n".join(evidence_parts)

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_retries=1,
        )

        prompt = f"""
You are a retrieval-planning assistant for NEXUS AI.

Your task is to identify one useful follow-up search query.

Original question:
{question}

Initially retrieved evidence:
{evidence}

Rules:
1. Do not answer the original question.
2. Identify a missing detail, related entity, or relationship
   that could help answer the original question.
3. Use only entities and concepts supported by the question
   or retrieved evidence.
4. Do not invent names, facts, dates, or assumptions.
5. Return exactly one concise search query.
6. If no useful follow-up search is possible, return NONE.
7. Return no explanation.
"""

        response = llm.invoke(prompt)
        candidate = _response_text(response).strip()

        candidate = candidate.splitlines()[0].strip()
        candidate = candidate.strip("\"'` ")

        if candidate.upper() in {"NONE", "N/A", "NO QUERY"}:
            return None

        if not candidate:
            return None

        if candidate.lower() == question.strip().lower():
            return None

        return candidate

    except Exception:
        # Keep the initial retrieval results if query planning fails.
        return None



def multi_hop_search(
    question,
    vector_store,
    api_key,
    model_name,
    top_k=4,
    fetch_k=20,
    max_hops=1,
    return_diagnostics=False,
):
    """Run multi-query retrieval and up to one evidence-informed follow-up hop."""

    if not question or not question.strip():
        if return_diagnostics:
            return [], {
                "strategy": "Multi-query hybrid retrieval with multi-hop",
                "initial": {},
                "followup": {
                    "attempted": False,
                    "query": None,
                    "results_found": 0,
                },
                "retrieved_chunks": 0,
                "hops_completed": 0,
            }
        return []

    if top_k < 1 or fetch_k < 1:
        raise ValueError("top_k and fetch_k must be positive.")

    if max_hops < 0:
        raise ValueError("max_hops cannot be negative.")

    # Stage 1: Initial multi-query retrieval.
    initial_output = multi_query_search(
        question=question,
        vector_store=vector_store,
        api_key=api_key,
        model_name=model_name,
        top_k=top_k,
        fetch_k=fetch_k,
        max_queries=2,
        return_diagnostics=return_diagnostics,
    )

    # Support both the normal return format and existing mocked tests.
    if (
        isinstance(initial_output, tuple)
        and len(initial_output) == 2
    ):
        initial_results, initial_diagnostics = initial_output
    else:
        initial_results = initial_output
        initial_diagnostics = {
            "strategy": "Multi-query hybrid retrieval",
            "queries": [],
            "query_count": 0,
            "retrieved_chunks": len(initial_results),
        }

    diagnostics = {
        "strategy": "Multi-query hybrid retrieval with multi-hop",
        "initial": initial_diagnostics,
        "followup": {
            "attempted": False,
            "query": None,
            "results_found": 0,
        },
        "retrieved_chunks": len(initial_results),
        "hops_completed": 0,
    }

    if not initial_results or max_hops < 1:
        if return_diagnostics:
            return initial_results, diagnostics
        return initial_results

    # Stage 2: Generate one evidence-informed follow-up query.
    followup_query = generate_followup_query(
        question=question,
        documents=initial_results,
        api_key=api_key,
        model_name=model_name,
    )

    if not followup_query:
        if return_diagnostics:
            return initial_results, diagnostics
        return initial_results

    diagnostics["followup"]["attempted"] = True
    diagnostics["followup"]["query"] = followup_query

    # Stage 3: Retrieve evidence using the follow-up query.
    followup_results = hybrid_search(
        query=followup_query,
        vector_store=vector_store,
        top_k=top_k,
        fetch_k=fetch_k,
    )

    diagnostics["followup"]["results_found"] = len(followup_results)

    if not followup_results:
        if return_diagnostics:
            return initial_results, diagnostics
        return initial_results

    # Stage 4: Fuse initial and follow-up evidence.
    final_results = reciprocal_rank_fusion(
        ranked_lists=[
            initial_results,
            followup_results,
        ],
        top_k=top_k,
    )

    diagnostics["retrieved_chunks"] = len(final_results)
    diagnostics["hops_completed"] = 1

    if return_diagnostics:
        return final_results, diagnostics

    return final_results
