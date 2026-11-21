
import re

from langchain_openai import ChatOpenAI

from modules.hybrid_retrieval import (
    hybrid_search,
    reciprocal_rank_fusion,
)


def generate_query_variations(
    question,
    api_key,
    model_name,
    max_queries=3,
):
    """Generate alternative search queries while preserving the original."""

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    queries = [question.strip()]

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_retries=1,
        )

        prompt = f"""
You are a search-query generation assistant for NEXUS AI.

Generate up to {max_queries} alternative search queries
that help retrieve relevant information from enterprise documents.

Rules:
1. Preserve the original meaning and intent.
2. Use different relevant keywords and phrasing.
3. Do not answer the question.
4. Do not introduce unsupported assumptions.
5. Return one search query per line.
6. Do not include explanations.

Original question:
{question}

Alternative search queries:
"""

        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        for line in str(content).splitlines():
            candidate = re.sub(
                r"^\s*(?:[-*\u2022]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            candidate = candidate.strip("\"'` ")

            if not candidate:
                continue

            if candidate.lower() in {
                query.lower() for query in queries
            }:
                continue

            queries.append(candidate)

            if len(queries) >= max_queries + 1:
                break

    except Exception:
        return queries

    return queries


def multi_query_search(
    question,
    vector_store,
    api_key,
    model_name,
    top_k=4,
    fetch_k=20,
    max_queries=3,
    return_diagnostics=False,
):
    """Retrieve and fuse results from multiple search queries."""

    queries = generate_query_variations(
        question=question,
        api_key=api_key,
        model_name=model_name,
        max_queries=max_queries,
    )

    ranked_lists = []
    query_diagnostics = []

    for search_query in queries:
        results = hybrid_search(
            query=search_query,
            vector_store=vector_store,
            top_k=top_k,
            fetch_k=fetch_k,
        )

        query_diagnostics.append({
            "query": search_query,
            "results_found": len(results),
        })

        if results:
            ranked_lists.append(results)

    if not ranked_lists:
        documents = []
    else:
        documents = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            top_k=top_k,
        )

    diagnostics = {
        "strategy": "Multi-query hybrid retrieval",
        "queries": query_diagnostics,
        "query_count": len(queries),
        "retrieved_chunks": len(documents),
    }

    if return_diagnostics:
        return documents, diagnostics

    return documents
