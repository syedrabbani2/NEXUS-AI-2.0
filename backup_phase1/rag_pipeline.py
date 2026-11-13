
from langchain_openai import ChatOpenAI

SYSTEM_INSTRUCTIONS = """You are NEXUS AI, an enterprise knowledge assistant.
Answer the user's question using ONLY the supplied document context.
If the context does not contain the answer, say:
'I couldn't find sufficient information in the uploaded documents.'
Do not invent policies, dates, amounts, names, or rules.
Be concise, clear, and professional. If sources disagree, explain the conflict.
The application will display source documents separately, so do not fabricate citations.
"""


def answer_question(question, vector_store, api_key, model_name, top_k=4):

    if not question.strip():
        raise ValueError("Please enter a question.")

    matches = vector_store.similarity_search(question, k=top_k)

    if not matches:
        return "I couldn't find sufficient information in the uploaded documents.", []

    context_parts = []

    for i, doc in enumerate(matches, start=1):
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page_number", "Unknown")

        context_parts.append(
            f"[Context {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # OpenRouter LLM connection
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2
    )

    prompt = f"""{SYSTEM_INSTRUCTIONS}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Write the answer based only on the document context above.
"""

    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in answer
        )

    sources = []
    seen = set()

    for doc in matches:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page_number")
        )

        if key not in seen:
            seen.add(key)

            sources.append({
                "file": key[0] or "Unknown document",
                "page": key[1] or "Unknown",
                "preview": doc.page_content[:260].strip(),
            })

    return str(answer), sources
