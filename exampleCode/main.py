from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

DEFAULT_CHROMA_PATH = Path(__file__).with_name("chroma_db")
DEFAULT_COLLECTION = "ysws_active"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_QUERY = "beginner-friendly hardware project with prizes"

SYSTEM_PROMPT = """
You are dinobox, a friendly Hack Club YSWS recommender.
Your job is to help users find YSWS programs they would genuinely enjoy.
Use only the retrieved context provided to you.
If context is weak or uncertain, say that clearly.
Return upto 3 recommendations in a warm, practical tone. If you have a strong recommendation, you can return just one. For each recommendation include:
1) Program name and link
2) Why it fits this user
3) One possible downside
Don't list the options out, instead write it conversationally. Do not make up any information. If you don't know, say you don't know.
Finish by asking one short follow-up question.
Assume the user is a high school student aged between 13 to 17 interested in tech, but don't make assumptions beyond that. Do not mention this information directly.
If the secondary options don't fit well, don't even mention or acknowledge them, it's better to just give one strong recommendation than to list multiple mediocre ones. But if there are multiple that apply well, tell all of them to the user.
Do not mention deadlines. Also don't mention that you have to be selected to participate, just talk about the program as if the user can join if they want to. If there is a link, provide it.
""".strip()
def retrieve_candidates(
    query: str,
    chroma_path: Path,
    collection_name: str,
    retrieve_k: int,
    top_n: int,
) -> list[dict]:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    query_embedding = list(model.embed([query]))[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieve_k,
        include=["metadatas", "documents", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        candidates.append({
            "name": str(metadata.get("name", "")),
            "website": str(metadata.get("website", "")),
            "category": str(metadata.get("category", "")),
            "deadline": str(metadata.get("deadline", "")),
            "distance": float(distance),
            "score": 1.0 / (1.0 + float(distance)),
            "snippet": str(document),
        })

    return sorted(candidates, key=lambda x: x["distance"])[:top_n]


def build_user_prompt(user_query: str, candidates: list[dict]) -> str:
    lines: list[str] = [f"User request: {user_query}", "", "Retrieved YSWS context:"]

    for index, item in enumerate(candidates, start=1):
        lines.extend(
            [
                f"Option {index}",
                f"Name: {item['name']}",
                f"Website: {item['website']}",
                f"Category: {item['category']}",
                f"Deadline: {item['deadline']}",
                f"Similarity score: {item['score']:.3f}",
                f"Snippet: {item['snippet'][:800]}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def main() -> None:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set.")

    candidates = retrieve_candidates(
        query=DEFAULT_QUERY,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection_name=DEFAULT_COLLECTION,
        retrieve_k=10,
        top_n=10,
    )

    if not candidates:
        raise SystemExit("No candidates found in the vector DB.")

    user_prompt = build_user_prompt(DEFAULT_QUERY, candidates)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.3,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    output = response.choices[0].message.content.strip()
    print(output)


if __name__ == "__main__":
    main()
