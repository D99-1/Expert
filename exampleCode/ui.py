from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from main import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_user_prompt,
    retrieve_candidates,
)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)


def get_response(query: str, retrieve_k: int = 10, top_n: int = 3) -> tuple[list[dict], str]:
    candidates = retrieve_candidates(
        query=query,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection_name=DEFAULT_COLLECTION,
        retrieve_k=retrieve_k,
        top_n=top_n,
    )

    client = get_groq_client()
    user_prompt = build_user_prompt(query, candidates)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.3,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return candidates, response.choices[0].message.content.strip()


def main() -> None:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    st.set_page_config(page_title="Hack Club YSWS Finder", page_icon="⚡", layout="centered")
    st.title("Hack Club YSWS Finder")
    
    with st.form("ysws_query_form"):
        query = st.text_area(
            "What kind of YSWS are you looking for?",
            placeholder="e.g. beginner-friendly hardware project with prizes",
            height=100,
        )
        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    with st.spinner("Searching the vector DB and asking Groq..."):
        candidates, answer = get_response(query.strip())

    st.subheader("Recommendation")
    st.write(answer)


if __name__ == "__main__":
    main()
