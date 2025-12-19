import os
import uuid
import pandas as pd
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests

# ===================== CONFIG =====================
EXCEL_PATH = os.getenv("EXCEL_PATH", "Career.xlsx")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# ===================== CACHES =====================
@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def init_vector_db():
    client = chromadb.Client(
        Settings(persist_directory="./chroma_db")
    )
    collection = client.get_or_create_collection(
        name="career_faq",
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    return collection


@st.cache_resource
def ingest_excel_to_vectordb(path: str):
    df = pd.read_excel(path).fillna("")
    encoder = load_encoder()
    collection = init_vector_db()

    if collection.count() > 0:
        return

    docs, embeddings, ids = [], [], []

    for _, row in df.iterrows():
        text = "\n".join(
            f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()
        )
        docs.append(text)
        embeddings.append(encoder.encode(text).tolist())
        ids.append(str(uuid.uuid4()))

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids
    )


# ===================== INTENT DETECTION =====================
def detect_intent(user_query: str) -> str:
    prompt = f"""
Classify the user's intent into ONE label only:

MENTAL_HEALTH - stress, anxiety, depression, loneliness, burnout, emotional pain
CAREER_GUIDANCE - job, career, resume, interview, skills, growth

Return only the label.

User query:
{user_query}
"""

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=120
    )

    return response.json()["message"]["content"].strip()


# ===================== RETRIEVAL =====================
def retrieve_context(query: str, top_k: int = 5) -> str:
    encoder = load_encoder()
    collection = init_vector_db()

    query_embedding = encoder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return "\n\n---\n\n".join(results["documents"][0])


# ===================== OLLAMA CALL =====================
def call_ollama(system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        },
        timeout=180
    )

    return response.json()["message"]["content"]


# ===================== MENTAL HEALTH HANDLER =====================
def mental_health_response():
    return (
        "I'm really sorry that you're feeling this way.\n\n"
        "It sounds like you're going through emotional or mental stress. "
        "I’m not a medical professional, but I strongly encourage you to "
        "reach out to a trained counselor or mental health professional.\n\n"
        "📞 **If this feels urgent**, please consider:\n"
        "- Talking to a trusted friend or family member\n"
        "- Consulting a licensed therapist or counselor\n"
        "- Contacting a local mental health helpline\n\n"
        "If you want, I can help you find professional support resources."
    )


# ===================== STREAMLIT APP =====================
def main():
    st.set_page_config(
        page_title="Intent-Aware RAG Assistant",
        page_icon="🧠",
        layout="centered"
    )

    st.title("🧠 Intent-Aware Career & Well-Being Assistant")
    st.write("Understands your intent and responds accordingly.")

    with st.sidebar:
        st.header("Settings")
        excel_path = st.text_input("Excel File Path", EXCEL_PATH)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 5)

    with st.spinner("Loading data into Vector DB..."):
        ingest_excel_to_vectordb(excel_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Understanding intent..."):
            intent = detect_intent(user_input)

        if intent == "MENTAL_HEALTH":
            answer = mental_health_response()

        elif intent == "CAREER_GUIDANCE":
            context = retrieve_context(user_input, top_k)
            answer = call_ollama(
                system_prompt=(
                    "You are a career guidance assistant. "
                    "Answer strictly using the provided context. "
                    "If unsure, say you don't know."
                ),
                user_prompt=f"Context:\n{context}\n\nQuestion:\n{user_input}"
            )

        else:
            answer = "I'm not sure how to help with that."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
