import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import requests

DEFAULT_EXCEL_PATH = os.getenv(
    "EXCEL_PATH",
    "Career.xlsx"
)

# ---------- DATA & EMBEDDINGS ----------

@st.cache_resource
def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.fillna("")

    def combine_row(row):
        parts = []
        for col in df.columns:
            col_lower = col.lower()
            val = str(row[col]).strip()
            if not val:
                continue
            if "question" in col_lower:
                parts.append(f"Question: {val}")
            elif "answer" in col_lower:
                parts.append(f"Answer: {val}")
            else:
                # Extra metadata if needed
                parts.append(f"{col}: {val}")
        return "\n".join(parts)

    df["combined"] = df.apply(combine_row, axis=1)
    return df


@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_corpus(excel_path: str):
    df = load_excel(excel_path)
    encoder = load_encoder()
    corpus = df["combined"].tolist()
    embeddings = encoder.encode(corpus, convert_to_tensor=True)
    return df, embeddings


def retrieve_context(query: str, df: pd.DataFrame, embeddings, top_k: int = 5) -> str:
    encoder = load_encoder()
    query_emb = encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)[0]

    chunks = []
    for hit in hits:
        idx = hit["corpus_id"]
        score = float(hit["score"])
        text = df["combined"].iloc[idx]
        chunks.append(f"[Match score: {score:.3f}]\n{text}")

    return "\n\n---\n\n".join(chunks)


# ---------- OLLAMA LLM CALL ----------

def call_ollama_llm(system_prompt: str, user_message: str) -> str:
    """
    Calls Ollama's chat API. Assumes Ollama is running on the host machine.
    On Windows Docker Desktop, host is reachable as host.docker.internal.
    """
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama2")

    url = f"{base_url}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"Error calling Ollama: {e}"


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="Career FAQ Chatbot (Ollama)", page_icon="💬")
    st.title("💬 Career FAQ Chatbot – Ollama (Llama 2)")
    st.write("Ask questions based on the Excel FAQ content.")

    with st.sidebar:
        st.header("Settings")
        excel_path = st.text_input("Excel file path", value=DEFAULT_EXCEL_PATH)
        top_k = st.slider("Top-K context documents", 1, 10, 5)
        st.markdown("---")
        default_system = (
            "You are a helpful assistant answering questions strictly based on the "
            "provided career FAQ context from the company. "
            "If the answer is not clearly present in the context, say "
            "'I am not sure based on the FAQ document.'"
        )
        system_prompt = st.text_area("System prompt", value=default_system, height=160)

    # Build corpus & embeddings
    with st.spinner("Loading Excel and building embeddings..."):
        df, corpus_embeddings = build_corpus(excel_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your question...")
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve context
        with st.spinner("Retrieving relevant FAQ context..."):
            context = retrieve_context(user_input, df, corpus_embeddings, top_k=top_k)

        # Build prompt for LLM
        llm_user_prompt = (
            f"Here is the FAQ context from the Excel file:\n\n"
            f"{context}\n\n"
            f"Now, answer the user's question based only on this context.\n\n"
            f"User's question: {user_input}"
        )

        # Call Ollama
        with st.spinner("Asking Llama 2 (via Ollama)..."):
            answer = call_ollama_llm(system_prompt, llm_user_prompt)

        # Show answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
