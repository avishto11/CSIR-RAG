# 🧠 Career & Well-Being Chatbots (RAG-based with Ollama)

This repository contains **two Streamlit chatbots** built using **Retrieval-Augmented Generation (RAG)** and **local LLMs via Ollama (LLaMA 2 / LLaMA 3)**.

Both chatbots use **Excel as the knowledge source**, **Sentence Transformers for embeddings**, and **Ollama** for local LLM inference.

---

## 📂 Repository Structure


```text
.
├── intent_aware_rag_chatbot.py   # Intent-aware Career & Mental Health Assistant
├── career_faq_chatbot.py         # Simple Career FAQ RAG Chatbot
├── Career.xlsx                  # Knowledge base (Excel)
├── chroma_db/                   # Local vector DB (auto-created)
└── README.md

---
```

## 🤖 Chatbots Overview

### 1️⃣ Intent-Aware Career & Well-Being Assistant
**File:** `intent_aware_rag_chatbot.py`

**Features**
- Detects **user intent** using an LLM:
  - `CAREER_GUIDANCE`
  - `MENTAL_HEALTH`
- Uses **ChromaDB (cosine similarity)** for career-related retrieval
- Routes mental health queries to a **safe escalation response**
- Uses **Ollama (LLaMA 2 / 3)** for reasoning and generation

**Best for**
- Career guidance systems
- Ethical handling of sensitive mental-health queries
- Agentic / intent-aware RAG use cases

---

### 2️⃣ Career FAQ RAG Chatbot
**File:** `career_faq_chatbot.py`

**Features**
- Simple semantic search + RAG
- Uses in-memory embeddings (no vector DB)
- Answers strictly from Excel content
- Lightweight and easy to run

**Best for**
- FAQ-style assistants
- Demos and PoCs
- Small datasets

---

## 🧩 Technology Stack

- **UI:** Streamlit  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector DB:** ChromaDB (local, cosine similarity)  
- **LLM Runtime:** Ollama  
- **Models:** LLaMA 2 / LLaMA 3  
- **Language:** Python 3.9+

---

## 🧠 Installing Ollama & LLaMA Models

### Step 1: Install Ollama

**macOS / Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh

## For running the application
streamlit run intent_aware_rag_chatbot.py
