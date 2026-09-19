# 🧠 Self-Corrective Agentic RAG Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama%20(phi3)-black.svg)](https://ollama.com/)
[![VectorDB](https://img.shields.io/badge/VectorStore-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)

An enterprise-grade, 100% open-source **Self-Corrective Agentic Retrieval-Augmented Generation (RAG) Engine**. Designed to eliminate context mismatch, silent hallucinations, and noisy vector lookups using local Small Language Models (SLMs), hybrid semantic retrieval, cross-encoder reranking, and cyclic evaluation state loops.

---

## 📌 Problem Statement

Standard naive RAG implementations suffer from three critical production vulnerabilities:
1. **Semantic Drift & Irrelevant Context:** Vector similarity does not guarantee semantic relevance, leading to noisy contexts entering the LLM window.
2. **Silent Hallucinations:** Generative models often fabricate facts if reference contexts are ambiguous or unverified.
3. **Static Query Execution:** If an initial retrieval yields zero relevant documents, naive RAG fails silently instead of reformulating the search strategy.

This engine implements an **autonomous feedback loop** that evaluates retrieved chunks, verifies factual grounding, and dynamically rewrites search queries when initial candidates fail quality thresholds.

---

## 🏗️ System Architecture

```text
                  [ User Query ]
                         │
                         ▼
             [ Hybrid Retrieval Layer ]
             ├── ChromaDB (Dense Vector Search)
             └── BM25 (Sparse Keyword Search)
                         │
                         ▼
             [ Cross-Encoder Reranker ]
            (ms-marco-MiniLM-L-6-v2)
                         │
                         ▼
             [ Document Relevance Grader ]
            /                             \
     (Irrelevant)                     (Relevant)
          │                                │
          ▼                                ▼
  [ Query Rewriter ]              [ SLM Generation Node ]
   (Reformulate)                      (Ollama: phi3)
          │                                │
          └──► (Loop to Retrieval)         ▼
                                  [ Groundedness Grader ]
                                   /                   \
                            (Hallucination)          (Grounded)
                                  │                       │
                                  ▼                       ▼
                         (Regenerate / Loop)        [ Final Output ]

```

---

## ✨ Key Features

* **Hybrid Search Strategy:** Pairs dense embeddings (`all-MiniLM-L6-v2` via ChromaDB) with sparse keyword matching (`BM25Okapi`) to capture both semantic similarity and exact keyword identifiers.
* **Cross-Encoder Reranking:** Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` over candidate document pools, filtering out low-scoring chunks before prompt assembly.
* **Self-RAG Document Evaluation:** Evaluates each candidate passage for direct relevance using strict JSON schema validation via local Small Language Models (SLMs).
* **Automated Query Expansion & Reformulation:** If context confidence falls below acceptable thresholds, an agent dynamically rewrites the query for multi-hop retrieval.
* **Hallucination Detection Guardrail:** Performs post-generation verification to confirm the output is strictly grounded in reference passages.
* **Local & Privacy-Centric:** Operates on local compute using Ollama (`phi3`) and Hugging Face transformers—zero data egress, zero third-party API dependencies.

---

## 📁 Repository Structure

```text
agentic-rag-engine/
├── data/
│   └── company_policies.txt    # Default domain knowledge base
├── src/
│   ├── __init__.py
│   ├── grader.py               # Self-RAG evaluators & query rewriters
│   ├── graph.py                # Cyclic state graph & orchestration
│   └── retriever.py            # Dense + sparse retriever with cross-encoder
├── app.py                      # Interactive Streamlit tracing console
├── run_agent.py                # Terminal-based end-to-end execution runner
├── test_retriever.py           # Verification script for indexing & reranking
├── requirements.txt            # Pinned project dependencies
└── README.md

```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.9+
* [Ollama](https://ollama.com/?utm_source=gemini) installed and running locally.

Pull the local evaluation model:

```bash
ollama pull phi3

```

### 2. Installation & Setup

Clone this repository and set up a virtual environment:

```powershell
git clone [https://github.com/](https://github.com/)<your-username>/agentic-rag-engine.git
cd agentic-rag-engine

python -m venv .venv
.\.venv\Scripts\activate

```

Install the dependencies:

```powershell
pip install -r requirements.txt

```

### 3. Run Pipeline via Terminal

Execute the end-to-end self-corrective agent trace:

```powershell
python run_agent.py

```

### 4. Launch the Interactive Dashboard

To inspect live node transitions, chunk relevance scores, and hallucination checks in real-time:

```powershell
streamlit run app.py

```

---

## 📊 Verification Trace Example

```text
User Query: What is the PagerDuty escalation trigger for schema drift?
======================================================================
[+] Indexed 4 chunks into ChromaDB & BM25 index.

--- Execution Trace ---
 -> Retrieving for: 'What is the PagerDuty escalation trigger for schema drift?'
 -> Cross-Encoder top rank score: 10.0101
 -> Document Relevance Grader: Accepted 1 relevant context chunk(s).
 -> Synthesizing answer with SLM (phi3)...
 -> Hallucination Groundedness Check: PASSED.

--- Synthesized Answer ---
After 3 consecutive schema drift recovery failures, an alert is dispatched to the on-call data platform engineer via PagerDuty.

```

---

## 🛡️ Tech Stack

| Component | Technology | Description |
| --- | --- | --- |
| **Inference Runtime** | [Ollama](https://ollama.com/?utm_source=gemini) (`phi3`) | Local SLM for grading, reformulation, and generation |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/?utm_source=gemini) | Local persistent vector database |
| **Dense Embeddings** | Hugging Face (`all-MiniLM-L6-v2`) | Lightweight local sentence embeddings |
| **Reranking Model** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | High-precision passage scoring |
| **Sparse Retrieval** | `rank_bm25` | Keyword frequency token matching |
| **User Interface** | [Streamlit](https://streamlit.io/?utm_source=gemini) | Interactive execution and graph state trace UI |

---

## 📜 License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

```

```