import streamlit as st
import time
from src.retriever import HybridRerankRetriever
from src.grader import AgentEvaluator
from src.graph import AgenticRAGGraph

st.set_page_config(
    page_title="Self-Corrective Agentic RAG Engine",
    page_icon="🧠",
    layout="wide"
)

# Custom header
st.title("🧠 Self-Corrective Agentic RAG Engine")
st.caption("Hybrid Dense/Sparse Retrieval • Cross-Encoder Reranking • Local SLM Evaluation & Hallucination Guardrails")

# Initialize backend instances once in Streamlit cache
@st.cache_resource
def load_system():
    retriever = HybridRerankRetriever()
    evaluator = AgentEvaluator(model_name="phi3")
    agent = AgenticRAGGraph(retriever=retriever, evaluator=evaluator, model_name="phi3")
    
    # Load and index default knowledge base
    with open("data/company_policies.txt", "r") as f:
        policies = f.read()
    retriever.index_documents([policies])
    
    return retriever, agent

with st.spinner("Initializing Local Embeddings, Reranker, and Knowledge Base..."):
    retriever, agent = load_system()

# Sidebar: Document Management & Config
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    st.markdown("**LLM Runtime:** Local Ollama (`phi3`)")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Reranker:** `ms-marco-MiniLM-L-6-v2`")
    st.markdown("**Vector Store:** ChromaDB + BM25")
    
    st.divider()
    st.subheader("📄 Upload Extra Context")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8")
        if st.button("Index Uploaded Text"):
            retriever.index_documents([file_text])
            st.success("New content indexed successfully!")

# Main Query Area
st.subheader("🔍 Ask a Domain Question")
default_query = "What happens if schema drift recovery fails 3 times?"
user_query = st.text_input("Enter question:", value=default_query)

col1, col2 = st.columns([1, 1])

if st.button("Run Agentic Pipeline", type="primary"):
    with st.spinner("Executing agent decision graph..."):
        start_time = time.time()
        result = agent.run(user_query)
        latency = time.time() - start_time

    with col1:
        st.subheader("💡 Synthesized Output")
        st.success(result["generation"])
        st.caption(f"⏱️ Total pipeline cycle time: {latency:.2f}s")
        
        st.subheader("📑 Accepted Verified Chunks")
        if result["documents"]:
            for i, doc in enumerate(result["documents"], 1):
                st.info(f"**Chunk {i}:**\n\n{doc}")
        else:
            st.warning("No context chunks passed the relevance grading threshold.")

    with col2:
        st.subheader("🔄 Agent Execution Graph Trace")
        for step in result["trace"]:
            if "PASSED" in step or "Accepted" in step:
                st.markdown(f"✅ `{step}`")
            elif "Reformulating" in step or "Hallucination" in step:
                st.markdown(f"⚠️ `{step}`")
            else:
                st.markdown(f"🔹 `{step}`")