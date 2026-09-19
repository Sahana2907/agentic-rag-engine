from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRerankRetriever:
    def __init__(self, collection_name: str = "agentic_rag_docs"):
        # Local free embedding model running on CPU
        self.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Local cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed_model,
            persist_directory="./chroma_db"
        )
        self.bm25 = None
        self.doc_store: List[str] = []

    def index_documents(self, raw_texts: List[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
        chunks = []
        for text in raw_texts:
            chunks.extend(splitter.split_text(text))
        
        self.doc_store = chunks
        self.vector_store.add_texts(texts=chunks)
        
        # Build BM25 index
        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[+] Indexed {len(chunks)} chunks into ChromaDB & BM25 index.")

    def retrieve_and_rerank(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # 1. Dense retrieval (ChromaDB)
        dense_results = self.vector_store.similarity_search(query, k=4)
        dense_texts = [doc.page_content for doc in dense_results]

        # 2. Sparse retrieval (BM25 keyword search)
        sparse_texts = []
        if self.bm25 and self.doc_store:
            query_tokens = query.lower().split()
            sparse_texts = self.bm25.get_top_n(query_tokens, self.doc_store, n=4)

        # 3. Combine and remove duplicates
        candidate_pool = list(set(dense_texts + sparse_texts))
        if not candidate_pool:
            return []

        # 4. Rerank using CrossEncoder
        pairs = [[query, text] for text in candidate_pool]
        scores = self.reranker.predict(pairs)

        # 5. Sort by highest score
        ranked = sorted(zip(candidate_pool, scores), key=lambda x: x[1], reverse=True)
        return [{"chunk": item[0], "relevance_score": float(item[1])} for item in ranked[:top_k]]