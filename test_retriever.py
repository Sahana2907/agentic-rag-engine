from src.retriever import HybridRerankRetriever

print("Loading Hybrid Retriever and Local Models...")
retriever = HybridRerankRetriever()

# Read the document
with open("data/company_policies.txt", "r") as f:
    text = f.read()

# Index the content
retriever.index_documents([text])

# Ask a test question
query = "What happens if schema drift recovery fails 3 times?"
print(f"\nQuery: {query}")
results = retriever.retrieve_and_rerank(query, top_k=2)

for idx, res in enumerate(results, 1):
    print(f"\n--- Rank {idx} (Relevance Score: {res['relevance_score']:.4f}) ---")
    print(res["chunk"])