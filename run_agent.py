from src.retriever import HybridRerankRetriever
from src.grader import AgentEvaluator
from src.graph import AgenticRAGGraph

# 1. Initialize Components
print("[*] Initializing Retriever & Evaluator...")
retriever = HybridRerankRetriever()
evaluator = AgentEvaluator(model_name="phi3")
agent = AgenticRAGGraph(retriever=retriever, evaluator=evaluator, model_name="phi3")

# 2. Index Policies
with open("data/company_policies.txt", "r") as f:
    policies = f.read()
retriever.index_documents([policies])

# 3. Execute Query
test_query = "What is the PagerDuty escalation trigger for schema drift?"
print(f"\nUser Query: {test_query}\n" + "="*50)

result = agent.run(test_query)

print("\n--- Execution Trace ---")
for step in result["trace"]:
    print(f" -> {step}")

print("\n--- Synthesized Answer ---")
print(result["generation"])