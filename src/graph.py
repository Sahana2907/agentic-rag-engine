import json
import urllib.request
from typing import Dict, Any, List
from src.retriever import HybridRerankRetriever
from src.grader import AgentEvaluator

class AgenticRAGGraph:
    def __init__(self, retriever: HybridRerankRetriever, evaluator: AgentEvaluator, model_name: str = "phi3"):
        self.retriever = retriever
        self.evaluator = evaluator
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def _generate_answer(self, question: str, context: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": f"Context:\n{context}\n\nQuestion: {question}\nProvide a concise and factual answer based solely on the context:",
            "stream": False
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def run(self, initial_query: str, max_retries: int = 2) -> Dict[str, Any]:
        state = {
            "question": initial_query,
            "current_query": initial_query,
            "documents": [],
            "generation": "",
            "retries": 0,
            "trace": []
        }

        while state["retries"] <= max_retries:
            # 1. Retrieve & Rerank
            state["trace"].append(f"Retrieving for: '{state['current_query']}'")
            ranked_results = self.retriever.retrieve_and_rerank(state["current_query"], top_k=2)
            raw_docs = [r["chunk"] for r in ranked_results]

            # 2. Grade Documents for Relevance
            relevant_docs = []
            for doc in raw_docs:
                is_rel = self.evaluator.grade_documents(state["question"], doc)
                if is_rel:
                    relevant_docs.append(doc)

            if not relevant_docs:
                # No relevant documents -> Rewrite Query and Retry Loop
                state["retries"] += 1
                state["trace"].append(f"No relevant docs found. Reformulating query (Attempt {state['retries']}/{max_retries})...")
                state["current_query"] = self.evaluator.rewrite_query(state["question"])
                continue

            state["documents"] = relevant_docs
            state["trace"].append(f"Accepted {len(relevant_docs)} relevant context chunk(s).")

            # 3. Generate Answer
            context_str = "\n\n".join(state["documents"])
            state["trace"].append("Synthesizing answer with SLM...")
            generation = self._generate_answer(state["question"], context_str)

            # 4. Check for Hallucination
            is_grounded = self.evaluator.grade_hallucination(context_str, generation)
            if is_grounded:
                state["generation"] = generation
                state["trace"].append("Groundedness check: PASSED.")
                break
            else:
                state["retries"] += 1
                state["trace"].append(f"Hallucination detected. Re-evaluating (Attempt {state['retries']}/{max_retries})...")

        if not state["generation"]:
            state["generation"] = "Unable to answer based strictly on verified domain documents."

        return state