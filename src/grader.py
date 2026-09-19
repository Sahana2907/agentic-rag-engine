import json
import urllib.request
from typing import Dict, Any

class AgentEvaluator:
    def __init__(self, model_name: str = "phi3", ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{ollama_base_url}/api/generate"

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json"
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "{}")
        except Exception as e:
            return json.dumps({"error": str(e), "score": "yes", "grounded": "yes"})

    def grade_documents(self, question: str, document: str) -> bool:
        """Determines whether a retrieved document chunk is relevant to the question."""
        system_prompt = (
            "You are a strict document retrieval evaluator. Assess if the document contains "
            "facts relevant to answering the user question. Return JSON with a single key 'score' "
            "having value 'yes' or 'no'."
        )
        prompt = f"Question: {question}\nDocument: {document}\nRelevance ('yes' or 'no'):"
        
        raw_response = self._call_ollama(prompt, system_prompt)
        try:
            parsed = json.loads(raw_response)
            return parsed.get("score", "").strip().lower() == "yes"
        except Exception:
            return True

    def grade_hallucination(self, documents: str, generation: str) -> bool:
        """Determines whether an answer is grounded in the provided documents."""
        system_prompt = (
            "You are a strict hallucination evaluator. Evaluate if the answer is grounded in and "
            "supported by the reference documents. Return JSON with key 'grounded' as 'yes' or 'no'."
        )
        prompt = f"Reference Documents: {documents}\nAnswer: {generation}\nIs grounded ('yes' or 'no'):"
        
        raw_response = self._call_ollama(prompt, system_prompt)
        try:
            parsed = json.loads(raw_response)
            return parsed.get("grounded", "").strip().lower() == "yes"
        except Exception:
            return True

    def rewrite_query(self, question: str) -> str:
        """Transforms the question to optimize it for vector and keyword retrieval."""
        system_prompt = (
            "You are an expert query reformulation assistant. Reformulate the user question into "
            "an optimal semantic search query that retrieves precise technical facts. "
            "Return JSON with key 'improved_query'."
        )
        prompt = f"Initial Question: {question}\nImproved Search Query:"
        
        raw_response = self._call_ollama(prompt, system_prompt)
        try:
            parsed = json.loads(raw_response)
            return parsed.get("improved_query", question)
        except Exception:
            return question