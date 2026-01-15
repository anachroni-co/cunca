"""
Minimal RAG pipeline aligned with Advanced RAG and prompt utilities.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from capibara.utils.logging import get_logger

from capibara.prompts import format_markdown_response

logger = get_logger(__name__)


@dataclass
class RAGContext:
    query: str
    context: str
    documents: List[Dict[str, Any]]


class RAGIntegrator:
    """Minimal RAG integrator.

    Expects a `rag_system` with method `retrieve_documents(query, k)` that returns
    documents with at least keys: {'text': str, 'score': float}.
    Optionally can have `generate_context(query, max_tokens)`.
    """

    def __init__(self, rag_system: Any, context_template: str = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"):
        self.rag_system = rag_system
        self.context_template = context_template

    def prepare_prompt(self, query: str, k: int = 3, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Prepare a prompt with RAG-retrieved context."""
        docs = []
        try:
            docs = self.rag_system.retrieve_documents(query, k)
        except Exception as e:
            logger.warning(f"RAG retrieve_documents failed, using empty: {e}")
            docs = []

        if hasattr(self.rag_system, "generate_context"):
            try:
                ctx_res = self.rag_system.generate_context(query, max_tokens=max_tokens or 2048)
                context_text = ctx_res.get("context", "")
            except Exception as e:
                logger.warning(f"generate_context failed, building simple context: {e}")
                context_text = "\n\n".join(doc.get("text", "") for doc in docs)
        else:
            context_text = "\n\n".join(doc.get("text", "") for doc in docs)

        prompt = self.context_template.format(context=context_text, query=query)
        return {
            "prompt": prompt,
            "context": context_text,
            "documents": docs,
        }

    def process_response(self, response: str, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns basic metadata about the response and documents used."""
        avg_score = (sum(d.get("score", 0.0) for d in docs) / len(docs)) if docs else 0.0
        return {
            "response": response,
            "query": query,
            "documents_used": len(docs),
            "avg_doc_score": avg_score,
        }

    def verify_response(self, response: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Minimal heuristic verification of context coverage."""
        try:
            resp_words = set(response.lower().split())
            ctx_words: set[str] = set()
            for doc in docs:
                text = doc.get("text", "")
                ctx_words.update(text.lower().split())
            coverage = (len(resp_words & ctx_words) / max(len(resp_words), 1)) if resp_words else 0.0
            length_ok = 10 <= len(response.split()) <= 1000
            score = max(0.0, min(1.0, coverage * (1.0 if length_ok else 0.8)))
            return {"coverage": coverage, "length_ok": length_ok, "verification_score": score}
        except Exception as e:
            logger.error(f"verify_response error: {e}")
            return {"coverage": 0.0, "length_ok": False, "verification_score": 0.5}


__all__ = ["RAGContext", "RAGIntegrator"]