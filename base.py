# clients/base_client.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BaseLLMClient(ABC):
    """
    Minimal abstract interface for LLM clients used in the project.
    Keep this tiny so other client types (LangChain, Agentic, etc.) can match it.
    """

    @abstractmethod
    def generate_review(
        self,
        system_message_chat_conversation: str,
        ai_chunk: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Returns a dict with at least:
          {
            "result": { "All_Responses": [...] },
            "tokens": { "total_tokens": int, "prompt_tokens": int, "completion_tokens": int },
            "raw_text": str
          }
        """
        raise NotImplementedError

    @abstractmethod
    def format_responses_with_meta(
        self,
        result_payload: Dict[str, Any],
        all_no: int,
        chapter_titles: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Adds All_No and AI_chunk to each response item and returns (responses, full_result_dict).
        """
        raise NotImplementedError
