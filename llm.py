# clients/llm_client.py
from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Tuple, Optional

from pydantic import BaseModel, Field, ValidationError
from openai import AzureOpenAI

from .base_client import BaseLLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas (kept close to the client)
# ──────────────────────────────────────────────────────────────────────────────

class ReviewPoint(BaseModel):
    No: int = Field(description="1からの連番")
    target_chunk: str = Field(..., description="レビュー指摘箇所を抽出して明示")
    point_why: str = Field(..., description="指摘理由の説明")
    point_imp: str = Field(..., description="重要性の説明")
    pointout_level: str = Field(..., description="重要度レベル (例: low/medium/high)")
    pointout_category: int = Field(..., description="カテゴリID (整数)")
    break_rules: bool = Field(..., description="ルール違反かどうか")

class ReviewResponse(BaseModel):
    All_Responses: List[ReviewPoint] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient(BaseLLMClient):
    """
    Azure OpenAI chat client with strict JSON output and reliable token accounting.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",        # keep your familiar default; change per your deployment name
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        request_timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        endpoint = os.getenv("AOAI_ENDPOINT")
        api_key = os.getenv("AOAI_API_KEY")
        api_version = os.getenv("AOAI_API_VERSION")
        missing = [name for name, val in [
            ("AOAI_ENDPOINT", endpoint), ("AOAI_API_KEY", api_key), ("AOAI_API_VERSION", api_version)
        ] if not val]
        if missing:
            raise RuntimeError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=self.request_timeout,
        )

        # Prebuilt system rules to force JSON
        self._json_contract = (
            "You MUST respond ONLY with a single valid JSON object matching this schema:\n"
            "{\n"
            '  "All_Responses": [\n'
            "    {\n"
            '      "No": number,\n'
            '      "target_chunk": string,\n'
            '      "point_why": string,\n'
            '      "point_imp": string,\n'
            '      "pointout_level": string,\n'
            '      "pointout_category": number,\n'
            '      "break_rules": boolean\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Do NOT include any extra keys, comments, markdown, or code fences."
        )

    # ---------- public API ----------

    def generate_review(
        self,
        system_message_chat_conversation: str,
        ai_chunk: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        JSON-only review call. Accepts common alias typos and returns a stable dict.
        """

        # Accept older/typo param names transparently
        sys_msg = kwargs.pop("system_message_chat_conversion", None) or system_message_chat_conversation
        user_chunk = kwargs.pop("AI_chunk", None) or ai_chunk

        messages = [
            {"role": "system", "content": f"{sys_msg}\n\n{self._json_contract}"},
            {"role": "user", "content": user_chunk},
        ]

        # Try with response_format=json_object (where supported by Azure model)
        use_response_format = True  # harmless if unsupported; Azure ignores for older models

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs_call = dict(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                )
                if use_response_format:
                    kwargs_call["response_format"] = {"type": "json_object"}
                # tokens control: max_tokens may be optional on Azure newer models
                kwargs_call["max_tokens"] = self.max_output_tokens

                resp = self.client.chat.completions.create(**kwargs_call)

                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)

                tokens = {
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                }

                parsed = self._parse_json_strict(content)
                validated = self._validate_payload(parsed)

                return {"result": validated, "tokens": tokens, "raw_text": content}

            except Exception as e:
                last_error = e
                # Retry after a short backoff (handles transient 429/500/timeout)
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                # On final failure, produce a safe empty result with error info in raw_text
                return {
                    "result": {"All_Responses": []},
                    "tokens": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    "raw_text": f"[LLMClient Error]: {type(e).__name__}: {e}",
                }

    def format_responses_with_meta(
        self,
        result_payload: Dict[str, Any],
        all_no: int,
        chapter_titles: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Accepts either:
          • the raw model payload: {"All_Responses": [...]}
          • OR the full dict from generate_review(...): {"result": {...}, "tokens": {...}, "raw_text": "..."}
        Returns (augmented_list, {"result": {"All_Responses": augmented}})
        """
        # If caller passed the whole generate_review dict, unwrap it
        if "result" in result_payload and "All_Responses" not in result_payload:
            result_payload = result_payload.get("result", {})

        all_responses = list(result_payload.get("All_Responses", []))
        augmented: List[Dict[str, Any]] = []

        for item in all_responses:
            row = dict(item)
            row["All_No"] = all_no
            row["AI_chunk"] = chapter_titles
            augmented.append(row)
            all_no += 1

        return augmented, {"result": {"All_Responses": augmented}}

    # ---------- internals ----------

    def _parse_json_strict(self, text: str) -> Dict[str, Any]:
        """
        Accept clean JSON or JSON fenced in ```...```. Refuse non-JSON politely.
        """
        t = text.strip()
        # Strip markdown code fences if present
        if t.startswith("```"):
            # common variants: ```json ... ```  or ``` ... ```
            first = t.find("\n")
            if first != -1:
                t = t[first + 1:]
            if t.endswith("```"):
                t = t[:-3].strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            # try to extract the first {...} block
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = t[start:end+1]
                return json.loads(candidate)
            # give up: return empty contract
            return {"All_Responses": []}

    def _validate_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate against Pydantic model; coerce to plain dict.
        """
        try:
            return ReviewResponse.model_validate(data).model_dump()
        except ValidationError:
            # If the model returns partial data, try to coerce item-by-item.
            items = data.get("All_Responses", [])
            cleaned: List[Dict[str, Any]] = []
            for idx, it in enumerate(items, start=1):
                try:
                    cleaned.append(ReviewPoint.model_validate(it).model_dump())
                except ValidationError:
                    # best-effort fill
                    cleaned.append({
                        "No": it.get("No", idx),
                        "target_chunk": it.get("target_chunk", ""),
                        "point_why": it.get("point_why", ""),
                        "point_imp": it.get("point_imp", ""),
                        "pointout_level": it.get("pointout_level", "medium"),
                        "pointout_category": int(it.get("pointout_category", 0)) if str(it.get("pointout_category", "0")).isdigit() else 0,
                        "break_rules": bool(it.get("break_rules", False)),
                    })
            return {"All_Responses": cleaned}
