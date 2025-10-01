# clients/llm_client.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import AzureOpenAI

# --- Schema the model must return (JSON only) ---
class ReviewPoint(BaseModel):
    No: int = Field(description="1からの連番")
    target_chunk: str
    point_why: str
    point_imp: str
    pointout_level: str
    pointout_category: int
    break_rules: bool

class ReviewResponse(BaseModel):
    All_Responses: List[ReviewPoint] = Field(default_factory=list)

class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4.1",          # IMPORTANT on Azure: this must be your DEPLOYMENT NAME
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
        missing = [n for n,v in [("AOAI_ENDPOINT",endpoint),("AOAI_API_KEY",api_key),("AOAI_API_VERSION",api_version)] if not v]
        if missing:
            raise RuntimeError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=request_timeout,
        )

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
            "Do NOT include markdown or code fences."
        )

    # ---------- main call ----------
    def generate_review(
        self,
        system_message_chat_conversation: str,
        ai_chunk: str,
        **_: Any
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": f"{system_message_chat_conversation}\n\n{self._json_contract}"},
            {"role": "user", "content": ai_chunk},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,                         # deployment name
                    temperature=self.temperature,
                    messages=messages,
                    response_format={"type": "json_object"}, # JSON-only where supported
                    max_tokens=self.max_output_tokens,
                )
                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                tokens = {
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                }

                parsed = self._parse_json(content)
                validated = self._validate_payload(parsed)
                return {"result": validated, "tokens": tokens, "raw_text": content}

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return {
                    "result": {"All_Responses": []},
                    "tokens": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    "raw_text": f"[LLMClient Error] {type(e).__name__}: {e}",
                }

    def format_responses_with_meta(
        self,
        result_payload: Dict[str, Any],
        all_no: int,
        chapter_titles: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Allow callers to pass the full dict from generate_review(...)
        if "result" in result_payload and "All_Responses" not in result_payload:
            result_payload = result_payload["result"]

        rows_in = list(result_payload.get("All_Responses", []))
        out: List[Dict[str, Any]] = []
        for item in rows_in:
            row = dict(item)
            row["All_No"] = all_no
            row["AI_chunk"] = chapter_titles
            out.append(row)
            all_no += 1
        return out, {"result": {"All_Responses": out}}

    # ---------- internals ----------
    def _parse_json(self, text: str) -> Dict[str, Any]:
        t = text.strip()
        if t.startswith("```"):
            first = t.find("\n")
            if first != -1: t = t[first+1:]
            if t.endswith("```"): t = t[:-3].strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            start, end = t.find("{"), t.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(t[start:end+1])
                except Exception:
                    pass
            return {"All_Responses": []}

    def _validate_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return ReviewResponse.model_validate(data).model_dump()
        except ValidationError:
            items = data.get("All_Responses", [])
            cleaned: List[Dict[str, Any]] = []
            for idx, it in enumerate(items, start=1):
                try:
                    cleaned.append(ReviewPoint.model_validate(it).model_dump())
                except ValidationError:
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
