from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        self.client = (
            OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=2)
            if self.api_key
            else None
        )
        self.last_usage: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.api_key)

    @property
    def mode(self) -> str:
        return "openai" if self.available else "offline"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY não configurada; usando modo offline.")
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return (response.choices[0].message.content or "").strip()

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float = 0.0,
    ) -> SchemaT:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY não configurada; usando modo offline.")
        response = self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=user,
            text_format=schema,
            temperature=temperature,
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        }
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI não retornou saída estruturada.")
        return parsed


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
