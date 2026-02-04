from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json
import os
import random


def _stable_seed(seed: int, messages: List[Dict[str, Any]]) -> int:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class StubLLMAdapter(LLMAdapter):
    seed: int = 123

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        rng = random.Random(_stable_seed(self.seed, messages))
        token = rng.randint(100000, 999999)
        return f"STUB_RESPONSE_{token}"

    def healthcheck(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "stub", "seed": self.seed}


@dataclass
class ProviderLLMAdapter(LLMAdapter):
    backend: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        from opto.utils.llm import LLM
        kwargs: Dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        self._llm = LLM(backend=self.backend, **kwargs)

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = self._llm(messages=messages, **kwargs)
        # Try the common OpenAI/LiteLLM response shape
        if hasattr(response, "choices"):
            try:
                return response.choices[0].message.content
            except Exception:
                pass
        if isinstance(response, dict):
            try:
                return response["choices"][0]["message"]["content"]
            except Exception:
                pass
        return str(response)

    def healthcheck(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "real", "backend": self.backend, "model": self.model}


def llm_from_config(mode: str, llm_cfg: Dict[str, Any]) -> LLMAdapter:
    provider = llm_cfg.get("provider") or llm_cfg.get("backend")
    if mode == "stub" or provider == "stub":
        return StubLLMAdapter(seed=int(llm_cfg.get("seed", 123)))
    backend = llm_cfg.get("backend") or os.getenv("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
    model = llm_cfg.get("model") or os.getenv("TRACE_LITELLM_MODEL")
    return ProviderLLMAdapter(backend=backend, model=model)


__all__ = ["LLMAdapter", "StubLLMAdapter", "ProviderLLMAdapter", "llm_from_config"]
