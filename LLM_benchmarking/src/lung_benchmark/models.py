from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import requests


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _normalize_label(label: Any) -> int:
    s = str(label).strip().upper()
    if s in {"Y", "YES", "1", "TRUE", "CANCEROUS"}:
        return 1
    if s in {"N", "NO", "0", "FALSE", "BENIGN"}:
        return 0
    return 0


@dataclass
class ModelSpec:
    model_id: str
    family: str
    provider: str
    endpoint: str
    api_key_env: str
    temperature: float = 0.0


class BaseModelRunner:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def fit(self, numeric_matrix: np.ndarray, labels: np.ndarray) -> None:
        return None

    def predict_batch(
        self,
        prompts: List[str],
        numeric_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class OpenAICompatibleRunner(BaseModelRunner):
    def _post(self, prompt: str) -> Dict[str, Any]:
        key = os.getenv(self.spec.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Missing API key env var '{self.spec.api_key_env}' for model {self.spec.model_id}"
            )
        if not self.spec.endpoint:
            raise RuntimeError(f"Missing endpoint for model {self.spec.model_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        # OpenRouter-specific attribution headers (optional but recommended)
        site_url = os.getenv("OPENROUTER_SITE_URL", "")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "")
        if "openrouter.ai" in self.spec.endpoint:
            if site_url:
                headers["HTTP-Referer"] = site_url
            if site_name:
                headers["X-Title"] = site_name
        payload = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.spec.temperature,
            "response_format": {"type": "json_object"},
        }
        res = requests.post(self.spec.endpoint, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        return _extract_json_object(content)

    def predict_batch(
        self,
        prompts: List[str],
        numeric_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        labels: List[int] = []
        probs: List[float] = []
        for prompt in prompts:
            obj = _normalize_response(self._post(prompt))
            labels.append(obj["label"])
            probs.append(obj["probability"])
        return {"labels": np.asarray(labels, dtype=int), "probs": np.asarray(probs, dtype=float)}


class AnthropicCompatibleRunner(BaseModelRunner):
    def _post(self, prompt: str) -> Dict[str, Any]:
        key = os.getenv(self.spec.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Missing API key env var '{self.spec.api_key_env}' for model {self.spec.model_id}"
            )
        if not self.spec.endpoint:
            raise RuntimeError(f"Missing endpoint for model {self.spec.model_id}")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.spec.model_id,
            "max_tokens": 256,
            "temperature": self.spec.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        res = requests.post(self.spec.endpoint, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        text = ""
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return _extract_json_object(text)

    def predict_batch(
        self,
        prompts: List[str],
        numeric_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        labels: List[int] = []
        probs: List[float] = []
        for prompt in prompts:
            obj = _normalize_response(self._post(prompt))
            labels.append(obj["label"])
            probs.append(obj["probability"])
        return {"labels": np.asarray(labels, dtype=int), "probs": np.asarray(probs, dtype=float)}


def _normalize_response(obj: Dict[str, Any]) -> Dict[str, Any]:
    label = _normalize_label(obj.get("label", 0))
    try:
        p = float(obj.get("probability", 0.5))
    except (TypeError, ValueError):
        p = 0.5
    p = max(0.0, min(1.0, p))
    return {"label": label, "probability": p}


def model_specs_from_config(config: Dict[str, Any]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for m in config.get("models", []):
        specs.append(
            ModelSpec(
                model_id=str(m["model_id"]),
                family=str(m.get("family", "")),
                provider=str(m.get("provider", "")),
                endpoint=str(m.get("endpoint", "")),
                api_key_env=str(m.get("api_key_env", "")),
                temperature=float(m.get("temperature", 0.0)),
            )
        )
    return specs


def get_runner(spec: ModelSpec) -> BaseModelRunner:
    if spec.provider == "openai-compatible":
        return OpenAICompatibleRunner(spec)
    if spec.provider == "anthropic-compatible":
        return AnthropicCompatibleRunner(spec)
    raise ValueError(
        f"Unsupported model provider/family for model {spec.model_id}: {spec.provider}/{spec.family}"
    )
