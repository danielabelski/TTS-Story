"""Helper for interacting with local LLM servers (LM Studio or Ollama)."""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

LLM_PROVIDER_LMSTUDIO = "lmstudio"
LLM_PROVIDER_OLLAMA = "ollama"

SUPPORTED_LOCAL_LLM_PROVIDERS = [LLM_PROVIDER_LMSTUDIO, LLM_PROVIDER_OLLAMA]

DEFAULT_LOCAL_LLM_BASE_URLS = {
    LLM_PROVIDER_LMSTUDIO: "http://localhost:1234/v1",
    LLM_PROVIDER_OLLAMA: "http://localhost:11434",
}


class LocalLLMProcessorError(RuntimeError):
    """Raised when local LLM processing fails."""


class LocalLLMProcessor:
    """Wrapper for local LLM providers."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        disable_reasoning: bool = False,
    ) -> None:
        provider = (provider or "").lower().strip()
        if provider not in SUPPORTED_LOCAL_LLM_PROVIDERS:
            raise LocalLLMProcessorError(f"Unsupported local LLM provider: {provider}")

        if not model_name:
            raise LocalLLMProcessorError("Local LLM model name is required")

        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url or DEFAULT_LOCAL_LLM_BASE_URLS.get(provider, "")
        self.api_key = api_key.strip() if api_key else ""
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.max_tokens = max_tokens
        self.disable_reasoning = disable_reasoning

        if not self.base_url:
            raise LocalLLMProcessorError("Local LLM base URL is required")

    def generate_text(self, prompt: str) -> str:
        """Send prompt to local LLM and return the text response."""
        if not prompt.strip():
            raise LocalLLMProcessorError("Prompt must not be empty")

        if self.provider == LLM_PROVIDER_LMSTUDIO:
            return self._generate_openai_compatible(prompt)

        if self.provider == LLM_PROVIDER_OLLAMA:
            return self._generate_ollama(prompt)

        raise LocalLLMProcessorError(f"Unsupported local LLM provider: {self.provider}")

    def _normalize_openai_base(self) -> str:
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _normalize_openai_base_url(base_url: str) -> str:
        base = (base_url or "").rstrip("/")
        if not base:
            return ""
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @classmethod
    def list_available_models(
        cls,
        provider: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> List[str]:
        provider = (provider or "").lower().strip()
        if provider not in SUPPORTED_LOCAL_LLM_PROVIDERS:
            raise LocalLLMProcessorError(f"Unsupported local LLM provider: {provider}")

        resolved_base = base_url or DEFAULT_LOCAL_LLM_BASE_URLS.get(provider, "")
        if not resolved_base:
            raise LocalLLMProcessorError("Local LLM base URL is required")

        if provider == LLM_PROVIDER_LMSTUDIO:
            url = f"{cls._normalize_openai_base_url(resolved_base)}/models"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
            except Exception as exc:
                logging.error("Local LLM model list failed: %s", exc, exc_info=True)
                raise LocalLLMProcessorError(f"Local LLM model list failed: {exc}") from exc

            if response.status_code >= 400:
                raise LocalLLMProcessorError(
                    f"Local LLM error ({response.status_code}): {response.text.strip()}"
                )

            data = response.json() if response.content else {}
            models = [entry.get("id") for entry in data.get("data", []) if entry.get("id")]
            if not models:
                raise LocalLLMProcessorError("No LM Studio models were returned")
            return sorted(models)

        if provider == LLM_PROVIDER_OLLAMA:
            url = f"{resolved_base.rstrip('/')}/api/tags"
            try:
                response = requests.get(url, timeout=timeout)
            except Exception as exc:
                logging.error("Ollama model list failed: %s", exc, exc_info=True)
                raise LocalLLMProcessorError(f"Ollama model list failed: {exc}") from exc

            if response.status_code >= 400:
                raise LocalLLMProcessorError(
                    f"Ollama error ({response.status_code}): {response.text.strip()}"
                )

            data = response.json() if response.content else {}
            models = [entry.get("name") for entry in data.get("models", []) if entry.get("name")]
            if not models:
                raise LocalLLMProcessorError("No Ollama models were returned")
            return sorted(models)

        raise LocalLLMProcessorError(f"Unsupported local LLM provider: {provider}")

    def _generate_openai_compatible(self, prompt: str) -> str:
        url = f"{self._normalize_openai_base()}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.top_k is not None and int(self.top_k) > 0:
            payload["top_k"] = int(self.top_k)
        if self.repeat_penalty is not None:
            payload["repeat_penalty"] = float(self.repeat_penalty)
        if self.max_tokens is not None and int(self.max_tokens) > 0:
            payload["max_tokens"] = int(self.max_tokens)
        if self.disable_reasoning:
            payload["reasoning"] = {"enabled": False}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except Exception as exc:
            logging.error("Local LLM request failed: %s", exc, exc_info=True)
            raise LocalLLMProcessorError(f"Local LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LocalLLMProcessorError(
                f"Local LLM error ({response.status_code}): {response.text.strip()}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LocalLLMProcessorError("Local LLM response did not include any choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise LocalLLMProcessorError("Local LLM response did not contain any text")

        return str(content).strip()

    def _generate_ollama(self, prompt: str) -> str:
        base = self.base_url.rstrip("/")
        url = f"{base}/api/chat"
        options = {}
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.top_p is not None:
            options["top_p"] = float(self.top_p)
        if self.top_k is not None and int(self.top_k) > 0:
            options["top_k"] = int(self.top_k)
        if self.repeat_penalty is not None:
            options["repeat_penalty"] = float(self.repeat_penalty)
        if self.max_tokens is not None and int(self.max_tokens) > 0:
            options["num_predict"] = int(self.max_tokens)
        if self.disable_reasoning:
            options["reasoning"] = False

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if options:
            payload["options"] = options

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except Exception as exc:
            logging.error("Ollama request failed: %s", exc, exc_info=True)
            raise LocalLLMProcessorError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LocalLLMProcessorError(
                f"Ollama error ({response.status_code}): {response.text.strip()}"
            )

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise LocalLLMProcessorError("Ollama response did not contain any text")

        return str(content).strip()
