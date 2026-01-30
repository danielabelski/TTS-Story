"""Helper for interacting with local LLM servers (LM Studio or Ollama)."""

from __future__ import annotations

import logging
from typing import Optional

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

    def _generate_openai_compatible(self, prompt: str) -> str:
        url = f"{self._normalize_openai_base()}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

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
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

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
