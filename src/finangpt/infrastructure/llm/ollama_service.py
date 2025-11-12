"""Ollama-backed LLM service with structured JSON responses."""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

__all__ = ["OllamaLLMService", "LLMServiceError", "LLMResponseValidationError"]

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(RuntimeError):
    """Base error for LLM adapter failures."""


class LLMResponseValidationError(LLMServiceError):
    """Raised when the LLM response cannot be coerced into the target schema."""


class OllamaLLMService:
    """Thin HTTP client around the Ollama API with JSON-mode helpers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        retry_wait_seconds: float = 0.1,
    ) -> None:
        self._base_url = base_url.rstrip("/") or "http://localhost:11434"
        self._model = model
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._retry_wait = max(0.0, retry_wait_seconds)
        self._client = client or httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def close(self) -> None:
        self._client.close()

    def generate_text(self, prompt: str) -> str:
        """Generate free-form text from the model."""

        return self._invoke(prompt=prompt, json_mode=False)

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Generate JSON output validated against the provided schema."""

        attempt_prompt = prompt
        last_error: str | None = None
        for attempt in range(2):
            raw = self._invoke(prompt=attempt_prompt, json_mode=True)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                last_error = f"Invalid JSON: {exc}"
            else:
                try:
                    return schema.model_validate(payload)
                except ValidationError as exc:
                    last_error = exc.json()
            if attempt == 0:
                attempt_prompt = self._augment_prompt(prompt, last_error)
                continue
            raise LLMResponseValidationError(last_error or "Structured output validation failed")
        raise LLMResponseValidationError(last_error or "Structured output validation failed")

    def _augment_prompt(self, original_prompt: str, error: str | None) -> str:
        correction = error or "Unknown validation error"
        return (
            f"{original_prompt}\n\n"
            "IMPORTANT: The previous response failed JSON schema validation. "
            f"Validation details: {correction}. Respond again with STRICT JSON only."
        )

    def _invoke(self, *, prompt: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        retrying = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_wait),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._send_request(payload)
        except RetryError as exc:  # pragma: no cover - network failure
            raise LLMServiceError("Failed to reach Ollama after retries") from exc
        raise LLMServiceError("Failed to reach Ollama")  # pragma: no cover - safety

    def _send_request(self, payload: dict[str, Any]) -> str:
        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise LLMServiceError("Ollama response missing 'response' field")
        return str(data["response"])
