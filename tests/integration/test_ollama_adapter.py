from __future__ import annotations

import os

import httpx
import pytest

from finangpt.infrastructure.llm.ollama_service import OllamaLLMService


def _ollama_available(url: str) -> bool:
    try:
        httpx.get(f"{url.rstrip('/')}/api/tags", timeout=1)
        return True
    except Exception:
        return False


@pytest.mark.integration
def test_generate_text_smoke(monkeypatch):
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if not _ollama_available(base_url):
        pytest.skip("Ollama endpoint unavailable")

    model = os.getenv("OLLAMA_MODEL", "phi4:latest")
    service = OllamaLLMService(base_url=base_url, model=model, timeout=10, max_retries=2)

    output = service.generate_text("Respond with the word FINANGPT only.")

    service.close()
    assert isinstance(output, str)
    assert output.strip()
