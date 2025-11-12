from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from finangpt.infrastructure.llm.ollama_service import OllamaLLMService


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
        self.requests.append({"path": path, "json": json})
        if not self._responses:
            raise AssertionError("Unexpected call")
        return self._responses.pop(0)


class DemoSchema(BaseModel):
    field: str


def test_generate_structured_validates_against_schema():
    client = FakeClient([FakeResponse({"response": '{"field": "value"}'})])
    service = OllamaLLMService(
        base_url="http://test",
        model="demo",
        timeout=1,
        max_retries=1,
        client=client,
        retry_wait_seconds=0,
    )

    result = service.generate_structured("prompt", DemoSchema)

    assert result.field == "value"
    assert len(client.requests) == 1


def test_generate_structured_retries_with_correction_on_validation_error():
    client = FakeClient(
        [
            FakeResponse({"response": '{"wrong": "value"}' }),
            FakeResponse({"response": '{"field": "fixed"}' }),
        ]
    )
    service = OllamaLLMService(
        base_url="http://test",
        model="demo",
        timeout=1,
        max_retries=1,
        client=client,
        retry_wait_seconds=0,
    )

    result = service.generate_structured("prompt", DemoSchema)

    assert result.field == "fixed"
    assert len(client.requests) == 2
    assert "json" in client.requests[1]["json"]["prompt"].lower()
