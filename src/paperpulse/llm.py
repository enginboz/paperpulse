"""
LLM access behind a small interface, so providers can be swapped.

Only structured output is needed: the caller passes a Pydantic model and
gets back a validated instance of it.

    ollama     local models, the default; nothing leaves the machine
    anthropic  Claude via the API (`uv sync --extra anthropic`, ANTHROPIC_API_KEY);
               much faster than a local model on a laptop CPU
"""

import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from paperpulse.config import LLM as LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """The provider cannot serve requests at all, as opposed to one bad response."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, output: type[T]) -> T:
        """Return `output` parsed from the model's reply. Raises ValueError on a bad reply."""
        ...


class OllamaLLM:
    """Ollama's chat API with `format` set to a JSON schema, which constrains decoding to it."""

    def __init__(self, model: str, base_url: str, client: httpx.Client | None = None):
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=300)

    def complete(self, system: str, user: str, output: type[T]) -> T:
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "format": output.model_json_schema(),
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
        except httpx.ConnectError as e:
            raise LLMUnavailable(f"cannot reach Ollama at {self.base_url}") from e
        if response.status_code == 404:
            raise LLMUnavailable(f"model {self.name!r} not found; run `ollama pull {self.name}`")
        response.raise_for_status()
        return output.model_validate(json.loads(response.json()["message"]["content"]))


class AnthropicLLM:
    """Claude with structured outputs; the SDK converts and validates the Pydantic model."""

    def __init__(self, model: str, effort: str, client=None):
        self.name = model
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMUnavailable(
                    "the anthropic provider needs the optional dependency: "
                    "uv sync --extra anthropic"
                ) from e
            # Resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
            client = anthropic.Anthropic()
        self.client = client

    def complete(self, system: str, user: str, output: type[T]) -> T:
        import anthropic

        try:
            response = self.client.beta.messages.parse(
                model=self.name,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=output,
                output_config={"effort": self.effort},
                # If a safety classifier declines, retry server-side on the recommended model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        ) as e:
            raise LLMUnavailable(f"Anthropic API: {e}") from e

        if response.stop_reason == "refusal":
            raise ValueError("the model declined to assess this paper")
        if response.parsed_output is None:
            raise ValueError(f"no structured output (stop_reason={response.stop_reason})")
        return response.parsed_output


def create_llm(config: LLMConfig) -> LLM:
    if config.provider == "anthropic":
        return AnthropicLLM(config.resolved_model, config.effort)
    return OllamaLLM(config.resolved_model, config.base_url)
