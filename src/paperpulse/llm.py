"""
LLM access behind a small interface, so providers can be swapped.

Only structured output is needed: the caller passes a Pydantic model and
gets back a validated instance of it.

    ollama     local models, the default; nothing leaves the machine
    anthropic  Claude via the API (`uv sync --extra anthropic`, ANTHROPIC_API_KEY);
               much faster than a local model on a laptop CPU
    groq       open models hosted by Groq (GROQ_API_KEY), e.g. gpt-oss-120b;
               fast, with a free tier whose token limits the client waits out
"""

import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from paperpulse.config import LLM as LLMConfig

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)
WAKE_POLL_SECONDS = 5


class LLMUnavailable(Exception):
    """The provider cannot serve requests at all, as opposed to one bad response."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, output: type[T]) -> T:
        """Return `output` parsed from the model's reply. Raises ValueError on a bad reply."""
        ...


class OllamaLLM:
    """Ollama's chat API with `format` set to a JSON schema, which constrains decoding to it."""

    # A complete assessment is ~150 tokens. Small models occasionally loop under
    # schema-constrained decoding and would otherwise write until the timeout.
    MAX_OUTPUT_TOKENS = 512

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
                    "options": {"temperature": 0, "num_predict": self.MAX_OUTPUT_TOKENS},
                },
            )
        except httpx.ConnectError as e:
            raise LLMUnavailable(f"cannot reach Ollama at {self.base_url}") from e
        except httpx.TimeoutException as e:
            # One stuck request is a bad answer for this paper, not an outage.
            raise ValueError(f"{self.name} timed out") from e
        if response.status_code == 404:
            raise LLMUnavailable(f"model {self.name!r} not found; run `ollama pull {self.name}`")
        if response.status_code >= 500:
            raise ValueError(f"{self.name} failed with HTTP {response.status_code}")
        response.raise_for_status()
        data = response.json()
        if data.get("done_reason") == "length":
            raise ValueError(f"{self.name} hit the {self.MAX_OUTPUT_TOKENS}-token output limit")
        return output.model_validate(json.loads(data["message"]["content"]))


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


class GroqLLM:
    """
    Groq's OpenAI-compatible chat API with a strict JSON schema. Rate limits
    (HTTP 429) are waited out when short; a long wait, such as an exhausted
    daily token budget, counts as the provider being unavailable.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"
    MAX_COMPLETION_TOKENS = 2048  # reasoning tokens plus the ~150-token answer
    MAX_RATE_LIMIT_WAIT = 120
    RATE_LIMIT_RETRIES = 5

    def __init__(
        self,
        model: str,
        effort: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.name = model
        self.effort = effort
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        self.client = client or httpx.Client(timeout=120)
        self.sleep = sleep

    def complete(self, system: str, user: str, output: type[T]) -> T:
        if not self.api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        body = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output.__name__,
                    "strict": True,
                    "schema": strict_schema(output.model_json_schema()),
                },
            },
            "temperature": 0,
            "max_completion_tokens": self.MAX_COMPLETION_TOKENS,
        }
        if self.name.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = self.effort
            body["include_reasoning"] = False

        for _ in range(self.RATE_LIMIT_RETRIES):
            response = self._post(body)
            if response.status_code != 429:
                break
            wait = float(response.headers.get("retry-after", 10))
            if wait > self.MAX_RATE_LIMIT_WAIT:
                raise LLMUnavailable(f"Groq rate limit, retry in {wait:.0f} s (daily budget?)")
            logger.info("Groq rate limit reached, waiting %.0f s", wait)
            self.sleep(wait)
        else:
            raise LLMUnavailable("Groq rate limit persists")

        if response.status_code in (401, 403):
            raise LLMUnavailable("Groq rejected GROQ_API_KEY")
        if response.status_code == 404:
            raise LLMUnavailable(f"Groq has no model {self.name!r}")
        if response.status_code >= 500:
            raise ValueError(f"{self.name} failed with HTTP {response.status_code}")
        if response.status_code == 400:
            # e.g. the model produced output that failed the schema server-side
            raise ValueError(f"{self.name} rejected the request: {response.text[:200]}")
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError(f"{self.name} hit the {self.MAX_COMPLETION_TOKENS}-token limit")
        return output.model_validate(json.loads(choice["message"]["content"]))

    def _post(self, body: dict) -> httpx.Response:
        try:
            return self.client.post(
                self.URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        except httpx.ConnectError as e:
            raise LLMUnavailable("cannot reach Groq") from e
        except httpx.TimeoutException as e:
            raise ValueError(f"{self.name} timed out") from e


# Keywords strict mode is not documented to support; bounds are enforced by
# validating the parsed answer instead.
_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "default"}


def strict_schema(schema: dict) -> dict:
    """Adapt a Pydantic JSON schema to strict mode: closed objects, every property required."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    result = {k: strict_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False
        result["required"] = list(result["properties"])
    return result


def wake_if_needed(
    config: LLMConfig,
    client: httpx.Client | None = None,
    run: Callable = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """
    True if the Ollama server answers, after running `wake_command` and waiting
    for it when it does not. Other providers are always considered awake.
    """
    if config.provider != "ollama":
        return True
    client = client or httpx.Client(timeout=3)
    url = f"{config.base_url.rstrip('/')}/api/version"

    def answers() -> bool:
        try:
            return client.get(url).status_code == 200
        except httpx.HTTPError:
            return False

    if answers():
        return True
    if not config.wake_command:
        return False
    logger.info("LLM server at %s is not answering; waking it", config.base_url)
    run(shlex.split(config.wake_command), check=False)
    for _ in range(max(1, config.wake_timeout // WAKE_POLL_SECONDS)):
        sleep(WAKE_POLL_SECONDS)
        if answers():
            logger.info("LLM server is up")
            return True
    logger.warning("LLM server did not wake within %d s", config.wake_timeout)
    return False


def create_llm(config: LLMConfig) -> LLM:
    if config.provider == "anthropic":
        return AnthropicLLM(config.resolved_model, config.effort)
    if config.provider == "groq":
        return GroqLLM(config.resolved_model, config.effort)
    return OllamaLLM(config.resolved_model, config.base_url)
