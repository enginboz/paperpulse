"""
LLM access behind a small interface, so providers can be swapped.

Only structured output is needed: the caller passes a Pydantic model and
gets back a validated instance of it.

    ollama     local models, the default; nothing leaves the machine
    anthropic  Claude via the API (`uv sync --extra anthropic`, ANTHROPIC_API_KEY);
               much faster than a local model on a laptop CPU
"""

import json
import logging
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
    return OllamaLLM(config.resolved_model, config.base_url)
