import importlib.util
import json
from types import SimpleNamespace

import httpx
import pytest

from paperpulse.config import LLM as LLMConfig
from paperpulse.config import Profile
from paperpulse.llm import AnthropicLLM, LLMUnavailable, OllamaLLM, create_llm
from paperpulse.models import Assessment
from paperpulse.ranking.assess import (
    MAX_ABSTRACT_CHARS,
    assess_papers,
    build_system_prompt,
    build_user_prompt,
    cache_key,
)
from tests.conftest import FakeLLM, make_paper

PROFILE = Profile(topics=["FHIR"], preferences="Prefer original research.")
VALID = {"study_type": "original_research", "rationale": "Solid.", "relevance": 4}


def test_system_prompt_lists_topics_and_preferences():
    prompt = build_system_prompt(PROFILE)
    assert "- FHIR" in prompt
    assert "Prefer original research." in prompt
    assert "preferences" not in build_system_prompt(Profile(topics=["FHIR"]))


def test_user_prompt_truncates_long_abstracts():
    paper = make_paper(1, abstract="x" * (MAX_ABSTRACT_CHARS + 500), publication_types=["Review"])
    prompt = build_user_prompt(paper)
    assert "Publication types: Review" in prompt
    assert prompt.count("x") == MAX_ABSTRACT_CHARS


def test_assessments_are_cached(store):
    store.upsert_papers([make_paper(1)])
    llm = FakeLLM({"Paper 1": 4})
    first = assess_papers(store, llm, PROFILE, [make_paper(1)])
    second = assess_papers(store, llm, PROFILE, [make_paper(1)])
    assert first == second
    assert len(llm.calls) == 1


def test_changing_the_profile_invalidates_the_cache(store):
    store.upsert_papers([make_paper(1)])
    llm = FakeLLM()
    assess_papers(store, llm, PROFILE, [make_paper(1)])
    assess_papers(store, llm, Profile(topics=["FHIR", "NLP"]), [make_paper(1)])
    assert len(llm.calls) == 2
    assert cache_key(llm, PROFILE) != cache_key(llm, Profile(topics=["FHIR"]))


def test_invalid_response_is_retried_once(store):
    store.upsert_papers([make_paper(1)])
    llm = FakeLLM(responses=iter([{"relevance": 9}, VALID]))
    assert assess_papers(store, llm, PROFILE, [make_paper(1)]) == {"pmid:1": Assessment(**VALID)}


def test_paper_is_skipped_after_two_invalid_responses(store):
    store.upsert_papers([make_paper(1), make_paper(2)])
    llm = FakeLLM(responses=iter([{}, {}, VALID]))
    result = assess_papers(store, llm, PROFILE, [make_paper(1), make_paper(2)])
    assert list(result) == ["pmid:2"]


def test_unavailable_llm_propagates(store):
    store.upsert_papers([make_paper(1)])
    llm = FakeLLM(responses=iter([LLMUnavailable("down")]))
    with pytest.raises(LLMUnavailable):
        assess_papers(store, llm, PROFILE, [make_paper(1)])


def _ollama(handler) -> OllamaLLM:
    return OllamaLLM(
        "llama3.2",
        "http://ollama:11434/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_ollama_sends_schema_and_parses_content():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(VALID)}})

    result = _ollama(handler).complete("sys", "user", Assessment)
    assert result == Assessment(**VALID)
    assert sent["format"] == Assessment.model_json_schema()
    assert sent["options"]["temperature"] == 0
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_ollama_missing_model_is_unavailable():
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        _ollama(lambda _: httpx.Response(404)).complete("s", "u", Assessment)


def test_ollama_connection_error_is_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable):
        _ollama(handler).complete("s", "u", Assessment)


requires_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="optional dependency: uv sync --extra anthropic",
)


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic(); records the request and returns a canned response."""

    def __init__(self, response=None, error=None):
        self.kwargs = None
        self.response, self.error = response, error
        self.beta = self
        self.messages = self

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _parsed(stop_reason="end_turn", parsed=None):
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed)


@requires_anthropic
def test_anthropic_requests_structured_output_with_fallback():
    client = FakeAnthropicClient(_parsed(parsed=Assessment(**VALID)))
    llm = AnthropicLLM("claude-opus-5-5", "low", client=client)

    assert llm.complete("sys", "user", Assessment) == Assessment(**VALID)
    assert client.kwargs["output_format"] is Assessment
    assert client.kwargs["output_config"] == {"effort": "low"}
    assert client.kwargs["system"] == "sys"
    assert client.kwargs["fallbacks"] == "default"
    assert client.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


@requires_anthropic
def test_anthropic_refusal_is_a_bad_response_not_an_outage():
    llm = AnthropicLLM("m", "low", client=FakeAnthropicClient(_parsed("refusal")))
    with pytest.raises(ValueError, match="declined"):
        llm.complete("s", "u", Assessment)


@requires_anthropic
def test_anthropic_connection_error_is_unavailable():
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    llm = AnthropicLLM("m", "low", client=FakeAnthropicClient(error=error))
    with pytest.raises(LLMUnavailable):
        llm.complete("s", "u", Assessment)


def test_provider_defaults():
    assert LLMConfig().resolved_model == "mistral"
    assert LLMConfig(provider="anthropic").resolved_model == "claude-opus-5-5"
    assert LLMConfig(provider="anthropic", model="claude-haiku-4-5").resolved_model == (
        "claude-haiku-4-5"
    )
    assert isinstance(create_llm(LLMConfig()), OllamaLLM)
