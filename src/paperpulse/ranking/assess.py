"""
LLM assessment: each shortlisted paper is judged on its own against a rubric.

Judging papers one at a time, rather than asking for "the best 3 of these 20",
keeps the task small enough for a local model, makes every verdict cacheable
and inspectable, and removes position bias from the prompt ordering.
"""

import hashlib
import json
import logging

from pydantic import ValidationError

from paperpulse.config import Profile
from paperpulse.llm import LLM, LLMUnavailable
from paperpulse.models import Assessment, Paper
from paperpulse.store import Store

logger = logging.getLogger(__name__)

# Bump whenever the prompt or rubric changes, so cached assessments are not reused.
PROMPT_VERSION = "1"
MAX_ABSTRACT_CHARS = 4000

SYSTEM_PROMPT = """\
You screen newly published biomedical papers for one reader and judge how \
worthwhile each paper is for them. Judge from the title and abstract only.

The reader's topics of interest:
{topics}
{preferences}
Rate relevance on this scale:
5 - Squarely within a topic and contributes new methods, systems, data or results \
the reader would want to read in full.
4 - Within a topic with a substantive contribution, but narrower or less novel.
3 - Related to a topic, but the connection is partial or the contribution is modest.
2 - Touches the topics only superficially, for example when AI or digital health is \
mentioned but is not what the paper is about.
1 - Not relevant to the reader.

Do not reward buzzwords: a paper is relevant because of what it does, not the terms it uses. \
Write the rationale as one plain sentence addressed to the reader, before deciding the score.\
"""


def cache_key(llm: LLM, profile: Profile) -> str:
    """Assessments depend on the model, the prompt and the profile; any change invalidates them."""
    profile_json = json.dumps(profile.model_dump(), sort_keys=True)
    digest = hashlib.sha256(profile_json.encode()).hexdigest()[:16]
    return f"{llm.name}|prompt-v{PROMPT_VERSION}|{digest}"


def build_system_prompt(profile: Profile) -> str:
    topics = "\n".join(f"- {t}" for t in profile.topics)
    preferences = (
        f"\nThe reader's preferences:\n{profile.preferences.strip()}\n"
        if (profile.preferences.strip())
        else ""
    )
    return SYSTEM_PROMPT.format(topics=topics, preferences=preferences)


def build_user_prompt(paper: Paper) -> str:
    types = ", ".join(paper.publication_types) or "unknown"
    return (
        f"Title: {paper.title}\n"
        f"Journal: {paper.journal}\n"
        f"Publication types: {types}\n\n"
        f"Abstract:\n{paper.abstract[:MAX_ABSTRACT_CHARS]}"
    )


def assess_papers(
    store: Store, llm: LLM, profile: Profile, papers: list[Paper], cached_only: bool = False
) -> dict[str, Assessment]:
    """
    Assess papers, reusing cached verdicts. Papers whose response stays invalid
    after one retry are left out, as are uncached papers when `cached_only`.
    Raises LLMUnavailable if the server is down.
    """
    key = cache_key(llm, profile)
    results = store.get_assessments([p.id for p in papers], key)
    missing = [p for p in papers if p.id not in results]
    if not missing or cached_only:
        return results

    logger.info("Assessing %d papers with %s (%d cached)", len(missing), llm.name, len(results))
    system = build_system_prompt(profile)
    for i, paper in enumerate(missing, 1):
        assessment = _assess_one(llm, system, build_user_prompt(paper))
        if assessment is None:
            logger.warning("No valid assessment for %s, skipping it", paper.id)
            continue
        logger.debug("[%d/%d] %s: %d", i, len(missing), paper.id, assessment.relevance)
        store.put_assessment(paper.id, key, assessment)
        results[paper.id] = assessment
    return results


def _assess_one(llm: LLM, system: str, user: str) -> Assessment | None:
    for _ in range(2):
        try:
            return llm.complete(system, user, Assessment)
        except LLMUnavailable:
            raise
        except (ValidationError, ValueError) as e:
            logger.debug("Invalid assessment response: %s", e)
    return None
