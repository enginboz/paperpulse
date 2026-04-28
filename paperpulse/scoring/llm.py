"""
LLM scoring module.

Uses a local Ollama model to select the 3 most relevant papers
from a list of candidates based on a predefined interest profile.

Two-stage approach to stay within the model's context window:
  Stage 1 — titles only: narrow 52 papers down to 15 candidates
  Stage 2 — full abstracts: pick the final top 3 from candidates

The module is designed to be provider-agnostic — the Ollama client
can be swapped for any OpenAI-compatible API (Anthropic, OpenAI, etc.)
by changing the base_url and model in the configuration.

Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Number of candidates to pass from stage 1 to stage 2
STAGE1_CANDIDATES = 15

# Interest profile — describes the ideal reader of PaperPulse.
# This is sent to the LLM as context for relevance scoring.
INTEREST_PROFILE = """
You are selecting papers for a physician with the following interests:

- AI-powered clinical decision support across all medical specialties
  (e.g. tumor board decisions, diagnosis assistance, outcome prediction)
- NLP applied to clinical notes and medical documentation
- Wearables and remote patient monitoring
- Health data infrastructure (EHRs, interoperability, FHIR)
- Digital therapeutics

The physician prioritizes papers showing real clinical applications and
results, but is also open to novel technical approaches that enable them.
Ethics and bias papers are less relevant unless they have strong clinical
implications.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """
    Send a prompt to the Ollama API and return the response text.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low temperature for consistent, focused output
        }
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600,  # 10 minutes — local models can be slow on CPU
    )
    response.raise_for_status()

    return response.json().get("response", "")


def _parse_indices(response_text: str, max_index: int) -> list[int]:
    """
    Parse a JSON array of indices from the LLM response.
    Returns a list of valid 1-based indices.
    """
    clean = response_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    indices = json.loads(clean)

    valid = []
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= max_index:
            valid.append(idx)
        else:
            print(f"Warning: LLM returned invalid index {idx} — skipping.")

    return valid


def _parse_selections(response_text: str, papers: list[dict]) -> list[dict]:
    """
    Parse a JSON array of {index, reason} objects from the LLM response.
    Returns the selected papers enriched with the reason field.
    """
    clean = response_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    selections = json.loads(clean)

    results = []
    for selection in selections[:3]:
        index = selection.get("index")
        reason = selection.get("reason", "")

        if not index or index < 1 or index > len(papers):
            print(f"Warning: LLM returned invalid index {index} — skipping.")
            continue

        paper = papers[index - 1].copy()
        paper["reason"] = reason
        results.append(paper)

    return results


def _stage1_shortlist(papers: list[dict]) -> list[dict]:
    """
    Stage 1: send titles only, ask the LLM to shortlist candidates.
    Returns a subset of papers for stage 2.
    """
    lines = []
    for i, paper in enumerate(papers, 1):
        lines.append(f"[{i}] {paper['title']} ({paper['journal']})")
    papers_text = "\n".join(lines)

    prompt = f"""You are a medical literature assistant helping a physician stay current with the latest research.

{INTEREST_PROFILE}

Below are {len(papers)} recently published paper titles. Select the {STAGE1_CANDIDATES} most likely to be relevant to this physician.

PAPERS:
{papers_text}

Respond ONLY with a valid JSON array of {STAGE1_CANDIDATES} integers representing the paper numbers. No preamble, no explanation, no markdown.
Example: [1, 3, 7, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]"""

    print(f"Stage 1: shortlisting {STAGE1_CANDIDATES} candidates from {len(papers)} papers (titles only)...")
    response_text = _call_ollama(prompt)
    indices = _parse_indices(response_text, len(papers))

    if not indices:
        print("Warning: Stage 1 parsing failed, using first candidates as fallback.")
        return papers[:STAGE1_CANDIDATES]

    candidates = [papers[i - 1] for i in indices[:STAGE1_CANDIDATES]]
    print(f"Stage 1 complete — {len(candidates)} candidates selected.")
    return candidates


def _stage2_select(candidates: list[dict]) -> list[dict]:
    """
    Stage 2: send full abstracts of candidates, ask LLM to pick final top 3.
    Returns 3 papers enriched with a relevance reason.
    """
    lines = []
    for i, paper in enumerate(candidates, 1):
        lines.append(f"[{i}] {paper['journal'].upper()}")
        lines.append(f"Title: {paper['title']}")
        lines.append(f"Abstract: {paper['abstract'][:600]}")
        lines.append("")
    papers_text = "\n".join(lines)

    prompt = f"""You are a medical literature assistant helping a physician stay current with the latest research.

{INTEREST_PROFILE}

Below are {len(candidates)} candidate papers. Select the 3 most relevant for this physician.

PAPERS:
{papers_text}

Select the 3 most relevant papers. For each, provide a one-sentence explanation of why it is relevant.

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown. Example format:
[
  {{
    "index": 1,
    "reason": "One sentence explaining clinical relevance."
  }},
  {{
    "index": 7,
    "reason": "One sentence explaining clinical relevance."
  }},
  {{
    "index": 12,
    "reason": "One sentence explaining clinical relevance."
  }}
]"""

    print(f"Stage 2: selecting top 3 from {len(candidates)} candidates (full abstracts)...")
    response_text = _call_ollama(prompt)
    selected = _parse_selections(response_text, candidates)
    print(f"Stage 2 complete — {len(selected)} papers selected.")
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_top_papers(papers: list[dict]) -> list[dict]:
    """
    Use a local LLM to select the 3 most relevant papers from a list.

    Uses a two-stage approach to stay within the model's context window:
    - Stage 1: titles only → shortlist of 15 candidates
    - Stage 2: full abstracts → final top 3 with relevance reasons

    Args:
        papers: List of paper dicts from the PubMed fetcher.

    Returns:
        List of 3 paper dicts, each enriched with a 'reason' field
        explaining why the LLM selected it.
    """
    if not papers:
        return []

    print(f"\nStarting LLM scoring with {OLLAMA_MODEL}...")

    try:
        candidates = _stage1_shortlist(papers)

        if not candidates:
            return []

        selected = _stage2_select(candidates)

        print(f"\nLLM scoring complete — {len(selected)} papers selected.")
        return selected

    except json.JSONDecodeError as e:
        print(f"Error: LLM response was not valid JSON — {e}")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not reach Ollama at {OLLAMA_BASE_URL} — {e}")
        print("Make sure Ollama is running: ollama serve")
        return []