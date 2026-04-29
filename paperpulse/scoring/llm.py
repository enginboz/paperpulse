"""
LLM scoring module.

Uses a local Ollama model to select the 3 most relevant papers
from a list of candidates based on a predefined interest profile.

Stage 2 of the selection pipeline — receives 15 pre-filtered candidates
from the embedding module and picks the final top 3 with relevance reasons.

The module is designed to be provider-agnostic — the Ollama client
can be swapped for any OpenAI-compatible API (Anthropic, OpenAI, etc.)
by changing the base_url and model in the configuration.

Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import os

import requests
from dotenv import load_dotenv

from paperpulse.config import INTEREST_PROFILE
from paperpulse.models import Paper

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Number of candidates to pass from stage 1 to stage 2
STAGE1_CANDIDATES = 15

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


def _parse_selections(response_text: str, papers: list[Paper]) -> list[Paper]:
    """
    Parse a JSON array of {index, reason} objects from the LLM response.
    Returns the selected papers with the reason field set.
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

        paper = papers[index - 1]
        paper.reason = reason
        results.append(paper)

    return results


def _stage2_select(candidates: list[Paper]) -> list[Paper]:
    """
    Stage 2: send full abstracts of candidates, ask LLM to pick final top 3.
    Returns 3 papers with the reason field set.
    """
    lines = []
    for i, paper in enumerate(candidates, 1):
        lines.append(f"[{i}] {paper.journal.upper()}")
        lines.append(f"Title: {paper.title}")
        lines.append(f"Abstract: {paper.abstract[:600]}")
        lines.append("")
    papers_text = "\n".join(lines)

    prompt = f"""You are a medical literature assistant helping a physician stay current with the latest research.

The physician has the following interests:
{INTEREST_PROFILE}

The physician prioritizes papers showing real clinical applications and results,
but is also open to novel technical approaches that enable them.
Ethics and bias papers are less relevant unless they have strong clinical implications.

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

def select_top_papers(papers: list[Paper]) -> list[Paper]:
    """
    Use embeddings + LLM to select the 3 most relevant papers from a list.

    Two-stage approach:
    - Stage 1: embedding similarity → shortlist of 15 candidates (fast, free)
    - Stage 2: LLM with full abstracts → final top 3 with relevance reasons

    Args:
        papers: List of Paper objects from the PubMed fetcher.

    Returns:
        List of 3 Paper objects with the reason field set.
    """
    if not papers:
        return []

    print(f"\nStarting paper selection...")

    try:
        from paperpulse.scoring.embeddings import shortlist_by_embedding
        candidates = shortlist_by_embedding(papers, n=STAGE1_CANDIDATES)

        if not candidates:
            return []

        print(f"\nRunning LLM stage 2 with {OLLAMA_MODEL}...")
        selected = _stage2_select(candidates)

        print(f"\nSelection complete — {len(selected)} papers selected.")
        return selected

    except json.JSONDecodeError as e:
        print(f"Error: LLM response was not valid JSON — {e}")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not reach Ollama at {OLLAMA_BASE_URL} — {e}")
        print("Make sure Ollama is running: ollama serve")
        return []