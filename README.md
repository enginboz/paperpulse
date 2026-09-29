# PaperPulse

PaperPulse finds the few new papers worth your time. It pulls recent literature from PubMed, ranks it against a research-interest profile you define, and prints the top picks as JSON, so the result can feed a dashboard, a newsletter, a notebook or any other tool.

It runs as a one-shot command on your laptop. There is no server to keep alive, and with the default local LLM no data leaves your machine apart from the PubMed queries.

> **Status:** v2 is a rewrite in progress. The previous prototype is kept at tag [`v1.0`](https://github.com/enginboz/paperpulse/tree/v1.0).

## How it works

```
ingest   PubMed ──▶ normalise + deduplicate ──▶ SQLite
select   time window ──▶ hard filters ──▶ dense retrieval ──▶ LLM assessment ──▶ top N ──▶ JSON
```

**Ingestion and selection are separate steps.** `ingest` is incremental: each source resumes from its last watermark, and papers are keyed by DOI, so re-running is always safe. `select` works only on the local store. You can re-rank as often as you like while tuning your profile, and every embedding is computed once and cached per model.

**Hard filters run before any model.** Editorials, comments, errata, letters and papers without an abstract are removed by publication type, and papers picked in the last few days are excluded.

**Each paper is scored by its best-matching topic.** Your profile is a list of focused topics. A paper's score is its highest cosine similarity to any one topic, using the biomedical embedding model [`S-PubMedBert-MS-MARCO`](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO). Taking the maximum instead of comparing against an averaged profile keeps niche interests from being drowned out by broad ones.

**An LLM judges each shortlisted paper on its own.** The best matches by similarity are assessed one at a time against a 1–5 rubric and your free-text preferences (for example, "original research over opinion pieces"). The model returns structured output: study type, a one-sentence rationale written before the score, and the relevance score. Judging papers individually, instead of asking for "the best 3 of these 20", keeps the task small enough for a local model, removes position bias, and makes every verdict cacheable and inspectable. Papers rated below a threshold are never shown, so a quiet week yields fewer picks rather than filler.

In a comparison on real data, a 3B model (`llama3.2`) rated everything 5/5 and mislabelled opinion pieces as systematic reviews, while `mistral` (7B) separated them correctly. Hence `mistral` is the local default, with Claude as a much faster cloud option.

**Every pick explains itself.** The output records the matched topic, similarity and LLM assessment for each paper, plus run metadata (time window, models, candidate counts), so any selection can be understood and reproduced. Embeddings and assessments are cached, keyed by model, prompt version and profile, so re-runs only pay for what changed.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and either [Ollama](https://ollama.com) (`ollama pull mistral`) or an Anthropic API key.

```bash
git clone https://github.com/enginboz/paperpulse.git
cd paperpulse
uv sync
cp paperpulse.example.toml paperpulse.toml    # then edit your topics
export PUBMED_EMAIL=you@example.org           # NCBI asks clients to identify themselves

uv run paperpulse run                          # ingest + select, JSON to stdout
```

The first run downloads the embedding model (~400 MB). On Linux, PyTorch is installed as a CPU-only build. To use Claude instead of a local model, run `uv sync --extra anthropic`, set `ANTHROPIC_API_KEY` and switch `[llm] provider = "anthropic"` in the config.

## Usage

```bash
paperpulse ingest                 # fetch new papers into ./paperpulse.db
paperpulse select                 # rank and print the digest
paperpulse select --top 5 -o digest.json
paperpulse select --no-record     # preview without marking papers as shown
paperpulse select --no-llm        # skip the LLM stage, rank by similarity only
paperpulse schema                 # JSON schema of the output
```

Global options: `-c/--config` (default `./paperpulse.toml`), `--db` (default `$PAPERPULSE_DB` or `./paperpulse.db`), `-v/--verbose`. Logs go to stderr, so stdout can be piped, e.g. `paperpulse select | jq '.papers[].title'`.

## Output

```json
{
  "schema_version": "1",
  "run": {
    "id": "40c92c44143d4065a124ad64d7236d58",
    "created_at": "2026-09-29T09:14:32.096950Z",
    "window_start": "2026-09-22",
    "window_end": "2026-09-29",
    "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "llm_model": "mistral",
    "candidates": 54,
    "after_filters": 52
  },
  "papers": [
    {
      "rank": 1,
      "id": "10.2196/90870",
      "title": "Automated Brief Hospital Course Summarization in Cardiac Surgery Using a Lightweight Large Language Model-Based Framework: Development and Evaluation Study on the Medical Information Mart for Intensive Care-IV.",
      "journal": "Journal of medical Internet research",
      "published": "2026-09-24",
      "authors": ["Xiaoyuan Gao", "Yang Wang", "Zixing Wang", "Jing Yuan", "Shengkang Huang", "Xu-Yao Zhang", "Zhaohong Sun", "Yun Xing", "Yiyang Liu", "Xintong Wu", "Zhan Hu", "Wei Zhao"],
      "doi": "10.2196/90870",
      "pmid": "42785733",
      "url": "https://pubmed.ncbi.nlm.nih.gov/42785733/",
      "score": {
        "total": 5.9063,
        "dense": 0.9063,
        "matched_topic": "NLP for medical documentation and discharge summaries",
        "assessment": {
          "study_type": "original_research",
          "rationale": "The paper presents a new lightweight, locally deployable framework for automating brief hospital course (BHC) summarization in cardiac surgery using a large language model (LLM)-based approach, which aligns with the reader's interest in AI-assisted diagnosis, NLP applied to clinical notes, and electronic health record systems.",
          "relevance": 5
        }
      }
    }
  ]
}
```

## Configuration

`paperpulse.toml` holds your profile, the journals to watch and selection settings; see [`paperpulse.example.toml`](paperpulse.example.toml). Secrets stay in the environment:

| Variable | Purpose |
|---|---|
| `PUBMED_EMAIL` | Contact email sent with NCBI requests (recommended) |
| `NCBI_API_KEY` | Optional; raises the NCBI rate limit from 3 to 10 requests/s |
| `PAPERPULSE_DB` | Database path, if not `./paperpulse.db` |
| `OLLAMA_BASE_URL` | Ollama server, if not `http://localhost:11434` |
| `ANTHROPIC_API_KEY` | Required for `provider = "anthropic"` |

## Roadmap

- Hybrid retrieval: BM25 keyword search fused with dense scores, for acronyms like FHIR or OMOP
- Cross-encoder reranking between dense retrieval and the LLM stage
- Diversity (MMR) so the top picks don't all come from one topic
- Europe PMC source, including preprints
- Feedback (👍/👎) and an evaluation set to measure precision@k
- Optional FastAPI server exposing the digest as a JSON API

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE)
