# PaperPulse

PaperPulse finds the few new papers worth your time. It pulls recent literature from PubMed, ranks it against a research-interest profile you define, and prints the top picks as JSON, so the result can feed a dashboard, a newsletter, a notebook or any other tool.

It runs locally as a one-shot command. There is no server to keep alive and no data leaves your machine apart from the PubMed queries.

> **Status:** v2 is a rewrite in progress. The previous prototype is kept at tag [`v1.0`](https://github.com/enginboz/paperpulse/tree/v1.0).

## How it works

```
ingest   PubMed ──▶ normalise + deduplicate ──▶ SQLite
select   time window ──▶ hard filters ──▶ dense retrieval ──▶ top N ──▶ JSON
```

**Ingestion and selection are separate steps.** `ingest` is incremental: each source resumes from its last watermark, and papers are keyed by DOI, so re-running is always safe. `select` works only on the local store. You can re-rank as often as you like while tuning your profile, and every embedding is computed once and cached per model.

**Hard filters run before any model.** Editorials, comments, errata, letters and papers without an abstract are removed by publication type, and papers picked in the last few days are excluded.

**Each paper is scored by its best-matching topic.** Your profile is a list of focused topics. A paper's score is its highest cosine similarity to any one topic, using the biomedical embedding model [`S-PubMedBert-MS-MARCO`](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO). Taking the maximum instead of comparing against an averaged profile keeps niche interests from being drowned out by broad ones.

**Every pick explains itself.** The output records the matched topic and score for each paper, plus run metadata (time window, model, candidate counts), so any selection can be understood and reproduced.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/enginboz/paperpulse.git
cd paperpulse
uv sync
cp paperpulse.example.toml paperpulse.toml    # then edit your topics
export PUBMED_EMAIL=you@example.org           # NCBI asks clients to identify themselves

uv run paperpulse run                          # ingest + select, JSON to stdout
```

The first run downloads the embedding model (~400 MB). On Linux, PyTorch is installed as a CPU-only build.

## Usage

```bash
paperpulse ingest                 # fetch new papers into ./paperpulse.db
paperpulse select                 # rank and print the digest
paperpulse select --top 5 -o digest.json
paperpulse select --no-record     # preview without marking papers as shown
paperpulse schema                 # JSON schema of the output
```

Global options: `-c/--config` (default `./paperpulse.toml`), `--db` (default `$PAPERPULSE_DB` or `./paperpulse.db`), `-v/--verbose`. Logs go to stderr, so stdout can be piped, e.g. `paperpulse select | jq '.papers[].title'`.

## Output

```json
{
  "schema_version": "1",
  "run": {
    "id": "3cffdad68d264b109346b5cb3ef3dcd3",
    "created_at": "2026-09-29T08:51:12.707139Z",
    "window_start": "2026-09-22",
    "window_end": "2026-09-29",
    "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "candidates": 54,
    "after_filters": 52
  },
  "papers": [
    {
      "rank": 1,
      "id": "10.2196/87806",
      "title": "AI in Neurological Health Care: Qualitative Study of Patient and Public Perceptions.",
      "journal": "Journal of medical Internet research",
      "published": "2026-09-24",
      "authors": ["Tina Bedenik", "Orna Fennelly", "Kathleen Bennett"],
      "doi": "10.2196/87806",
      "pmid": "42785740",
      "url": "https://pubmed.ncbi.nlm.nih.gov/42785740/",
      "score": {
        "total": 0.9228,
        "dense": 0.9228,
        "matched_topic": "Novel technical approaches enabling clinical AI"
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

## Roadmap

- Hybrid retrieval: BM25 keyword search fused with dense scores, for acronyms like FHIR or OMOP
- Cross-encoder reranking and per-paper LLM assessment with structured output (Ollama by default)
- Diversity (MMR) so the top picks don't all come from one topic
- Europe PMC source, including preprints
- Feedback (👍/👎) and an evaluation set to measure precision@k
- Optional FastAPI server exposing the digest as a JSON API

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE)
