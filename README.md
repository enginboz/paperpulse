# PaperPulse

[![CI](https://github.com/enginboz/paperpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/enginboz/paperpulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PaperPulse finds the few new papers worth your time. It pulls recent literature from PubMed, ranks it against a research-interest profile you define, and prints the top picks as JSON, so the result can feed a dashboard, a newsletter, a notebook or any other tool.

It runs as a one-shot command on your laptop. There is no server to keep alive, and with the default local LLM no data leaves your machine apart from the PubMed queries.

## How it works

```
ingest   PubMed ──▶ normalise + deduplicate ──▶ SQLite
select   time window ──▶ hard filters ──┬─▶ dense retrieval ──┬─▶ rank fusion ──▶ LLM assessment ──▶ top N ──▶ JSON
                                        └─▶ BM25 keywords ───┘
```

**Ingestion and selection are separate steps.** `ingest` is incremental: each source resumes from its last watermark, and papers are keyed by DOI, so re-running is always safe. `select` works only on the local store. You can re-rank as often as you like while tuning your profile, and every embedding is computed once and cached per model.

**Hard filters run before any model.** Editorials, comments, errata and letters are removed by publication type, correction notices by title (journals often file them as ordinary articles), papers without an abstract are dropped, and papers picked in the last few days are excluded.

**Retrieval is hybrid.** Your profile is a list of focused topics, and every paper is ranked two ways:

- *Dense:* its highest cosine similarity to any one topic, using the biomedical embedding model [`S-PubMedBert-MS-MARCO`](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO). Taking the maximum instead of comparing against an averaged profile keeps niche interests from being drowned out by broad ones.
- *Keyword:* its best BM25 score across topics, via SQLite FTS5 with stemming. Embeddings blur exact terms, and acronyms such as FHIR, HL7 or OMOP mean little to them. BM25 weighs rare terms heavily, so a paper that literally names one ranks high.

The two rankings are merged with [reciprocal rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), which uses positions only, so similarity and BM25 scores never have to be put on one scale. On one week of real data, the keyword side pulled two papers into the LLM shortlist that embeddings had ranked 20th and 22nd (LLM-based pneumonia detection in radiology reports, and fact-checked medical question answering). The LLM rated both 5/5; one 5/5 paper that embeddings alone had kept (AI triage of vessel occlusion on CT) dropped out in exchange, so the shortlist gained one top-rated paper net. One week is a small sample, which is why an evaluation set is on the roadmap.

**An LLM judges each shortlisted paper on its own.** The best papers after fusion are assessed one at a time against a 1–5 rubric and your free-text preferences (for example, "original research over opinion pieces"). The model returns structured output: study type, a one-sentence rationale written before the score, and the relevance score. Judging papers individually, instead of asking for "the best 3 of these 20", keeps the task small enough for a local model, removes position bias, and makes every verdict cacheable and inspectable. Papers rated below a threshold are never shown, so a quiet week yields fewer picks rather than filler.

In a comparison on real data, a 3B model (`llama3.2`) rated everything 5/5 and mislabelled opinion pieces as systematic reviews, while `mistral` (7B) separated them correctly. Hence `mistral` is the local default, with Claude as a much faster cloud option.

**Every pick explains itself.** The output records each paper's dense and keyword ranks, the topics behind them and the LLM assessment, plus run metadata (time window, models, candidate counts), so any selection can be understood and reproduced. Embeddings and assessments are cached, keyed by model, prompt version and profile, so re-runs only pay for what changed.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and either [Ollama](https://ollama.com) (`ollama pull mistral`) or an Anthropic API key.

```bash
git clone https://github.com/enginboz/paperpulse.git
cd paperpulse
uv sync
cp paperpulse.example.toml paperpulse.toml    # then edit your topics
cp .env.example .env                          # then set PUBMED_EMAIL (NCBI asks for one)

uv run paperpulse run                          # ingest + select, JSON to stdout
```

The first run downloads the embedding model (~400 MB). On Linux, PyTorch is installed as a CPU-only build. To use Claude instead of a local model, run `uv sync --extra anthropic`, set `ANTHROPIC_API_KEY` in `.env` and switch `[llm] provider = "anthropic"` in the config.

## Usage

The `paperpulse` command lives in the project environment. Prefix commands with `uv run` (as in the Quickstart), activate the environment with `source .venv/bin/activate`, or install it globally with `uv tool install --editable .`; the examples below omit the prefix.

```bash
paperpulse ingest                 # fetch new papers into ./paperpulse.db
paperpulse select                 # rank and print the digest
paperpulse select --top 5 -o digest.json
paperpulse select --no-record     # preview without marking papers as shown
paperpulse select --no-llm        # skip the LLM stage, rank by similarity only
paperpulse history                # list saved digests
paperpulse history 2026-10-02     # show one again (a date or a run id)
paperpulse schema                 # JSON schema of the output
```

Global options: `-c/--config` (default `./paperpulse.toml`), `--db` (default `$PAPERPULSE_DB` or `./paperpulse.db`), `-v/--verbose`. Logs go to stderr, so stdout can be piped, e.g. `paperpulse select | jq '.papers[].title'`.

## Evaluating the ranking

Every ranking decision in PaperPulse (mistral over llama3.2, hybrid over dense, the thresholds) should be backed by numbers, not by one week of eyeballing. The evaluation workflow makes that measurable against your own judgements:

```bash
paperpulse feedback 1 up            # rate a digest pick (rank, DOI or pmid:<id>)
paperpulse feedback 3 down --note "opinion piece"
paperpulse label                    # rate a pool of papers interactively (y/n/skip/quit)
paperpulse eval                     # compare ranking variants against your labels
paperpulse eval --models mistral llama3.2   # compare LLMs on the same labels
paperpulse eval --no-llm            # retrieval variants only, in seconds
paperpulse eval --no-save           # quick check without writing a result file
paperpulse labels export labels.jsonl
paperpulse labels import labels.jsonl   # re-fetches missing papers from PubMed
```

Rating only what a digest showed would make the current ranking look perfect by construction, because nothing it missed ever gets judged. `paperpulse label` therefore asks about a **pool**: the union of the top papers of the dense, keyword and hybrid rankings, interleaved so that quitting early still covers the head of each (the pooling method used in TREC evaluations). The pool goes as deep as the LLM shortlist, so any paper an LLM could promote has a label instead of silently counting as not relevant.

`paperpulse eval` re-ranks every labelled time window with each variant (dense, hybrid, and hybrid plus each LLM given with `--models`), using exactly the code path of `select`. Labels belong to papers, not to variants, so one label set serves every comparison. It reports:

| Metric | Meaning |
|---|---|
| `P@k` | Share of relevant papers among the ones the variant would show |
| `R@n` | Share of all relevant papers that reach the top n, the LLM shortlist; the ceiling for the LLM stage |
| `judged` | Share of the top k that carry a label; unlabelled papers count as not relevant, so low coverage means label more |

Example of the report format (made-up numbers, not a measured result):

```
2 window(s), 9 relevant labelled papers

variant              P@3   R@12  judged
dense                33%    67%     100%
hybrid               50%    89%     100%
hybrid+mistral       83%    89%     100%
hybrid+llama3.2      50%    89%     100%
```

LLM variants assess any shortlisted paper missing from the cache, so the first run per model takes a few minutes per labelled week and later runs reuse the cache; `--cached-only` skips the LLM and reports how complete the cached rows are.

Every run is saved to [`evaluation/`](evaluation/) as `results/<timestamp>.json`, together with the label set it used as `labels.jsonl`. A result file records the setup, a checksum of the labels, the scores per week and the papers each variant picked, and is marked incomplete when part of the comparison is missing. Anyone can import the label set and re-run the evaluation.

## Output

Besides the selected `papers`, every digest carries its `shortlist`: all papers the LLM assessed, in retrieval order, with their assessment and whether they were selected. It shows why a paper did *not* make the cut, for example an opinion piece rated 2 or a solid paper rated 4 that lost a close race. The example below shows one selected paper and omits the shortlist for brevity.

```json
{
  "schema_version": "1",
  "run": {
    "id": "a6ef71d838964f67b6852d63063724e5",
    "created_at": "2026-09-29T09:43:09.457714Z",
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
        "total": 5.032,
        "fusion": 0.032,
        "dense": 0.9063,
        "dense_rank": 4,
        "matched_topic": "NLP for medical documentation and discharge summaries",
        "keyword_rank": 1,
        "keyword_topic": "NLP for medical documentation and discharge summaries",
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

`paperpulse.toml` holds your profile, the journals to watch and selection settings; see [`paperpulse.example.toml`](paperpulse.example.toml). Personal values and secrets stay out of it, in environment variables or a `.env` file in the working directory (see [`.env.example`](.env.example)); variables set in the shell take precedence:

| Variable | Purpose |
|---|---|
| `PUBMED_EMAIL` | Contact email sent with NCBI requests (recommended) |
| `NCBI_API_KEY` | Optional; raises the NCBI rate limit from 3 to 10 requests/s |
| `PAPERPULSE_DB` | Database path, if not `./paperpulse.db` |
| `OLLAMA_BASE_URL` | Ollama server, if not `http://localhost:11434` |
| `ANTHROPIC_API_KEY` | Required for `provider = "anthropic"` |

## Project structure

```
src/paperpulse/
├── sources/pubmed.py     # NCBI E-utilities client, incremental by Entrez date
├── store.py              # SQLite: papers, FTS5 index, embedding/assessment caches, runs
├── embeddings.py         # sentence-transformers wrapper with per-model cache
├── llm.py                # Ollama and Anthropic providers behind one interface
├── ranking/
│   ├── filters.py        # publication type, title, abstract, recent picks
│   ├── dense.py          # best-topic cosine similarity
│   ├── keyword.py        # best-topic BM25
│   ├── fusion.py         # reciprocal rank fusion
│   └── assess.py         # per-paper LLM rubric with structured output
├── pipeline.py           # ingest(), select() and the shared rank_papers()
├── labelling.py          # feedback, interactive pool labelling, JSONL export/import
├── evaluation.py         # pooling, P@k / R@n per variant, saved result records
└── cli.py
```

## From v1 to v2

The first version ([`v1.0`](https://github.com/enginboz/paperpulse/tree/v1.0)) was a working prototype: a daily cron job fetched papers into PostgreSQL, embeddings shortlisted 15 of them, a single prompt asked a local LLM to pick the best 3, and a Flask/HTMX widget displayed them. It worked, but it was hard to run anywhere else and hard to tell *why* a paper was chosen.

v2 is a rewrite around three ideas: separate ingestion from selection so ranking can be re-run and tuned offline; make every stage explain itself in the output; and replace "pick 3 of 15" with per-paper judgements a small local model can make reliably. It is a one-shot CLI with SQLite, so trying it takes three commands instead of a database server and a web app.

## Roadmap

- **Next:** learn from feedback, e.g. use papers rated 👍 as extra positive examples in the profile, validated with `paperpulse eval`
- Cross-encoder reranking between retrieval and the LLM stage
- Diversity (MMR) so the top picks don't all come from one topic
- Europe PMC source, including preprints
- Optional FastAPI server exposing the digest as a JSON API

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE)
