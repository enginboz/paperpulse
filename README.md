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

The two rankings are merged with [reciprocal rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), which uses positions only, so similarity and BM25 scores never have to be put on one scale. On one week of real data, the keyword side pulled two papers into the LLM shortlist that embeddings had ranked 20th and 22nd (LLM-based pneumonia detection in radiology reports, and fact-checked medical question answering). The LLM rated both 5/5; one 5/5 paper that embeddings alone had kept (AI triage of vessel occlusion on CT) dropped out in exchange, so the shortlist gained one top-rated paper net. Over [four weeks of evaluation](#results-so-far), hybrid retrieval showed more relevant papers than dense (9 against 7 of 12 picks), though it did not reach more of them.

**An LLM judges each shortlisted paper on its own.** The best papers after fusion are assessed one at a time against a 1–5 rubric and your free-text preferences (for example, "original research over opinion pieces"). The model returns structured output: study type, a one-sentence rationale written before the score, and the relevance score. Judging papers individually, instead of asking for "the best 3 of these 20", keeps the task small enough for a local model, removes position bias, and makes every verdict cacheable and inspectable. Papers rated below a threshold are never shown, so a quiet week yields fewer picks rather than filler. If the LLM cannot be reached, the digest is ranked by retrieval alone, marked with `"llm_model": null`, and not saved, so its papers stay available for a run with the LLM.

In an early spot check on five papers, a 3B model (`llama3.2`) rated everything 5/5 and mislabelled opinion pieces as systematic reviews, while `mistral` (7B) separated them correctly. Over [four weeks of evaluation](#results-so-far), however, both models picked the same number of relevant papers, and both pass most papers through. In a [comparison of small models](#small-models-on-a-4-gb-gpu), `qwen3:4b-instruct` was the only clearly selective one, and it runs on a 4 GB GPU in seconds per paper. `mistral` stays the default for now, with Claude as a much faster cloud option.

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
paperpulse read                   # latest digest as a reading list: why, and a link
paperpulse read --top-only        # without the runners-up the LLM also rated 4 or 5
paperpulse feedback 1 up          # after reading: was pick 1 worth it? (rank, DOI or pmid:<id>)
paperpulse feedback 2 down --note "thin methods"
paperpulse history                # list saved digests
paperpulse history 2026-10-02     # show one again (a date or a run id)
paperpulse schema                 # JSON schema of the output
```

Global options: `-c/--config` (default `./paperpulse.toml`), `--db` (default `$PAPERPULSE_DB` or `./paperpulse.db`), `-v/--verbose`. Logs go to stderr, so stdout can be piped, e.g. `paperpulse select | jq '.papers[].title'`.

## Evaluating the ranking

Every ranking decision in PaperPulse (mistral over llama3.2, hybrid over dense, the thresholds) should be backed by numbers, not by one week of eyeballing. The evaluation workflow makes that measurable against your own judgements:

```bash
paperpulse label                    # rate a pool of papers from title and abstract (y/n/skip/quit)
paperpulse eval                     # compare ranking variants against your labels
paperpulse eval --models mistral llama3.2   # compare LLMs on the same labels
paperpulse eval --no-llm            # retrieval variants only, in seconds
paperpulse eval --no-save           # quick check without writing a result file
paperpulse labels export labels.jsonl
paperpulse labels import labels.jsonl   # re-fetches missing papers from PubMed
```

Labels are given from title and abstract only, the same information every ranking stage sees; judging with the full text in hand would hold the system to something it could not know. For the same reason, `paperpulse feedback` (your verdict after reading a paper) is stored separately and never enters the evaluation set.

Rating only what a digest showed would also make the current ranking look perfect by construction, because nothing it missed ever gets judged. `paperpulse label` therefore asks about a **pool**: the union of the top papers of the dense, keyword and hybrid rankings, interleaved so that quitting early still covers the head of each (the pooling method used in TREC evaluations). The pool goes as deep as the LLM shortlist, so any paper an LLM could promote has a label instead of silently counting as not relevant.

`paperpulse eval` re-ranks every labelled time window with each variant (dense, hybrid, and hybrid plus each LLM given with `--models`), using exactly the code path of `select`. Labels belong to papers, not to variants, so one label set serves every comparison. It reports:

| Metric | Meaning |
|---|---|
| `P@k` | Share of relevant papers among the ones the variant would show |
| `R@n` | Share of all relevant papers that reach the top n, the LLM shortlist; the ceiling for the LLM stage |
| `judged` | Share of the top k that carry a label; unlabelled papers count as not relevant, so low coverage means label more |

### Results so far

Four weeks of PubMed papers (19 Sep to 16 Oct 2026), 82 labelled papers of which 42 relevant, every top pick labelled. Source: [result file](evaluation/results/2026-10-16T200731.json), [label set](evaluation/labels.jsonl).

```
variant              P@3   R@12  judged
dense                58%    63%    100%
hybrid               75%    65%    100%
hybrid+mistral       83%    65%    100%
hybrid+llama3.2      83%    65%    100%
```

- **Hybrid retrieval shows more relevant papers than dense:** 9 of 12 picks against 7 of 12. It does not *find* more: 27 of the 42 relevant papers reached its shortlist, against 26 for dense. The gain comes from ordering.
- **The LLM stage adds a little, and the model made no difference here.** Both `mistral` and `llama3.2` picked 10 of 12 relevant papers, each doing better in a different week. The two-week result in which `mistral` led 6 to 4 did not hold up.
- **Retrieval is the bottleneck.** About a third of the relevant papers never reach the LLM shortlist, so no model can pick them.
- **The sample is small.** Each precision figure rests on 12 picks and the differences are one to three papers. These are measurements, not significant results; labelling continues weekly.

#### Small models on a 4 GB GPU

The same labels, with the LLM stage served by Ollama on an AMD Radeon RX 6500 XT (4 GB). Only models that fit entirely on the GPU were compared; `gemma3:4b` ran 77% on the GPU and was left out. On the GPU an assessment takes 2 to 5 seconds, against about 45 for `mistral` on a laptop CPU. Source: [result file](evaluation/results/2026-10-16T203316.json).

P@3 rests on 12 picks per model, so the table adds a less noisy view: how each model's filter (relevance 3 or higher) agrees with the labels on all 47 labelled papers of the shortlists, 27 of them relevant. A filter that passes everything would reach 57% precision.

| LLM | P@3 | passes filter | precision | recall | agreement |
|---|---|---|---|---|---|
| `mistral` (7B, laptop CPU) | 83% | 37 of 47 | 65% | 89% | 66% |
| `llama3.2:3b` | 83% | 35 of 47 | 66% | 85% | 66% |
| `qwen3:4b-instruct` | 92% | 24 of 47 | 79% | 70% | 72% |
| `phi4-mini` (3.8B) | 83% | 46 of 47 | 57% | 96% | 55% |

- **`phi4-mini` approves almost everything**, so it does not filter at all.
- **`mistral` and `llama3.2` barely beat passing everything** (65% against 57% precision); they mostly re-sort.
- **`qwen3:4b-instruct` is the only clearly selective model**: the highest precision and agreement, and 11 of 12 relevant picks. The cost is recall: it rejects 8 of the 27 relevant papers, so quiet weeks may yield fewer than three picks.
- **The same model gives slightly different answers on different hardware.** `llama3.2` on the laptop CPU and on the server GPU agreed on 42 of 47 ratings, even at temperature 0.

With 47 papers these differences are indicative rather than significant.

The earlier [two-week result](evaluation/results/2026-10-02T135101.json) is kept for reference, but its numbers are not directly comparable, for two reasons that are worth knowing:

1. **Late arrivals.** PubMed keeps adding records dated a given day after that day; a labelling session on Friday misses papers dated Friday that arrive later. Unlabelled, they count as not relevant. `paperpulse label` now offers the previous week's late arrivals first.
2. **Keyword weights drift.** BM25 weighs a term by how rare it is across *all* stored papers, so as the corpus grows, rankings in past weeks shift slightly. Variants within one evaluation are always compared on the same state; numbers from different evaluations are not.

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
| `OLLAMA_BASE_URL` | Ollama server, if not `http://localhost:11434`; a server that sleeps can be woken with `[llm] wake_command`, e.g. a Wake-on-LAN call |
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
├── labelling.py          # pool labelling, feedback, JSONL export/import
├── evaluation.py         # pooling, P@k / R@n per variant, saved result records
└── cli.py
```

## From v1 to v2

The first version ([`v1.0`](https://github.com/enginboz/paperpulse/tree/v1.0)) was a working prototype: a daily cron job fetched papers into PostgreSQL, embeddings shortlisted 15 of them, a single prompt asked a local LLM to pick the best 3, and a Flask/HTMX widget displayed them. It worked, but it was hard to run anywhere else and hard to tell *why* a paper was chosen.

v2 is a rewrite around three ideas: separate ingestion from selection so ranking can be re-run and tuned offline; make every stage explain itself in the output; and replace "pick 3 of 15" with per-paper judgements a small local model can make reliably. It is a one-shot CLI with SQLite, so trying it takes three commands instead of a database server and a web app.

## Roadmap

- **Next:** widen the LLM shortlist (12 to 20 papers) and measure whether recall rises without hurting precision; [retrieval is currently the bottleneck](#results-so-far)
- Learn from feedback, e.g. use papers rated 👍 as extra positive examples in the profile, validated with `paperpulse eval`
- Compare abstract-level relevance (labels and LLM) with post-reading feedback: how well does the abstract predict a paper worth reading?
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
