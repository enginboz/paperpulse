# PaperPulse

PaperPulse is a daily medical literature digest tool. It automatically fetches recent papers from leading digital medicine journals and uses a two-stage AI pipeline to surface the 3 most relevant papers of the day based on a configurable interest profile.

Designed for clinicians and researchers who want to stay current without drowning in journal alerts.

## How it works

1. **Fetch** — queries PubMed daily for papers from a curated journal whitelist (focused journals fetched in full, broad journals filtered by topic keywords)
2. **Embed** — ranks all papers by semantic similarity to your interest profile using a biomedical embedding model (`pritamdeka/S-PubMedBert-MS-MARCO`)
3. **Score** — passes the top 15 candidates to a local LLM (Ollama) which selects the final 3 with a one-sentence relevance explanation
4. **Serve** — results are stored in PostgreSQL and served as an HTMX widget, refreshing every 6 hours

Papers shown in the last 7 days are automatically excluded to ensure fresh selections daily.

## Stack

- **Python** — pipeline, fetching, scoring, and scheduling
- **PostgreSQL** — stores fetched papers and daily digests
- **Flask + HTMX** — lightweight web server and dashboard widget
- **sentence-transformers** — biomedical embedding model for semantic pre-filtering
- **Ollama** — local LLM for final paper selection (provider-configurable)

## Project Structure

```
paperpulse/
├── paperpulse/
│   ├── fetchers/
│   │   └── pubmed.py        # PubMed E-utilities API client
│   ├── scoring/
│   │   ├── embeddings.py    # Biomedical embedding pre-filter
│   │   └── llm.py           # LLM-based final selection (Ollama)
│   ├── models.py            # Paper and Digest dataclasses
│   ├── config.py            # Interest profile configuration
│   ├── db.py                # Database ORM models and queries
│   └── app.py               # Flask app and HTMX endpoints
├── tests/
├── .env.example
├── pyproject.toml
├── poetry.lock
└── run.py                   # Daily pipeline entry point
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- [Ollama](https://ollama.com/) with a model pulled (e.g. `ollama pull llama3.2`)

### Installation

1. Clone the repo and install dependencies:
```bash
git clone https://github.com/enginboz/paperpulse.git
cd paperpulse
poetry install
```

2. Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

3. Create the PostgreSQL database and tables:
```bash
createdb paperpulse
poetry run python -c "from paperpulse.db import init_db; init_db()"
```

4. Run the pipeline manually:
```bash
poetry run python run.py
```

5. Start the Flask server:
```bash
poetry run python paperpulse/app.py
```

The widget is available at `http://localhost:5001`. Past digests can be viewed at `http://localhost:5001/digest/YYYY-MM-DD`.

## Configuration

Edit `paperpulse/config.py` to customize your interest profile. This text is used by both the embedding pre-filter and the LLM scoring step to determine paper relevance.

Edit the journal lists in `paperpulse/fetchers/pubmed.py` to adjust which journals are included.

## Scheduling

Add to crontab to run daily at 6am:
```
0 6 * * * /path/to/venv/bin/python /path/to/paperpulse/run.py
```

## License

MIT — see [LICENSE](LICENSE)