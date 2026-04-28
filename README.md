# PaperPulse 📄

PaperPulse is a daily medical literature digest tool. It automatically fetches recent papers from leading digital medicine journals, enriches them with community impact data, and uses an LLM to surface the 3 most relevant papers of the day based on your interest profile.

Designed for clinicians and researchers who want to stay current without drowning in journal alerts.

## Features

- **Automated fetching** — queries PubMed daily for papers from a curated journal whitelist
- **Impact enrichment** — adds Altmetric scores to surface papers generating real-world discussion
- **LLM-powered selection** — picks the top 3 papers most relevant to your interest profile
- **Lightweight web interface** — served as an HTMX-powered HTML fragment, easy to embed in any dashboard

## Stack

- **Python** — fetching, scoring, and scheduling pipeline
- **PostgreSQL** — stores fetched papers and daily selections
- **Flask + HTMX** — lightweight web server and dashboard widget
- **LLM** — scoring and relevance selection (provider configurable)

## Project Structure

```
paperpulse/
├── paperpulse/
│   ├── fetchers/
│   │   ├── pubmed.py        # PubMed E-utilities API client
│   │   └── altmetric.py     # Altmetric API client
│   ├── scoring/
│   │   └── llm.py           # LLM-based paper selection
│   ├── db.py                # Database models and queries
│   └── app.py               # Flask app and HTMX endpoints
├── tests/
├── .env.example
├── pyproject.toml
├── poetry.lock
└── run.py                   # Daily pipeline entry point
```

## Setup

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

3. Create the PostgreSQL database:
```bash
createdb paperpulse
python -c "from paperpulse.db import init_db; init_db()"
```

4. Run the pipeline manually:
```bash
poetry run python run.py
```

5. Start the Flask server:
```bash
poetry run flask --app paperpulse.app run
```

## Scheduling

Add to crontab to run daily at 6am:
```
0 6 * * * /path/to/venv/bin/python /path/to/paperpulse/run.py
```

## License

MIT — see [LICENSE](LICENSE)