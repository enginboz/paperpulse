"""
Command-line interface.

    paperpulse ingest            fetch new papers into the local store
    paperpulse select            rank stored papers, print the digest as JSON
    paperpulse run               ingest, then select
    paperpulse schema            print the JSON schema of the digest

The digest goes to stdout and logs go to stderr, so output can be piped.
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from paperpulse.config import Config, load_config
from paperpulse.embeddings import SentenceTransformerEmbedder
from paperpulse.llm import LLM, create_llm
from paperpulse.models import Digest
from paperpulse.pipeline import ingest, select
from paperpulse.sources import PubMedSource
from paperpulse.store import Store

logger = logging.getLogger("paperpulse")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "schema":
        print(json.dumps(Digest.model_json_schema(), indent=2))
        return 0

    if not args.config.exists():
        logger.error(
            "Config file %s not found. Copy paperpulse.example.toml to start.", args.config
        )
        return 2
    config = load_config(args.config)
    if getattr(args, "top", None):
        config.selection.top = args.top

    store = Store(args.db)
    try:
        if args.command in ("ingest", "run"):
            _ingest(store, config, args.days)
        if args.command in ("select", "run"):
            digest = select(
                store,
                config,
                SentenceTransformerEmbedder(config.embedding.model),
                llm=None if args.no_llm or not config.llm.enabled else _llm(config),
                record=not args.no_record,
            )
            _write(digest, args.output)
            if not digest.papers:
                logger.warning("No eligible papers in the window. Try `paperpulse ingest` first.")
    finally:
        store.close()
    return 0


def _ingest(store: Store, config: Config, days: int | None) -> None:
    sources = [PubMedSource(config.sources.pubmed)]
    ingest(store, sources, default_days=days or config.selection.window_days, today=date.today())


def _llm(config: Config) -> LLM:
    return create_llm(config.llm)


def _write(digest: Digest, output: Path | None) -> None:
    text = digest.model_dump_json(indent=2)
    if output:
        output.write_text(text + "\n")
        logger.info("Digest written to %s", output)
    else:
        print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperpulse", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("paperpulse.toml"),
        help="config file (default: ./paperpulse.toml)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.getenv("PAPERPULSE_DB", "paperpulse.db")),
        help="SQLite database (default: $PAPERPULSE_DB or ./paperpulse.db)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_args = argparse.ArgumentParser(add_help=False)
    ingest_args.add_argument(
        "--days", type=int, help="look-back on the first ingest (default: selection.window_days)"
    )

    select_args = argparse.ArgumentParser(add_help=False)
    select_args.add_argument("--top", type=int, help="number of papers (default: selection.top)")
    select_args.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    select_args.add_argument(
        "--no-llm", action="store_true", help="skip the LLM stage and rank by similarity only"
    )
    select_args.add_argument(
        "--no-record",
        action="store_true",
        help="don't save this run, so its papers can be selected again",
    )

    commands.add_parser("ingest", parents=[ingest_args], help="fetch new papers")
    commands.add_parser("select", parents=[select_args], help="rank papers, output JSON")
    commands.add_parser("run", parents=[ingest_args, select_args], help="ingest, then select")
    commands.add_parser("schema", help="print the digest JSON schema")
    return parser


if __name__ == "__main__":
    sys.exit(main())
