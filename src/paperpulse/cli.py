"""
Command-line interface.

    paperpulse ingest            fetch new papers into the local store
    paperpulse select            rank stored papers, print the digest as JSON
    paperpulse run               ingest, then select
    paperpulse schema            print the JSON schema of the digest
    paperpulse history           list saved digests (history <date|run id> shows one)
    paperpulse feedback 2 up     label a paper from the latest digest
    paperpulse label             label a pool of papers for evaluation
    paperpulse eval              compare ranking variants against the labels
    paperpulse labels export     write labels as JSONL (also: labels import FILE)

The digest goes to stdout and logs go to stderr, so output can be piped.
Environment variables can also be set in a .env file in the working directory.
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from paperpulse.config import Config, load_config
from paperpulse.embeddings import SentenceTransformerEmbedder
from paperpulse.evaluation import (
    eligible_papers,
    evaluate,
    evaluation_record,
    format_report,
    labelling_pool,
    save_evaluation,
)
from paperpulse.labelling import (
    export_labels,
    import_labels,
    label_interactively,
    record_feedback,
    resolve_paper,
)
from paperpulse.llm import LLM, create_llm
from paperpulse.models import Digest
from paperpulse.pipeline import ingest, select, window_start
from paperpulse.sources import PubMedSource
from paperpulse.store import Store

logger = logging.getLogger("paperpulse")


def main(argv: list[str] | None = None) -> int:
    # Before parsing: argument defaults such as --db read the environment.
    # Variables already set in the shell take precedence over the file.
    load_dotenv(find_dotenv(usecwd=True))
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
        if args.command == "history":
            return _history(store, args.run)
        if args.command in ("feedback", "label", "eval", "labels"):
            return _labels_and_eval(args, store, config)
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


def _history(store: Store, run: str | None) -> int:
    runs = store.all_runs()
    if run is None:
        if not runs:
            print("No saved digests yet. `paperpulse run` creates one.")
        for d in runs:
            r = d.run
            local = r.created_at.astimezone()
            print(f"{local:%Y-%m-%d %H:%M}  {r.id[:8]}  {r.llm_model or 'no LLM'}")
            for p in d.papers:
                print(f"    {p.rank}. {p.title[:90]}")
        return 0
    # A date (local time, as listed) picks that day's latest digest; else a run id prefix.
    matches = [
        d
        for d in runs
        if d.run.created_at.astimezone().date().isoformat() == run or d.run.id.startswith(run)
    ]
    if not matches:
        logger.error("No saved digest matches %r. `paperpulse history` lists them.", run)
        return 1
    print(matches[-1].model_dump_json(indent=2))
    return 0


def _labels_and_eval(args: argparse.Namespace, store: Store, config: Config) -> int:
    if args.command == "feedback":
        paper = resolve_paper(store, args.target)
        if paper is None:
            logger.error("No paper %r in the latest digest or the store.", args.target)
            return 1
        record_feedback(store, paper, relevant=args.verdict == "up", note=args.note)
        print(f"{'👍' if args.verdict == 'up' else '👎'} {paper.title}")
        return 0

    if args.command == "labels":
        if args.action == "export":
            lines = list(export_labels(store))
            text = "\n".join(lines) + ("\n" if lines else "")
            if args.file:
                args.file.write_text(text)
                logger.info("%d labels written to %s", len(lines), args.file)
            else:
                sys.stdout.write(text)
        else:
            source = PubMedSource(config.sources.pubmed)
            imported, skipped = import_labels(
                store, args.file.read_text().splitlines(), fetch_by_pmid=source.fetch_pmids
            )
            logger.info(
                "Imported %d labels, skipped %d without a matching paper", imported, skipped
            )
        return 0

    embedder = SentenceTransformerEmbedder(config.embedding.model)
    if args.command == "label":
        end = args.end or date.today()
        start = window_start(end, config.selection.window_days)
        papers = eligible_papers(store, config, start, end)
        depth = args.depth or config.llm.candidates
        pool = labelling_pool(store, config, embedder, papers, depth=depth)
        if not pool:
            logger.warning("Nothing to label between %s and %s.", start, end)
            return 0
        print(f"{len(pool)} unlabelled papers from the top {depth} of each ranking, ", end="")
        print(f"{start} to {end}.")
        written = label_interactively(store, pool)
        print(f"\n{written} labels saved.")
        return 0

    llms = []
    if not args.no_llm and config.llm.enabled:
        for model in args.models or [config.llm.resolved_model]:
            model_config = config.model_copy(deep=True)
            model_config.llm.model = model
            llms.append(_llm(model_config))
    k = args.k or config.selection.top
    results = evaluate(
        store,
        config,
        embedder,
        llms=llms,
        k=k,
        cached_only=args.cached_only,
        end=args.end or date.today(),
    )
    labels_jsonl = "".join(line + "\n" for line in export_labels(store))
    record = evaluation_record(results, config, llms, labels_jsonl, k, args.cached_only)
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(format_report(results, k=k, n=config.llm.candidates))
    if not args.no_save and results[0].windows:
        path = save_evaluation(record, labels_jsonl, args.save_dir)
        logger.info("Result saved to %s, label set to %s", path, args.save_dir / "labels.jsonl")
        if not record["complete"]:
            logger.warning("Saved as incomplete: %s", "; ".join(record["incomplete_reasons"]))
    return 0


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
    parser = argparse.ArgumentParser(
        prog="paperpulse",
        description="Surface the most relevant new papers for your research interests.",
    )
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

    history = commands.add_parser("history", help="list saved digests or show one")
    history.add_argument("run", nargs="?", help="a date (YYYY-MM-DD) or a run id prefix")

    feedback = commands.add_parser("feedback", help="label a paper as relevant or not")
    feedback.add_argument("target", help="rank in the latest digest, a DOI, or pmid:<id>")
    feedback.add_argument("verdict", choices=["up", "down"])
    feedback.add_argument("--note", help="optional free-text reason")

    label = commands.add_parser("label", help="interactively label a pool of papers")
    label.add_argument("--depth", type=int, help="top N of each ranking (default: llm.candidates)")
    label.add_argument(
        "--end", type=date.fromisoformat, help="last day of the window (default: today)"
    )

    evaluate_cmd = commands.add_parser("eval", help="compare ranking variants against labels")
    evaluate_cmd.add_argument("-k", type=int, help="precision cut-off (default: selection.top)")
    evaluate_cmd.add_argument(
        "--cached-only",
        action="store_true",
        help="use only cached LLM assessments: fast, but LLM rows may be incomplete",
    )
    evaluate_cmd.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="compare these LLMs, one hybrid+<model> row each (default: llm.model)",
    )
    evaluate_cmd.add_argument("--no-llm", action="store_true", help="skip the LLM variants")
    evaluate_cmd.add_argument(
        "--json", action="store_true", help="print the full result record as JSON"
    )
    evaluate_cmd.add_argument(
        "--end",
        type=date.fromisoformat,
        help="last day of the newest window (default: today; use your labelling weekday)",
    )
    evaluate_cmd.add_argument(
        "--no-save", action="store_true", help="don't write the result and label set to disk"
    )
    evaluate_cmd.add_argument(
        "--save-dir",
        type=Path,
        default=Path("evaluation"),
        help="where results/<timestamp>.json and labels.jsonl go (default: ./evaluation)",
    )

    labels = commands.add_parser("labels", help="export or import labels as JSONL")
    labels.add_argument("action", choices=["export", "import"])
    labels.add_argument("file", type=Path, nargs="?", help="JSONL file (export: default stdout)")
    return parser


if __name__ == "__main__":
    sys.exit(main())
