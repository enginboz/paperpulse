from datetime import UTC, date, datetime

from paperpulse.evaluation import evaluate, evaluation_windows, format_report, labelling_pool
from paperpulse.models import Label
from tests.conftest import FakeLLM, make_paper

NOW = datetime(2026, 9, 29, tzinfo=UTC)


def label(store, pid: str, relevant: bool) -> None:
    store.set_label(Label(paper_id=pid, relevant=relevant, source="pool", labelled_at=NOW))


def seed_hl7_week(store, config):
    """Dense ranking prefers paper 1; only paper 2 mentions HL7 (see test_pipeline)."""
    config.profile.topics = ["HL7"]
    config.selection.top = 1
    store.upsert_papers(
        [
            make_paper(1, title="HL", abstract="hl hl hl"),
            make_paper(2, title="Interface engines", abstract="hl7 feeds in hospitals"),
            make_paper(3, title="Knee surgery", abstract="orthopedic outcomes"),
        ]
    )


def test_windows_tile_the_labelled_range_without_overlap():
    papers = [make_paper(n, added=d) for n, d in [(1, date(2026, 9, 1)), (2, date(2026, 9, 20))]]
    windows = evaluation_windows([(None, p) for p in papers], days=7)
    assert windows == [
        (date(2026, 8, 31), date(2026, 9, 6)),
        (date(2026, 9, 7), date(2026, 9, 13)),
        (date(2026, 9, 14), date(2026, 9, 20)),
    ]
    assert evaluation_windows([], days=7) == []


def test_pool_interleaves_rankings_and_skips_labelled(store, config, embedder):
    seed_hl7_week(store, config)
    papers = store.papers_added_between(date(2026, 9, 1), date(2026, 9, 30))
    label(store, "pmid:3", False)

    pool = labelling_pool(store, config, embedder, papers, depth=1)
    # depth 1: dense's top (1), hybrid's top (2), keyword's top (2, already pooled)
    assert [p.pmid for p in pool] == ["1", "2"]


def test_hybrid_beats_dense_when_labels_favour_the_keyword_hit(store, config, embedder):
    seed_hl7_week(store, config)
    label(store, "pmid:1", False)
    label(store, "pmid:2", True)
    label(store, "pmid:3", False)

    dense, hybrid = evaluate(store, config, embedder, k=1)
    assert (dense.name, dense.mean("precision")) == ("dense", 0.0)
    assert (hybrid.name, hybrid.mean("precision")) == ("hybrid", 1.0)
    assert dense.mean("recall") == hybrid.mean("recall") == 1.0  # both shortlists reach it
    assert dense.windows[0].relevant == 1
    assert dense.mean("judged") == 1.0


def test_llm_variant_assesses_missing_papers_unless_cached_only(store, config, embedder):
    seed_hl7_week(store, config)
    label(store, "pmid:2", True)
    llm = FakeLLM({"Interface engines": 5})

    *_, cached_only = evaluate(store, config, embedder, llms=[llm], k=1, cached_only=True)
    assert cached_only.name == "hybrid+fake-llm"
    assert cached_only.mean("assessed") == 0.0
    assert cached_only.mean("precision") is None  # nothing assessed, nothing returned
    assert llm.calls == []

    *_, assessed = evaluate(store, config, embedder, llms=[llm], k=1)
    assert assessed.mean("assessed") == 1.0
    assert assessed.mean("precision") == 1.0


def test_windows_without_labels_are_skipped(store, config, embedder):
    seed_hl7_week(store, config)
    store.upsert_papers([make_paper(9, abstract="hl7", added=date(2026, 9, 1))])
    label(store, "pmid:9", True)
    label(store, "pmid:2", True)

    (dense, _) = evaluate(store, config, embedder, k=1)
    assert [(w.start, w.end) for w in dense.windows] == [
        (date(2026, 9, 1), date(2026, 9, 7)),
        (date(2026, 9, 22), date(2026, 9, 28)),
    ]


def test_report_flags_low_label_coverage(store, config, embedder):
    seed_hl7_week(store, config)
    config.selection.top = 3
    label(store, "pmid:2", True)

    report = format_report(evaluate(store, config, embedder), k=3, n=12)
    assert "1 window(s), 1 relevant labelled papers" in report
    assert "paperpulse label" in report
    lines = {line.split()[0]: line.split()[1:] for line in report.splitlines()[3:5]}
    assert lines["dense"][0] == "33%"


def test_report_without_labels_explains_how_to_start(store, config, embedder):
    report = format_report(evaluate(store, config, embedder), k=3, n=12)
    assert report.startswith("No labelled papers yet.")


def test_models_are_compared_side_by_side_on_the_same_labels(store, config, embedder):
    seed_hl7_week(store, config)
    label(store, "pmid:1", False)
    label(store, "pmid:2", True)
    good = FakeLLM({"Interface engines": 5})
    good.name = "good"
    bad = FakeLLM({"HL": 5})
    bad.name = "bad"

    results = evaluate(store, config, embedder, llms=[good, bad], k=1)
    by_name = {r.name: r.mean("precision") for r in results}
    assert by_name == {"dense": 0.0, "hybrid": 1.0, "hybrid+good": 1.0, "hybrid+bad": 0.0}


def test_unreachable_llm_is_reported_not_disguised_as_hybrid(store, config, embedder):
    from paperpulse.llm import LLMUnavailable

    seed_hl7_week(store, config)
    label(store, "pmid:2", True)
    down = FakeLLM(responses=iter(LLMUnavailable("down") for _ in range(10)))

    *_, result = evaluate(store, config, embedder, llms=[down], k=1)
    assert result.unavailable and result.windows == []
    report = format_report([*_, result], k=1, n=12)
    assert "hybrid+fake-llm" in report and "LLM unavailable" in report


def test_record_documents_setup_labels_and_picks(store, config, embedder):
    import hashlib
    import json

    from paperpulse.evaluation import evaluation_record
    from paperpulse.labelling import export_labels

    seed_hl7_week(store, config)
    label(store, "pmid:1", False)
    label(store, "pmid:2", True)
    llm = FakeLLM({"Interface engines": 5})
    results = evaluate(store, config, embedder, llms=[llm], k=1)
    labels_jsonl = "".join(line + "\n" for line in export_labels(store))

    record = evaluation_record(
        results, config, [llm], labels_jsonl, k=1, cached_only=False, now=NOW
    )

    assert record["complete"] and record["incomplete_reasons"] == []
    assert record["created_at"] == "2026-09-29T00:00:00+00:00"
    assert record["setup"]["llm_models"] == ["fake-llm"]
    assert record["setup"]["k"] == 1
    assert record["labels"] == {
        "total": 2,
        "relevant": 1,
        "sha256": hashlib.sha256(labels_jsonl.encode()).hexdigest()[:16],
    }
    assert record["summary"]["dense"]["precision"] == 0.0
    assert record["summary"]["hybrid+fake-llm"]["precision"] == 1.0
    window = record["windows"][0]
    assert window["variants"]["dense"]["top"] == ["pmid:1"]
    assert window["variants"]["hybrid+fake-llm"]["top"] == ["pmid:2"]
    json.dumps(record)  # must be serialisable as is


def test_record_flags_incomplete_runs(store, config, embedder):
    from paperpulse.evaluation import evaluation_record

    seed_hl7_week(store, config)
    label(store, "pmid:2", True)

    no_llm = evaluation_record(evaluate(store, config, embedder), config, [], "", 1, False)
    assert not no_llm["complete"]
    assert no_llm["incomplete_reasons"] == ["no LLM variant evaluated"]

    llm = FakeLLM()
    cached = evaluate(store, config, embedder, llms=[llm], k=1, cached_only=True)
    record = evaluation_record(cached, config, [llm], "", 1, cached_only=True)
    assert not record["complete"]
    assert "hybrid+fake-llm: only 0% of shortlist assessed" in record["incomplete_reasons"]
    assert record["setup"]["cached_only"] is True


def test_save_writes_timestamped_result_and_label_set(tmp_path):
    import json

    from paperpulse.evaluation import save_evaluation

    record = {"created_at": "2026-10-01T15:30:12+00:00", "complete": True}
    path = save_evaluation(record, '{"id": "x"}\n', tmp_path / "evaluation")

    assert path == tmp_path / "evaluation" / "results" / "2026-10-01T153012.json"
    assert json.loads(path.read_text()) == record
    assert (tmp_path / "evaluation" / "labels.jsonl").read_text() == '{"id": "x"}\n'
