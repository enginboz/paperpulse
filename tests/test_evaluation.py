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


def test_llm_variant_uses_only_cached_assessments_unless_asked(store, config, embedder):
    seed_hl7_week(store, config)
    label(store, "pmid:2", True)
    llm = FakeLLM({"Interface engines": 5})

    *_, cached_only = evaluate(store, config, embedder, llms=[llm], k=1)
    assert cached_only.name == "hybrid+fake-llm"
    assert cached_only.mean("assessed") == 0.0
    assert cached_only.mean("precision") is None  # nothing assessed, nothing returned
    assert llm.calls == []

    *_, assessed = evaluate(store, config, embedder, llms=[llm], k=1, assess_missing=True)
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

    results = evaluate(store, config, embedder, llms=[good, bad], k=1, assess_missing=True)
    by_name = {r.name: r.mean("precision") for r in results}
    assert by_name == {"dense": 0.0, "hybrid": 1.0, "hybrid+good": 1.0, "hybrid+bad": 0.0}
