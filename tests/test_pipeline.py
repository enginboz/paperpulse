from datetime import UTC, date, datetime, timedelta

from paperpulse.llm import LLMUnavailable
from paperpulse.pipeline import ingest, select
from tests.conftest import FakeLLM, make_paper

NOW = datetime(2026, 9, 29, 6, tzinfo=UTC)


def seed(store):
    store.upsert_papers(
        [
            make_paper(
                1, title="FHIR interoperability in hospitals", abstract="fhir interoperability"
            ),
            make_paper(2, title="NLP on clinical notes", abstract="clinical notes nlp"),
            make_paper(3, title="Knee surgery outcomes", abstract="orthopedic cohort"),
            make_paper(
                4,
                title="FHIR editorial",
                abstract="fhir interoperability",
                publication_types=["Editorial"],
            ),
            make_paper(
                5, title="Old FHIR paper", abstract="fhir interoperability", added=date(2026, 8, 1)
            ),
        ]
    )


def test_select_ranks_by_best_topic(store, config, embedder):
    seed(store)
    digest = select(store, config, embedder, now=NOW)

    topics = {p.pmid: p.score.matched_topic for p in digest.papers}
    assert topics == {"1": "fhir interoperability", "2": "clinical notes nlp"}
    assert [p.rank for p in digest.papers] == [1, 2]
    assert digest.papers[0].score.total >= digest.papers[1].score.total
    assert digest.run.candidates == 4  # paper 5 is outside the window
    assert digest.run.after_filters == 3  # the editorial is filtered
    assert digest.run.embedding_model == "fake-bow"


def test_select_does_not_repeat_recent_picks(store, config, embedder):
    seed(store)
    select(store, config, embedder, now=NOW)
    again = select(store, config, embedder, now=NOW + timedelta(hours=1))
    assert [p.pmid for p in again.papers] == ["3"]


def test_select_without_record_can_repeat(store, config, embedder):
    seed(store)
    first = select(store, config, embedder, now=NOW, record=False)
    second = select(store, config, embedder, now=NOW)
    assert [p.id for p in first.papers] == [p.id for p in second.papers]


def test_select_reuses_cached_embeddings(store, config, embedder):
    seed(store)
    select(store, config, embedder, now=NOW, record=False)
    select(store, config, embedder, now=NOW, record=False)
    paper_batches = [c for c in embedder.calls if c != config.profile.topics]
    assert len(paper_batches) == 1


def test_select_with_nothing_eligible(store, config, embedder):
    digest = select(store, config, embedder, now=NOW)
    assert digest.papers == []
    assert store.latest_run() is None


class FakeSource:
    name = "fake"

    def __init__(self, papers):
        self.papers = papers
        self.windows = []

    def fetch(self, start, end):
        self.windows.append((start, end))
        return self.papers


def test_ingest_resumes_from_watermark_with_overlap(store):
    source = FakeSource([make_paper(1)])
    assert ingest(store, [source], default_days=7, today=date(2026, 9, 28)) == 1
    assert ingest(store, [source], default_days=7, today=date(2026, 9, 29)) == 0
    assert source.windows == [
        (date(2026, 9, 21), date(2026, 9, 28)),
        (date(2026, 9, 27), date(2026, 9, 29)),
    ]


def test_failed_ingest_keeps_watermark(store):
    class Broken(FakeSource):
        def fetch(self, start, end):
            yield make_paper(1)
            raise RuntimeError("network down")

    try:
        ingest(store, [Broken([])], default_days=7, today=date(2026, 9, 29))
    except RuntimeError:
        pass
    assert store.last_ingested("fake") is None
    assert store.papers_added_between(date(2026, 9, 1), date(2026, 9, 30)) == []


def test_llm_relevance_overrides_dense_order(store, config, embedder):
    seed(store)
    llm = FakeLLM({"Knee surgery outcomes": 5, "NLP on clinical notes": 4})
    digest = select(store, config, embedder, llm=llm, now=NOW)

    assert [p.pmid for p in digest.papers] == ["3", "2"]
    assert digest.papers[0].score.assessment.relevance == 5
    assert digest.papers[0].score.total == round(5 + digest.papers[0].score.dense, 4)
    assert digest.run.llm_model == "fake-llm"


def test_llm_drops_papers_below_min_relevance(store, config, embedder):
    seed(store)
    digest = select(store, config, embedder, llm=FakeLLM({"NLP on clinical notes": 3}), now=NOW)
    assert [p.pmid for p in digest.papers] == ["2"]


def test_llm_only_assesses_the_dense_shortlist(store, config, embedder):
    seed(store)
    config.llm.candidates = 2
    llm = FakeLLM()
    select(store, config, embedder, llm=llm, now=NOW)
    assessed = {user.splitlines()[0] for _, user in llm.calls}
    assert "Title: Knee surgery outcomes" not in assessed
    assert len(assessed) == 2


def test_llm_unavailable_falls_back_to_dense(store, config, embedder):
    seed(store)
    llm = FakeLLM(responses=iter([LLMUnavailable("down")]))
    digest = select(store, config, embedder, llm=llm, now=NOW)
    assert {p.pmid for p in digest.papers} == {"1", "2"}
    assert digest.run.llm_model is None
    assert digest.papers[0].score.assessment is None
