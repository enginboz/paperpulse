from datetime import UTC, date, datetime

import numpy as np

from paperpulse.models import Digest, RankedPaper, RunInfo, Score
from paperpulse.pipeline import RANKED_FIELDS
from tests.conftest import make_paper


def test_upsert_counts_only_new_papers(store):
    assert store.upsert_papers([make_paper(1), make_paper(2)]) == 2
    assert store.upsert_papers([make_paper(2), make_paper(3)]) == 1


def test_round_trips_papers_in_window(store):
    paper = make_paper(
        1, authors=["A B"], publication_types=["Journal Article"], published=date(2026, 9, 1)
    )
    store.upsert_papers([paper, make_paper(2, added=date(2026, 8, 1))])
    assert store.papers_added_between(date(2026, 9, 21), date(2026, 9, 28)) == [paper]


def test_embeddings_survive_unchanged_upsert(store):
    store.upsert_papers([make_paper(1)])
    store.put_embeddings("m", {"pmid:1": np.ones(4)})
    store.upsert_papers([make_paper(1, journal="Renamed")])
    assert "pmid:1" in store.get_embeddings(["pmid:1"], "m")


def test_embeddings_dropped_when_text_changes(store):
    store.upsert_papers([make_paper(1)])
    store.put_embeddings("m", {"pmid:1": np.ones(4)})
    store.upsert_papers([make_paper(1, abstract="revised abstract")])
    assert store.get_embeddings(["pmid:1"], "m") == {}


def test_paper_gaining_a_doi_is_rekeyed_with_its_embeddings(store):
    store.upsert_papers([make_paper(1)])
    store.put_embeddings("m", {"pmid:1": np.ones(4)})
    store.upsert_papers([make_paper(1, id="10.1/x", doi="10.1/x")])

    papers = store.papers_added_between(date(2026, 9, 1), date(2026, 9, 30))
    assert [p.id for p in papers] == ["10.1/x"]
    assert "10.1/x" in store.get_embeddings(["10.1/x"], "m")


def test_embeddings_are_per_model(store):
    store.upsert_papers([make_paper(1)])
    store.put_embeddings("a", {"pmid:1": np.array([1, 0], dtype=np.float32)})
    assert store.get_embeddings(["pmid:1"], "b") == {}
    np.testing.assert_array_equal(store.get_embeddings(["pmid:1"], "a")["pmid:1"], [1, 0])


def test_ingest_watermark(store):
    assert store.last_ingested("pubmed") is None
    store.set_last_ingested("pubmed", date(2026, 9, 29))
    assert store.last_ingested("pubmed") == date(2026, 9, 29)


def test_runs_and_selection_history(store):
    store.upsert_papers([make_paper(1)])
    created = datetime(2026, 9, 28, 6, tzinfo=UTC)
    digest = Digest(
        run=RunInfo(
            id="r1",
            created_at=created,
            window_start=date(2026, 9, 21),
            window_end=date(2026, 9, 28),
            embedding_model="m",
            candidates=1,
            after_filters=1,
        ),
        papers=[
            RankedPaper(
                rank=1,
                **make_paper(1).model_dump(include=RANKED_FIELDS),
                score=Score(total=0.5, fusion=0.5, dense=0.5, dense_rank=1, matched_topic="t"),
            )
        ],
    )
    store.save_run(digest)

    assert store.latest_run() == digest
    assert store.selected_since(datetime(2026, 9, 27, tzinfo=UTC)) == {"pmid:1"}
    assert store.selected_since(datetime(2026, 9, 29, tzinfo=UTC)) == set()


def test_keyword_search_ranks_and_restricts_to_given_ids(store):
    store.upsert_papers(
        [
            make_paper(1, title="FHIR servers", abstract="FHIR FHIR interoperability"),
            make_paper(2, title="Other", abstract="mentions fhir once among many other words"),
            make_paper(3, title="FHIR", abstract="fhir"),
        ]
    )
    hits = store.keyword_search('"fhir"', ["pmid:1", "pmid:2"])
    assert [pid for pid, _ in hits] == ["pmid:1", "pmid:2"]
    assert hits[0][1] > hits[1][1] > 0
    assert store.keyword_search('"fhir"', []) == []


def test_keyword_index_stems_words(store):
    store.upsert_papers([make_paper(1, abstract="we implemented predictive models")])
    assert store.keyword_search('"implementation" OR "prediction"', ["pmid:1"])


def test_keyword_index_follows_text_changes_and_rekeying(store):
    store.upsert_papers([make_paper(1, abstract="about fhir")])
    store.upsert_papers([make_paper(1, abstract="about omop")])
    assert store.keyword_search('"fhir"', ["pmid:1"]) == []
    assert store.keyword_search('"omop"', ["pmid:1"])

    store.upsert_papers([make_paper(1, id="10.1/x", doi="10.1/x", abstract="about omop")])
    assert [pid for pid, _ in store.keyword_search('"omop"', ["10.1/x"])] == ["10.1/x"]


def test_keyword_index_is_backfilled_for_older_databases(tmp_path):
    from paperpulse.store import Store

    path = tmp_path / "old.db"
    s = Store(path)
    s.upsert_papers([make_paper(1, abstract="about fhir")])
    s.db.execute("DELETE FROM papers_fts")
    s.db.commit()
    s.close()

    reopened = Store(path)
    assert reopened.keyword_search('"fhir"', ["pmid:1"])
    reopened.close()
