import numpy as np

from paperpulse.config import Selection
from paperpulse.ranking.dense import best_topic_matches
from paperpulse.ranking.filters import apply_filters
from paperpulse.ranking.fusion import reciprocal_rank_fusion
from paperpulse.ranking.keyword import best_keyword_matches, topic_query
from tests.conftest import make_paper


def test_filters_drop_excluded_types_case_insensitively():
    papers = [
        make_paper(1, publication_types=["Journal Article"]),
        make_paper(2, publication_types=["Journal Article", "editorial"]),
    ]
    assert [p.pmid for p in apply_filters(papers, Selection(), set())] == ["1"]


def test_filters_drop_papers_without_abstract_unless_allowed():
    papers = [make_paper(1, abstract="")]
    assert apply_filters(papers, Selection(), set()) == []
    assert apply_filters(papers, Selection(require_abstract=False), set()) == papers


def test_filters_drop_excluded_ids():
    papers = [make_paper(1), make_paper(2)]
    assert [p.pmid for p in apply_filters(papers, Selection(), {"pmid:1"})] == ["2"]


def test_best_topic_match_uses_max_not_mean():
    topics = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    papers = np.array(
        [
            [0, 1, 0],  # exactly topic 1
            [0.6, 0.0, 0.8],  # weakly topic 0
        ],
        dtype=np.float32,
    )
    first, second = best_topic_matches(papers, topics)
    assert (first.topic_index, first.score) == (1, 1.0)
    assert second.topic_index == 0
    assert np.isclose(second.score, 0.6)


def test_topic_query_quotes_terms_and_drops_stopwords():
    assert topic_query("FHIR implementation and clinical data exchange") == (
        '"fhir" OR "implementation" OR "clinical" OR "data" OR "exchange"'
    )
    assert topic_query('AI-assisted "diagnosis" (NLP)') == (
        '"ai" OR "assisted" OR "diagnosis" OR "nlp"'
    )
    assert topic_query("of the and") is None


def test_keyword_ranking_prefers_rare_terms_over_generic_ones(store):
    store.upsert_papers([make_paper(n, abstract="clinical study of patients") for n in range(1, 6)])
    store.upsert_papers([make_paper(6, abstract="fhir based exchange")])
    ids = [f"pmid:{n}" for n in range(1, 7)]
    matches = best_keyword_matches(store, ["clinical patients", "FHIR"], ids)

    assert matches["pmid:6"].rank == 1
    assert matches["pmid:6"].topic_index == 1
    assert matches["pmid:6"].score > matches["pmid:1"].score
    assert sorted(m.rank for m in matches.values()) == [1, 2, 3, 4, 5, 6]


def test_rrf_rewards_agreement_and_tolerates_missing_items():
    fused = reciprocal_rank_fusion([{"a": 1, "b": 2, "c": 3}, {"c": 1, "a": 2}], k=60)
    assert fused["a"] == 1 / 61 + 1 / 62
    assert fused["b"] == 1 / 62
    assert sorted(fused, key=fused.get, reverse=True) == ["a", "c", "b"]
