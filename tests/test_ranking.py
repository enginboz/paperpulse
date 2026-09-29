import numpy as np

from paperpulse.config import Selection
from paperpulse.ranking.dense import best_topic_matches
from paperpulse.ranking.filters import apply_filters
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
