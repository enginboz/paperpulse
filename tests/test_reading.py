from datetime import UTC, date, datetime

from paperpulse import cli
from paperpulse.pipeline import select
from paperpulse.reading import format_reading_list
from paperpulse.store import Store
from tests.conftest import FakeEmbedder, FakeLLM, make_paper

NOW = datetime(2026, 10, 7, 8, tzinfo=UTC)


def digest_with_extras(store, config, embedder):
    config.selection.top = 1
    store.upsert_papers(
        [
            make_paper(
                1,
                title="NLP on clinical notes",
                abstract="clinical notes nlp",
                id="10.1/nlp",
                doi="10.1/NLP",
                added=date(2026, 10, 5),
                published=date(2026, 10, 1),
            ),
            make_paper(
                2,
                title="FHIR in hospitals",
                abstract="fhir interoperability",
                id="10.1/fhir",
                doi="10.1/fhir",
                added=date(2026, 10, 5),
            ),
            make_paper(3, title="Knee surgery", abstract="orthopedic", added=date(2026, 10, 5)),
        ]
    )
    llm = FakeLLM({"NLP on clinical notes": 5, "FHIR in hospitals": 4, "Knee surgery": 2})
    return select(store, config, embedder, llm=llm, now=NOW)


def test_reading_list_shows_picks_with_rationale_and_doi_link(store, config, embedder):
    text = format_reading_list(digest_with_extras(store, config, embedder))

    assert "1 paper(s) · fake-llm" in text
    assert "Papers added to PubMed 2026-10-01 to 2026-10-07" in text
    assert "1. NLP on clinical notes" in text
    assert "J Test · 2026-10-01 · original research · relevance 5/5" in text
    assert "Rated NLP on clinical notes." in text
    assert "https://doi.org/10.1/NLP" in text  # the DOI as published, not lowercased


def test_reading_list_adds_well_rated_runners_up_with_ids(store, config, embedder):
    digest = digest_with_extras(store, config, embedder)
    text = format_reading_list(digest)

    extras = text.split("Also rated 4+ by the LLM")[1]
    assert "FHIR in hospitals" in extras and "id: 10.1/fhir" in extras
    assert "https://doi.org/10.1/fhir" in extras
    assert "Knee surgery" not in text  # rated 2

    assert "Also rated" not in format_reading_list(digest, extras=False)


def test_reading_list_without_llm_or_papers(store, config, embedder):
    store.upsert_papers([make_paper(1, abstract="fhir", added=date(2026, 10, 5))])
    text = format_reading_list(select(store, config, embedder, now=NOW, record=False))
    assert "no LLM" in text and "relevance" not in text
    assert "https://pubmed.ncbi.nlm.nih.gov/1/" in text  # no DOI: PubMed link

    store.db.execute("DELETE FROM papers")
    empty = format_reading_list(select(store, config, embedder, now=NOW, record=False))
    assert "Nothing in this digest" in empty


def test_cli_read_latest_or_by_date(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    config = tmp_path / "paperpulse.toml"
    config.write_text('[profile]\ntopics = ["fhir"]\n[llm]\nenabled = false\n')
    db = tmp_path / "pp.db"
    base = ["-c", str(config), "--db", str(db)]

    assert cli.main([*base, "read"]) == 0
    assert "No saved digests yet" in capsys.readouterr().out

    s = Store(db)
    s.upsert_papers([make_paper(1, title="FHIR paper", abstract="fhir", added=date.today())])
    s.close()
    cli.main([*base, "select"])
    capsys.readouterr()

    for args in ([], [date.today().isoformat()]):
        assert cli.main([*base, "read", *args]) == 0
        assert "1. FHIR paper" in capsys.readouterr().out
    assert cli.main([*base, "read", "1999-01-01"]) == 1
