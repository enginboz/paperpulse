import json
from datetime import UTC, date, datetime

from paperpulse import cli
from paperpulse.labelling import (
    export_labels,
    import_labels,
    label_interactively,
    record_feedback,
    resolve_paper,
)
from paperpulse.models import Label
from paperpulse.pipeline import select
from paperpulse.store import Store
from tests.conftest import FakeEmbedder, FakeLLM, make_paper

NOW = datetime(2026, 9, 29, 6, tzinfo=UTC)


def _label(pid: str, relevant: bool, source="pool") -> Label:
    return Label(paper_id=pid, relevant=relevant, source=source, labelled_at=NOW)


def test_later_labels_replace_earlier_ones(store):
    store.upsert_papers([make_paper(1)])
    store.set_label(_label("pmid:1", True))
    store.set_label(_label("pmid:1", False, source="feedback"))
    labels = store.get_labels()
    assert (labels["pmid:1"].relevant, labels["pmid:1"].source) == (False, "feedback")


def test_labels_follow_a_paper_that_gains_a_doi(store):
    store.upsert_papers([make_paper(1)])
    store.set_label(_label("pmid:1", True))
    store.upsert_papers([make_paper(1, id="10.1/x", doi="10.1/x")])
    assert list(store.get_labels()) == ["10.1/x"]


def test_resolve_paper_by_digest_rank_doi_or_pmid(store, config, embedder):
    store.upsert_papers(
        [
            make_paper(1, abstract="fhir interoperability"),
            make_paper(2, id="10.1/abc", doi="10.1/ABC", abstract="clinical notes nlp"),
        ]
    )
    digest = select(store, config, embedder, now=NOW)

    assert resolve_paper(store, "1").id == digest.papers[0].id
    assert resolve_paper(store, "10.1/ABC").id == "10.1/abc"
    assert resolve_paper(store, "pmid:1").id == "pmid:1"
    assert resolve_paper(store, "9") is None
    assert resolve_paper(store, "10.9/missing") is None


def test_feedback_is_stored_as_a_label(store):
    store.upsert_papers([make_paper(1)])
    record_feedback(store, make_paper(1), relevant=True, note="great method", now=NOW)
    assert store.get_labels()["pmid:1"] == Label(
        paper_id="pmid:1", relevant=True, source="feedback", note="great method", labelled_at=NOW
    )


def test_interactive_labelling_handles_skip_invalid_and_quit(store):
    papers = [make_paper(n) for n in (1, 2, 3, 4)]
    store.upsert_papers(papers)
    answers = iter(["y", "maybe", "n", "s", "q"])
    shown = []

    written = label_interactively(store, papers, ask=lambda _: next(answers), show=shown.append)

    assert written == 2
    assert {pid: lab.relevant for pid, lab in store.get_labels().items()} == {
        "pmid:1": True,
        "pmid:2": False,
    }
    assert "Please answer y, n, s or q." in shown
    assert "[1/4]" in shown[0] and "Paper 1" in shown[0]


def test_export_import_round_trip_fetches_missing_papers(store, tmp_path):
    store.upsert_papers([make_paper(1), make_paper(2)])
    store.set_label(_label("pmid:1", True))
    store.set_label(_label("pmid:2", False))
    lines = list(export_labels(store))
    assert json.loads(lines[0])["relevant"] is True

    fresh = Store(tmp_path / "fresh.db")
    fresh.upsert_papers([make_paper(1)])
    requested = []

    def fetch(pmids):
        requested.extend(pmids)
        return [make_paper(int(p)) for p in pmids]

    assert import_labels(fresh, lines, fetch_by_pmid=fetch) == (2, 0)
    assert requested == ["2"]
    assert {lab.source for lab in fresh.get_labels().values()} == {"import"}
    fresh.close()


def test_import_skips_papers_it_cannot_find(store):
    line = json.dumps({"id": "10.1/gone", "pmid": None, "relevant": True})
    assert import_labels(store, [line]) == (0, 1)


def test_cli_feedback_and_export(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    config = tmp_path / "paperpulse.toml"
    config.write_text('[profile]\ntopics = ["fhir"]\n[llm]\nenabled = false\n')
    db = tmp_path / "pp.db"
    s = Store(db)
    s.upsert_papers([make_paper(1, abstract="fhir", added=date.today())])
    s.close()
    base = ["-c", str(config), "--db", str(db)]

    assert cli.main([*base, "select"]) == 0
    assert cli.main([*base, "feedback", "1", "up", "--note", "useful"]) == 0
    assert "👍 Paper 1" in capsys.readouterr().out
    assert cli.main([*base, "feedback", "7", "down"]) == 1

    assert cli.main([*base, "labels", "export"]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert (exported["id"], exported["relevant"], exported["note"]) == ("pmid:1", True, "useful")


def _cli_setup(tmp_path, monkeypatch, toml_extra=""):
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    config = tmp_path / "paperpulse.toml"
    config.write_text(f'[profile]\ntopics = ["fhir"]\n{toml_extra}')
    db = tmp_path / "pp.db"
    s = Store(db)
    s.upsert_papers([make_paper(n, abstract="fhir", added=date.today()) for n in range(1, 4)])
    s.close()
    return ["-c", str(config), "--db", str(db)]


def test_cli_label_pool_goes_as_deep_as_the_llm_shortlist(tmp_path, monkeypatch, capsys):
    base = _cli_setup(tmp_path, monkeypatch, "[llm]\ncandidates = 15\n")
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert cli.main([*base, "label"]) == 0
    assert "from the top 15 of each ranking" in capsys.readouterr().out


def test_cli_eval_compares_requested_models(tmp_path, monkeypatch, capsys):
    base = _cli_setup(tmp_path, monkeypatch)

    def fake_llm(config):
        llm = FakeLLM({"Paper 1": 5})
        llm.name = config.llm.model
        return llm

    monkeypatch.setattr(cli, "_llm", fake_llm)
    s = Store(tmp_path / "pp.db")
    s.set_label(Label(paper_id="pmid:1", relevant=True, source="pool", labelled_at=NOW))
    s.close()

    assert cli.main([*base, "eval", "--models", "mistral", "llama3.2"]) == 0
    report = capsys.readouterr().out
    assert "hybrid+mistral" in report and "hybrid+llama3.2" in report


def test_cli_eval_saves_by_default_and_not_with_no_save(tmp_path, monkeypatch, capsys):
    base = _cli_setup(tmp_path, monkeypatch, "[llm]\nenabled = false\n")
    s = Store(tmp_path / "pp.db")
    s.set_label(Label(paper_id="pmid:1", relevant=True, source="pool", labelled_at=NOW))
    s.close()

    assert cli.main([*base, "eval", "--no-save"]) == 0
    assert not (tmp_path / "evaluation").exists()

    assert cli.main([*base, "eval"]) == 0
    saved = list((tmp_path / "evaluation" / "results").glob("*.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert record["complete"] is False  # LLM disabled in this config
    assert json.loads((tmp_path / "evaluation" / "labels.jsonl").read_text())["id"] == "pmid:1"


def test_cli_eval_without_labels_saves_nothing(tmp_path, monkeypatch):
    base = _cli_setup(tmp_path, monkeypatch, "[llm]\nenabled = false\n")
    assert cli.main([*base, "eval"]) == 0
    assert not (tmp_path / "evaluation").exists()


def test_label_view_shows_as_much_abstract_as_the_llm_sees(store):
    from paperpulse.ranking.assess import MAX_ABSTRACT_CHARS, build_user_prompt

    long_paper = make_paper(1, abstract="word " * 1000)  # 5000 characters
    store.upsert_papers([long_paper])
    shown = []
    label_interactively(store, [long_paper], ask=lambda _: "q", show=shown.append)

    llm_abstract = build_user_prompt(long_paper).split("Abstract:\n", 1)[1]
    assert len(llm_abstract) == MAX_ABSTRACT_CHARS
    assert llm_abstract in shown[0]
    assert shown[0].rstrip().endswith("…")
