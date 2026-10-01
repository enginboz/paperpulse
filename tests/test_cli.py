import json
import os
from datetime import date

from paperpulse import cli
from paperpulse.store import Store
from tests.conftest import FakeEmbedder, FakeLLM, make_paper


def test_schema_command(capsys):
    assert cli.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "Digest"
    assert "papers" in schema["properties"]


def test_missing_config_fails_cleanly(tmp_path):
    assert cli.main(["-c", str(tmp_path / "nope.toml"), "select"]) == 2


def test_select_prints_json_digest(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    monkeypatch.setattr(cli, "_llm", lambda _: FakeLLM({"Paper 1": 5}))
    config = tmp_path / "paperpulse.toml"
    config.write_text('[profile]\ntopics = ["fhir interoperability"]\n')
    db = tmp_path / "pp.db"
    store = Store(db)
    store.upsert_papers([make_paper(1, abstract="fhir interoperability", added=date.today())])
    store.close()

    assert cli.main(["-c", str(config), "--db", str(db), "select", "--top", "1"]) == 0
    digest = json.loads(capsys.readouterr().out)
    assert digest["schema_version"] == "1"
    assert digest["papers"][0]["pmid"] == "1"
    assert digest["papers"][0]["score"]["assessment"]["relevance"] == 5
    assert digest["run"]["llm_model"] == "fake-llm"


def test_dotenv_in_working_directory_is_loaded(tmp_path, monkeypatch, capsys):
    # setenv then delenv: the variables start unset, and teardown removes what .env adds
    for var in ("PAPERPULSE_DB", "PUBMED_EMAIL"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    (tmp_path / ".env").write_text("PAPERPULSE_DB=from-dotenv.db\nPUBMED_EMAIL=me@example.org\n")
    (tmp_path / "paperpulse.toml").write_text('[profile]\ntopics = ["x"]\n[llm]\nenabled = false\n')

    assert cli.main(["select"]) == 0
    assert (tmp_path / "from-dotenv.db").exists()
    assert os.environ["PUBMED_EMAIL"] == "me@example.org"


def test_shell_variables_win_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBMED_EMAIL", "shell@example.org")
    (tmp_path / ".env").write_text("PUBMED_EMAIL=file@example.org\n")
    cli.main(["schema"])
    assert os.environ["PUBMED_EMAIL"] == "shell@example.org"


def test_history_lists_and_shows_saved_digests(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _: FakeEmbedder())
    config = tmp_path / "paperpulse.toml"
    config.write_text('[profile]\ntopics = ["fhir"]\n[llm]\nenabled = false\n')
    db = tmp_path / "pp.db"
    base = ["-c", str(config), "--db", str(db)]

    assert cli.main([*base, "history"]) == 0
    assert "No saved digests yet" in capsys.readouterr().out

    store = Store(db)
    store.upsert_papers([make_paper(1, abstract="fhir", added=date.today())])
    store.close()
    cli.main([*base, "select"])
    run_id = json.loads(capsys.readouterr().out)["run"]["id"]

    assert cli.main([*base, "history"]) == 0
    listing = capsys.readouterr().out
    assert run_id[:8] in listing and "1. Paper 1" in listing

    for selector in (date.today().isoformat(), run_id[:6]):
        assert cli.main([*base, "history", selector]) == 0
        assert json.loads(capsys.readouterr().out)["run"]["id"] == run_id
    assert cli.main([*base, "history", "1999-01-01"]) == 1
