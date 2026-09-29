import json
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
