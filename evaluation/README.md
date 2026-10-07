# Evaluation

Measured results of PaperPulse's ranking variants against the author's own relevance judgements. See [Evaluating the ranking](../README.md#evaluating-the-ranking) for the method.

| File | Contents |
|---|---|
| `labels.jsonl` | The label set: one paper per line with DOI, PMID, title, the judgement (`relevant`, from title and abstract only) and how it was given (`pool` via `paperpulse label`, or `import`). Post-reading feedback is not part of this set |
| `results/<timestamp>.json` | One file per `paperpulse eval` run |

Each result file records:

- **setup**: embedding model, LLM models, prompt version, a fingerprint of the profile, and the cut-offs used
- **labels**: how many labels the run used, and a checksum of `labels.jsonl` at that moment
- **summary**: precision@k, recall into the LLM shortlist and label coverage per variant, averaged over weeks
- **windows**: the same per week, plus the ids of the papers each variant would have shown
- **complete**: `false` when part of the comparison is missing (no LLM, LLM unreachable, or assessments missing), with the reasons

## Reproducing a result

```bash
uv run paperpulse labels import evaluation/labels.jsonl   # fetches the papers from PubMed
uv run paperpulse eval --models mistral llama3.2
```

Check that `labels.sha256` in the result matches the label set you imported. LLM assessments run at temperature 0 but are not guaranteed to be bit-identical across machines; the per-window `top` lists show exactly which papers each variant picked, so differences are visible.
