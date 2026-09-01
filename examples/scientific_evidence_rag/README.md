# Scientific Materials Evidence RAG

A public-safe retrieval-and-evaluation demo for scientific evidence workflows.

The goal is not to build another PDF chatbot. The demo separates four concerns that are easy to conflate in scientific RAG systems:

1. **retrieval** — did the system retrieve the right evidence?
2. **grounding** — is the answer supported by retrieved evidence?
3. **citation coverage** — can each material claim be traced to evidence IDs?
4. **abstention** — does the system refuse unsupported questions instead of inventing an answer?

## Why this exists

Scientific and engineering workflows often mix papers, patents, experimental notes, model cards and process documents. A useful assistant must preserve provenance and uncertainty rather than merely produce fluent text. This example therefore ships with a deterministic offline retrieval evaluator and an optional OpenAI `file_search` path.

## Public-safety boundary

- Only public or synthetic demonstration text belongs here.
- No employer-confidential data, unpublished experimental results, private PDFs, credentials or private paths.
- The included corpus is deliberately small and non-sensitive; it demonstrates evaluation mechanics rather than domain completeness.

## Layout

- `corpus/evidence.jsonl` — public/synthetic evidence records with stable evidence IDs.
- `evals/retrieval_eval.jsonl` — 30+ retrieval and abstention queries.
- `offline_eval.py` — stdlib lexical retrieval baseline and deterministic metrics.
- `online_openai_file_search.py` — optional OpenAI Responses API + hosted `file_search` demo. Requires the `openai` Python package and `OPENAI_API_KEY`; the model is supplied via `OPENAI_MODEL` rather than hard-coded.

## Offline evaluation

```bash
python examples/scientific_evidence_rag/offline_eval.py
```

The evaluator reports:

- hit@1
- hit@3
- reciprocal-rank mean
- unsupported-query abstention accuracy

The offline retriever is intentionally simple. It is a transparent lower bound, not a claim that lexical retrieval is sufficient for scientific RAG.

## Optional OpenAI hosted retrieval

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=<current-model-id>
python examples/scientific_evidence_rag/online_openai_file_search.py "What does the evidence say about human authority in the agent workflow?"
```

The script uploads only the included public-safe demo evidence, creates a temporary vector store, calls the Responses API with `file_search`, requests retrieval results, prints a compact grounded answer and then attempts cleanup.

The exact OpenAI SDK/API surface changes over time; verify the current official OpenAI file-search documentation before production use. This example deliberately avoids embedding a stale model ID.

## Hiring-evidence interpretation

A passing demo proves only that the repository contains a reproducible retrieval/evaluation scaffold. It does **not** prove production RAG deployment, enterprise scale, domain completeness or autonomous scientific discovery.
