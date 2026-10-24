# Engramme_test

Enron **count Q&A** on a 10K email slice: synthetic questions with exact gold sets, retrieval benchmarks.

```bat
pip install rank-bm25 sentence-transformers openai-codex kaggle
python eval_enron.py
```

---

## Corpus Q&A (how it was built)

1. **Corpus slice** — Reservoir sample **10,000** emails from the Kaggle Enron CSV (`seed=42`) → `corpus_sample_10000.jsonl`.
2. **Index** — Parse headers (sender, recipients, date, subject) → `corpus_index_10000.jsonl`.
3. **Inverted index** — Lists by sender/recipient; dates sorted for before/after filters.
4. **1,000 questions** — Round-robin over **9 templates** (`seed=42`); each question stores `relevant_record_ids` = exact matching rows (algorithmic gold, no LLM).
5. **Templates** — `sent_by`, `before`, `after`, `received_by`, plus AND/OR combos (`sent_by_and_before`, `sent_by_or_received_by`, …). Questions are natural-language **counts** (“How many emails were sent by …?”).
6. **Eval** — Retrieval: **mean recall@50** on 100 held questions. Codex (optional): answer from top‑k evidence vs `answer_count`.


---

## Retrieval results

**Setup:** 100 synthetic questions, **k = 50**, corpus **10K**. Metric: **mean recall@k** (fraction of gold IDs in top‑k). Latency: **mean ms per query** (retrieval only).

| Method | Mean recall@50 | Mean latency | Notes |
|--------|----------------|--------------|--------|
| **Dense** | 36.6% | 678 ms | MiniLM-L3 over full corpus |
| **BM25** | 29.4% | 39 ms | Lexical baseline |
| **BM25 + SPLADE** | — | — | Re-run after SPLADE doc cache (`--rebuild-splade-cache` once) |
| **ColBERT** | — | — | — |
| **Cascade** | — | — | BM25 → SPLADE → dense → cross-encoder |

The last three methods are functional, but they were taking longer than the 3h assigned.



## Methods

| Method | What it does |
|--------|----------------|
| **dense** | Embed query + all docs with **paraphrase-MiniLM-L3-v2**; cosine top‑k. |
| **bm25** | **BM25** over tokenized metadata text (date, from, to, subject). |
| **bm25_splade** | BM25 shortlist (≤200) → **SPLADE** rerank → top‑k. Doc vectors cached once. |
| **colbert** | **ColBERT v2** late interaction (RAGatouille) over full corpus. |
| **cascade** | **BM25 → SPLADE → dense → cross-encoder** |

Run one method: benchmark with `--methods <name>`, or Codex Q&A with `--retrieval-method <name>` (`dense`, `bm25`, `bm25_splade`, `colbert`, `cascade`).

```bat
python -m tools.retrieval_benchmark_report --methods colbert --output-json results/engramme_test/retrieval_benchmark_colbert.json
python eval_enron.py --retrieval-method cascade --corpus-qa-limit 10
```

## AI traces

- ChatGPT: https://chatgpt.com/share/6adc2cbf-a418-83ed-a5c7-8cb6b0aea34d
- cursor_ai_chat.md