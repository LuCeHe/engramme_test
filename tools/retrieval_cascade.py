"""Multi-stage retrieval: BM25 → SPLADE → dense → cross-encoder cascade."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.colbert_retrieval import (
    DEFAULT_COLBERT_MODEL,
    ColBERTRetriever,
    build_colbert_index,
)
from tools.corpus_qa import CorpusRecord, load_corpus_index
from tools.dense_retrieval import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    build_dense_index,
    recall_at_k,
    record_to_text,
)
from tools.splade_retrieval import build_splade_doc_cache

# HF id is splade-v3-distilbert (not ...-base-uncased). Use rasyosef/splade-mini for lighter CPU.
DEFAULT_SPLADE_MODEL = "naver/splade-v3-distilbert"
DEFAULT_SPLADE_MODEL_LITE = "rasyosef/splade-mini"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PRUNE_FRACTION = 0.1
DEFAULT_SPLADE_POOL = 200
DEFAULT_CROSS_POOL = 80

_TOKEN_RE = re.compile(r"[a-z0-9@._-]+", re.I)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


@dataclass
class CorpusStore:
    records: list[CorpusRecord]
    record_ids: list[str]
    texts: list[str]
    id_to_idx: dict[str, int]


def load_corpus_store(index_path: Path) -> CorpusStore:
    records = load_corpus_index(index_path)
    record_ids = [r.record_id for r in records]
    texts = [record_to_text(r) for r in records]
    id_to_idx = {rid: i for i, rid in enumerate(record_ids)}
    return CorpusStore(
        records=records,
        record_ids=record_ids,
        texts=texts,
        id_to_idx=id_to_idx,
    )


@dataclass
class RetrievalOutput:
    record_ids: list[str]
    method: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "BM25 requires rank-bm25. Install with: pip install rank-bm25"
            ) from exc

        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int) -> tuple[list[int], float]:
        started = time.perf_counter()
        scores = self._bm25.get_scores(tokenize(query))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return top.tolist(), elapsed_ms


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


class SpladeEncoder:
    """Sparse neural encoder (SPLADE-style) via sentence-transformers when available."""

    def __init__(
        self,
        model_name: str = DEFAULT_SPLADE_MODEL,
        doc_embeddings: Any | None = None,
    ):
        self.model_name = model_name
        self._model = None
        self._doc_embeddings = doc_embeddings

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise RuntimeError(
                "SPLADE requires sentence-transformers >= 3.0 with SparseEncoder. "
                "Install with: pip install -U sentence-transformers"
            ) from exc

        print(f"Loading SPLADE encoder {self.model_name} ...", flush=True)
        self._model = SparseEncoder(self.model_name)
        print("SPLADE encoder ready.", flush=True)
        return self._model

    def rank_on_indices(
        self,
        query: str,
        doc_indices: list[int],
        k: int,
    ) -> tuple[list[int], float]:
        """Score BM25 candidates using cached corpus SPLADE vectors (fast path)."""
        started = time.perf_counter()
        if self._doc_embeddings is None:
            raise RuntimeError("SPLADE doc cache missing")
        model = self._load()
        if hasattr(model, "encode_query"):
            query_emb = model.encode_query([query])
        else:
            query_emb = model.encode([query])
        import torch

        sub = torch.stack([self._doc_embeddings[i] for i in doc_indices])
        sims = model.similarity(query_emb, sub)[0]
        scores = np.asarray(sims, dtype=np.float32)
        k = min(k, len(scores))
        order = np.argsort(-scores)[:k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return order.tolist(), elapsed_ms

    def rank(
        self,
        query: str,
        doc_texts: list[str],
        k: int,
    ) -> tuple[list[int], float]:
        started = time.perf_counter()
        model = self._load()
        if hasattr(model, "encode_query"):
            query_emb = model.encode_query([query])
            doc_emb = model.encode_document(
                doc_texts,
                batch_size=32,
                show_progress_bar=False,
            )
        else:
            query_emb = model.encode([query])
            doc_emb = model.encode(
                doc_texts,
                batch_size=32,
                show_progress_bar=False,
            )
        sims = model.similarity(query_emb, doc_emb)[0]
        scores = np.asarray(sims, dtype=np.float32)
        k = min(k, len(scores))
        order = np.argsort(-scores)[:k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return order.tolist(), elapsed_ms


class DenseReranker:
    def __init__(self, dense_index: dict[str, Any]):
        self.dense_index = dense_index
        self._encoder = None

    def _encoder_model(self):
        if self._encoder is None:
            from tools.dense_retrieval import _load_encoder

            self._encoder = _load_encoder(self.dense_index["model_name"])
        return self._encoder

    def rank_on_indices(
        self,
        query: str,
        candidate_indices: list[int],
        k: int,
    ) -> tuple[list[int], float]:
        started = time.perf_counter()
        model = self._encoder_model()
        q = model.encode([query], normalize_embeddings=True)[0]
        vectors = self.dense_index["vectors"]
        sub = vectors[candidate_indices]
        scores = sub @ q
        k = min(k, len(scores))
        local = np.argpartition(-scores, k - 1)[:k]
        local = local[np.argsort(-scores[local])]
        ranked = [candidate_indices[i] for i in local.tolist()]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ranked, elapsed_ms


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(self.model_name)
        return self._model

    def rank(
        self,
        query: str,
        doc_texts: list[str],
        k: int,
    ) -> tuple[list[int], float]:
        started = time.perf_counter()
        model = self._load()
        pairs = [[query, text] for text in doc_texts]
        scores = model.predict(pairs)
        k = min(k, len(scores))
        order = np.argsort(-np.asarray(scores))[:k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return order.tolist(), elapsed_ms


@dataclass
class RetrievalPipeline:
    store: CorpusStore
    bm25: BM25Index
    splade: SpladeEncoder
    dense: DenseReranker
    cross: CrossEncoderReranker
    colbert: ColBERTRetriever | None = None
    prune_fraction: float = DEFAULT_PRUNE_FRACTION
    splade_pool: int = DEFAULT_SPLADE_POOL
    cross_pool: int = DEFAULT_CROSS_POOL

    @classmethod
    def build(
        cls,
        index_path: Path,
        dense_index: dict[str, Any],
        *,
        splade_model: str = DEFAULT_SPLADE_MODEL,
        cross_model: str = DEFAULT_CROSS_ENCODER,
        prune_fraction: float = DEFAULT_PRUNE_FRACTION,
        colbert: ColBERTRetriever | None = None,
        splade_doc_embeddings: Any | None = None,
        splade_pool: int = DEFAULT_SPLADE_POOL,
    ) -> RetrievalPipeline:
        store = load_corpus_store(index_path)
        corpus_tokens = [tokenize(t) for t in store.texts]
        bm25 = BM25Index(corpus_tokens)
        splade = SpladeEncoder(splade_model, doc_embeddings=splade_doc_embeddings)
        dense = DenseReranker(dense_index)
        cross = CrossEncoderReranker(cross_model)
        return cls(
            store=store,
            bm25=bm25,
            splade=splade,
            dense=dense,
            cross=cross,
            colbert=colbert,
            prune_fraction=prune_fraction,
            splade_pool=splade_pool,
        )

    def _indices_to_ids(self, indices: list[int]) -> list[str]:
        return [self.store.record_ids[i] for i in indices]

    def _splade_rerank(
        self, query: str, doc_indices: list[int], k: int
    ) -> tuple[list[int], float]:
        if self.splade._doc_embeddings is not None:
            return self.splade.rank_on_indices(query, doc_indices, k)
        texts = [self.store.texts[i] for i in doc_indices]
        return self.splade.rank(query, texts, k)

    def retrieve_dense_only(self, query: str, k: int) -> RetrievalOutput:
        from tools.dense_retrieval import retrieve_top_k_timed

        ids, timings = retrieve_top_k_timed(query, self.dense.dense_index, k)
        return RetrievalOutput(
            record_ids=ids,
            method="dense_only",
            timings_ms=timings,
            stage_counts={"corpus": len(self.store.record_ids), "final": len(ids)},
        )

    def retrieve_colbert_only(self, query: str, k: int) -> RetrievalOutput:
        if self.colbert is None:
            raise RuntimeError(
                "ColBERT is not loaded. Install ragatouille and include "
                "'colbert' in benchmark methods."
            )
        ids, colbert_ms = self.colbert.search(query, k)
        return RetrievalOutput(
            record_ids=ids,
            method="colbert",
            timings_ms={"colbert_ms": colbert_ms, "total_ms": colbert_ms},
            stage_counts={"corpus": len(self.store.record_ids), "final": len(ids)},
        )

    def retrieve_bm25_only(self, query: str, k: int) -> RetrievalOutput:
        indices, bm25_ms = self.bm25.search(query, k)
        return RetrievalOutput(
            record_ids=self._indices_to_ids(indices),
            method="bm25_only",
            timings_ms={"bm25_ms": bm25_ms, "total_ms": bm25_ms},
            stage_counts={"corpus": len(self.store.record_ids), "final": len(indices)},
        )

    def retrieve_bm25_splade(self, query: str, k: int) -> RetrievalOutput:
        n_bm25 = max(k, int(len(self.store.texts) * self.prune_fraction))
        n_splade = min(n_bm25, self.splade_pool)
        bm25_idx, bm25_ms = self.bm25.search(query, n_splade)
        splade_local, splade_ms = self._splade_rerank(
            query, bm25_idx, min(k, len(bm25_idx))
        )
        final_idx = [bm25_idx[i] for i in splade_local]
        total = bm25_ms + splade_ms
        return RetrievalOutput(
            record_ids=self._indices_to_ids(final_idx),
            method="bm25_splade",
            timings_ms={
                "bm25_ms": bm25_ms,
                "splade_ms": splade_ms,
                "total_ms": total,
            },
            stage_counts={
                "corpus": len(self.store.texts),
                "bm25_pool": len(bm25_idx),
                "final": len(final_idx),
            },
        )

    def retrieve_cascade(self, query: str, k: int) -> RetrievalOutput:
        """Fastest → slowest: BM25 prune → SPLADE → dense → cross-encoder."""
        n_bm25 = max(k, int(len(self.store.texts) * self.prune_fraction))
        bm25_idx, bm25_ms = self.bm25.search(query, n_bm25)

        splade_n = min(self.splade_pool, len(bm25_idx))
        splade_idx_bm25 = bm25_idx[:splade_n]
        splade_local, splade_ms = self._splade_rerank(
            query, splade_idx_bm25, min(k, splade_n)
        )
        splade_idx = [splade_idx_bm25[i] for i in splade_local]

        dense_idx, dense_ms = self.dense.rank_on_indices(
            query, splade_idx, min(self.cross_pool, len(splade_idx))
        )

        cross_texts = [self.store.texts[i] for i in dense_idx]
        cross_local, cross_ms = self.cross.rank(query, cross_texts, k)
        final_idx = [dense_idx[i] for i in cross_local]

        total = bm25_ms + splade_ms + dense_ms + cross_ms
        return RetrievalOutput(
            record_ids=self._indices_to_ids(final_idx),
            method="cascade",
            timings_ms={
                "bm25_ms": bm25_ms,
                "splade_ms": splade_ms,
                "dense_ms": dense_ms,
                "cross_encoder_ms": cross_ms,
                "total_ms": total,
            },
            stage_counts={
                "corpus": len(self.store.texts),
                "bm25_pool": len(bm25_idx),
                "splade_pool": len(splade_idx),
                "dense_pool": len(dense_idx),
                "final": len(final_idx),
            },
        )


RETRIEVAL_METHODS: dict[str, Callable[[RetrievalPipeline, str, int], RetrievalOutput]] = {
    "dense": RetrievalPipeline.retrieve_dense_only,
    "bm25": RetrievalPipeline.retrieve_bm25_only,
    "bm25_splade": RetrievalPipeline.retrieve_bm25_splade,
    "colbert": RetrievalPipeline.retrieve_colbert_only,
    "cascade": RetrievalPipeline.retrieve_cascade,
}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[rank]


def benchmark_retrieval_methods(
    pipeline: RetrievalPipeline,
    questions: list[dict[str, Any]],
    *,
    k: int,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    methods = methods or list(RETRIEVAL_METHODS)
    summary: dict[str, Any] = {"k": k, "questions": len(questions), "methods": {}}

    for name in methods:
        fn = RETRIEVAL_METHODS.get(name)
        if fn is None:
            continue

        recalls: list[float] = []
        totals_ms: list[float] = []
        stage_totals: dict[str, list[float]] = {}

        print(f"Benchmarking `{name}` on {len(questions)} questions (k={k}) ...", flush=True)
        query_secs: list[float] = []
        try:
            for qi, q in enumerate(questions):
                t0 = time.perf_counter()
                gold = set(q.get("relevant_record_ids") or [])
                out = fn(pipeline, q["question"], k)
                dt = time.perf_counter() - t0
                query_secs.append(dt)
                if qi == 0:
                    est_total = dt * len(questions)
                    print(
                        f"  `{name}`: first query {dt:.1f}s → "
                        f"~{_format_duration(est_total)} estimated for all {len(questions)}",
                        flush=True,
                    )
                elif (qi + 1) % 10 == 0 or qi + 1 == len(questions):
                    avg = sum(query_secs) / len(query_secs)
                    remaining = avg * (len(questions) - qi - 1)
                    print(
                        f"  `{name}`: {qi + 1}/{len(questions)} done, "
                        f"~{_format_duration(remaining)} remaining "
                        f"(avg {avg:.1f}s/query)",
                        flush=True,
                    )
                recall = recall_at_k(out.record_ids, gold)
                if recall is not None:
                    recalls.append(recall)
                totals_ms.append(out.timings_ms.get("total_ms", 0.0))
                for stage, ms in out.timings_ms.items():
                    stage_totals.setdefault(stage, []).append(ms)
        except Exception as exc:
            summary["methods"][name] = {"error": str(exc)}
            continue

        summary["methods"][name] = {
            "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
            "latency_ms": {
                "mean_total": sum(totals_ms) / len(totals_ms) if totals_ms else None,
                "p50_total": _percentile(totals_ms, 50),
                "p95_total": _percentile(totals_ms, 95),
                "mean_by_stage": {
                    stage: sum(vals) / len(vals) for stage, vals in stage_totals.items()
                },
            },
        }

    return summary


def retrieve_for_qa(
    pipeline: RetrievalPipeline,
    query: str,
    k: int,
    method: str,
) -> RetrievalOutput:
    fn = RETRIEVAL_METHODS.get(method, RETRIEVAL_METHODS["cascade"])
    return fn(pipeline, query, k)


def _methods_need_dense_index(methods: list[str] | None) -> bool:
    if not methods:
        return True
    return any(m in ("dense", "cascade") for m in methods)


def _methods_need_colbert(methods: list[str] | None) -> bool:
    return methods is not None and "colbert" in methods


def _methods_need_splade_cache(methods: list[str] | None) -> bool:
    if not methods:
        return True
    return any(m in ("bm25_splade", "cascade") for m in methods)


def build_pipeline(
    index_path: Path,
    cache_dir: Path,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    force_dense: bool = False,
    force_colbert: bool = False,
    splade_model: str = DEFAULT_SPLADE_MODEL,
    cross_model: str = DEFAULT_CROSS_ENCODER,
    colbert_model: str = DEFAULT_COLBERT_MODEL,
    prune_fraction: float = DEFAULT_PRUNE_FRACTION,
    splade_pool: int = DEFAULT_SPLADE_POOL,
    force_splade_cache: bool = False,
    methods: list[str] | None = None,
) -> RetrievalPipeline:
    store = load_corpus_store(index_path)

    if _methods_need_dense_index(methods):
        dense_index = build_dense_index(
            index_path,
            cache_dir,
            model_name=embed_model,
            force=force_dense,
        )
    else:
        dense_index = {
            "record_ids": [],
            "vectors": np.zeros((0, 1), dtype=np.float32),
            "model_name": embed_model,
        }

    colbert_retriever = None
    if _methods_need_colbert(methods):
        rag = build_colbert_index(
            store.texts,
            store.record_ids,
            cache_dir,
            corpus_index_path=index_path,
            model_name=colbert_model,
            force=force_colbert,
        )
        colbert_retriever = ColBERTRetriever(rag)

    splade_doc_emb = None
    if _methods_need_splade_cache(methods):
        splade_doc_emb = build_splade_doc_cache(
            store.texts,
            cache_dir,
            corpus_index_path=index_path,
            model_name=splade_model,
            force=force_splade_cache,
        )

    return RetrievalPipeline.build(
        index_path,
        dense_index,
        splade_model=splade_model,
        cross_model=cross_model,
        prune_fraction=prune_fraction,
        colbert=colbert_retriever,
        splade_doc_embeddings=splade_doc_emb,
        splade_pool=splade_pool,
    )
