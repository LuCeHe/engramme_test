"""Dense retrieval over a small corpus index (laptop-friendly)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.corpus_qa import CorpusRecord, load_corpus_index, sha256_file

# L3-v2: ~2× faster encode, smaller than all-MiniLM-L6-v2; L6 is stronger retrieval.
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
DEFAULT_EMBED_MODEL_QUALITY = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 50


def record_to_text(rec: CorpusRecord) -> str:
    senders = ", ".join(rec.senders) or "unknown sender"
    recipients = ", ".join(rec.recipients) or "unknown recipients"
    subject = rec.subject or "(no subject)"
    date = rec.date_utc[:10] if rec.date_utc else "unknown date"
    return f"date {date} | from {senders} | to {recipients} | subject {subject}"


_ENCODER_CACHE: dict[str, Any] = {}


def _load_encoder(model_name: str):
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from exc

    _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]


def build_dense_index(
    index_path: Path,
    cache_dir: Path,
    model_name: str = DEFAULT_EMBED_MODEL,
    force: bool = False,
) -> dict[str, Any]:
    """Embed corpus rows; cache vectors + record_id order on disk."""
    index_path = index_path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_sha = sha256_file(index_path)

    meta_path = cache_dir / "dense_index_manifest.json"
    ids_path = cache_dir / "dense_record_ids.json"
    vectors_path = cache_dir / "dense_vectors.npy"

    if not force and meta_path.exists() and vectors_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("corpus_index_sha256") == index_sha
            and meta.get("model_name") == model_name
        ):
            record_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path)
            return {
                "record_ids": record_ids,
                "vectors": vectors,
                "model_name": model_name,
                "manifest": meta,
            }

    records = load_corpus_index(index_path)
    texts = [record_to_text(r) for r in records]
    record_ids = [r.record_id for r in records]

    print(
        f"Encoding {len(texts):,} corpus rows with {model_name} "
        "(first run may download the model) ..."
    )
    model = _load_encoder(model_name)
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    np.save(vectors_path, vectors)
    ids_path.write_text(json.dumps(record_ids), encoding="utf-8")
    meta = {
        "corpus_index_sha256": index_sha,
        "model_name": model_name,
        "num_vectors": len(record_ids),
        "vectors_file": str(vectors_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "record_ids": record_ids,
        "vectors": vectors,
        "model_name": model_name,
        "manifest": meta,
    }


def retrieve_top_k_timed(
    question: str,
    dense_index: dict[str, Any],
    k: int = DEFAULT_TOP_K,
) -> tuple[list[str], dict[str, float]]:
    import time

    encode_start = time.perf_counter()
    model = _load_encoder(dense_index["model_name"])
    query = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]
    encode_ms = (time.perf_counter() - encode_start) * 1000

    score_start = time.perf_counter()
    vectors: np.ndarray = dense_index["vectors"]
    scores = vectors @ query
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    score_ms = (time.perf_counter() - score_start) * 1000

    record_ids: list[str] = dense_index["record_ids"]
    ids = [record_ids[i] for i in top.tolist()]
    total_ms = encode_ms + score_ms
    return ids, {
        "dense_encode_ms": round(encode_ms, 4),
        "dense_score_ms": round(score_ms, 4),
        "total_ms": round(total_ms, 4),
    }


def retrieve_top_k(
    question: str,
    dense_index: dict[str, Any],
    k: int = DEFAULT_TOP_K,
) -> list[str]:
    ids, _ = retrieve_top_k_timed(question, dense_index, k)
    return ids


def recall_at_k(retrieved: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    hit = len(set(retrieved) & gold)
    return hit / len(gold)


def aggregate_retrieval_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    recalls = [
        r["retrieval_recall_at_k"]
        for r in rows
        if r.get("retrieval_recall_at_k") is not None
    ]
    return {
        "items_with_retrieval": len(recalls),
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "median_recall_at_k": sorted(recalls)[len(recalls) // 2]
        if recalls
        else None,
    }
