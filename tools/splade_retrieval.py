"""Cached SPLADE document embeddings over the corpus (encode once, rerank many)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.corpus_qa import sha256_file


def build_splade_doc_cache(
    texts: list[str],
    cache_dir: Path,
    *,
    corpus_index_path: Path,
    model_name: str,
    force: bool = False,
) -> Any:
    """
    Encode all corpus documents once. Returns a torch.Tensor [N, ...] for similarity.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "splade_doc_manifest.json"
    emb_path = cache_dir / "splade_doc_embeddings.pkl"
    corpus_sha = sha256_file(corpus_index_path)

    if not force and manifest_path.exists() and emb_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("corpus_index_sha256") == corpus_sha
            and manifest.get("model_name") == model_name
            and manifest.get("num_documents") == len(texts)
        ):
            import pickle

            print(f"Loading SPLADE doc cache from {emb_path} ...", flush=True)
            with emb_path.open("rb") as fh:
                return pickle.load(fh)

    try:
        from sentence_transformers import SparseEncoder
    except ImportError as exc:
        raise RuntimeError(
            "SPLADE requires sentence-transformers >= 3.0. Install: pip install -U sentence-transformers"
        ) from exc

    print(
        f"One-time SPLADE corpus encode ({model_name}, {len(texts):,} docs). "
        "This can take 20–60+ min on CPU; later queries are much faster.",
        flush=True,
    )
    model = SparseEncoder(model_name)
    import pickle

    doc_rows: list[Any] = []
    batch_size = 32
    from tqdm import tqdm

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="SPLADE corpus encode",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        batch_emb = model.encode_document(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        for j in range(len(batch)):
            doc_rows.append(batch_emb[j])

    with emb_path.open("wb") as fh:
        pickle.dump(doc_rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "model_name": model_name,
        "corpus_index_sha256": corpus_sha,
        "num_documents": len(texts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SPLADE doc cache saved to {emb_path}", flush=True)
    return doc_rows
