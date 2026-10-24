"""ColBERT retrieval via RAGatouille (late interaction)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.corpus_qa import sha256_file

DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_INDEX_NAME = "engramme_corpus_colbert"

_RAGATOUILLE_IMPORT_ERROR: str | None = None


def _ensure_colbert_transformers_compat() -> None:
    """
    colbert-ai HF_ColBERT omits post_init(), which transformers 5.x requires
    (all_tied_weights_keys) during from_pretrained finalization.
    """
    try:
        import colbert.modeling.hf_colbert as hf_colbert_mod
    except ImportError:
        return

    if getattr(hf_colbert_mod, "_ENGRAMME_TRANSFORMERS5_PATCH", False):
        return

    original_factory = hf_colbert_mod.class_factory

    def class_factory(name_or_path):  # noqa: ANN001 — matches upstream signature
        hf_colbert_cls = original_factory(name_or_path)
        original_init = hf_colbert_cls.__init__

        def __init__(self, config, colbert_config):  # noqa: ANN001
            original_init(self, config, colbert_config)
            if hasattr(self, "post_init"):
                self.post_init()

        hf_colbert_cls.__init__ = __init__
        return hf_colbert_cls

    hf_colbert_mod.class_factory = class_factory
    # colbert copies `class_factory` at import time; rebind after patch.
    import importlib
    import sys

    for mod_name in (
        "colbert.modeling.base_colbert",
        "colbert.modeling.tokenization.query_tokenization",
        "colbert.modeling.tokenization.doc_tokenization",
    ):
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "class_factory"):
            mod.class_factory = class_factory
        sys.modules[mod_name] = mod

    hf_colbert_mod._ENGRAMME_TRANSFORMERS5_PATCH = True


def _segmented_maxsim_py(scores, lengths):
    """MaxSim: per document, max over tokens then sum over query vectors."""
    import torch

    lengths = lengths.to(dtype=torch.int64, device="cpu")
    starts = torch.zeros(lengths.size(0), dtype=torch.int64)
    if lengths.numel() > 1:
        starts[1:] = torch.cumsum(lengths[:-1], dim=0)
    out = scores.new_zeros(lengths.size(0))
    for i, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
        if length <= 0:
            continue
        out[i] = scores[start : start + length].max(dim=0).values.sum()
    return out


def _segmented_lookup_py(tensor, pids, lengths, offsets):
    import torch

    lengths = lengths.to(dtype=torch.int64, device="cpu")
    offsets = offsets.to(dtype=torch.int64, device="cpu")
    pieces = [
        tensor[int(off) : int(off) + int(length)]
        for off, length in zip(offsets.tolist(), lengths.tolist())
    ]
    if not pieces:
        return tensor.new_empty((0, *tensor.shape[1:]))
    return torch.cat(pieces, dim=0)


def _approx_pid_scores(pids, centroid_scores, codes, doclens, offsets, idx):
    import torch

    pids = torch.as_tensor(pids, dtype=torch.int64)
    scores = centroid_scores.new_full((pids.numel(),), -9999.0)
    for i, pid in enumerate(pids.tolist()):
        start = int(offsets[pid])
        n = int(doclens[pid])
        if n <= 0:
            continue
        doc_codes = codes[start : start + n].long()
        if idx.dtype == torch.bool and not bool(idx.all()):
            doc_codes = doc_codes[idx[doc_codes]]
        if doc_codes.numel() == 0:
            continue
        doc_codes = torch.unique(doc_codes)
        scores[i] = centroid_scores[doc_codes].max(dim=0).values.sum()
    return scores


def _filter_pids_py(
    pids,
    centroid_scores,
    codes,
    doclens,
    offsets,
    idx,
    nfiltered_docs,
):
    import torch

    def _top(current_pids, current_idx, k):
        approx = _approx_pid_scores(
            current_pids, centroid_scores, codes, doclens, offsets, current_idx
        )
        k = min(int(k), int(approx.numel()))
        if k <= 0:
            return current_pids.new_empty((0,), dtype=torch.int32)
        chosen = approx.topk(k).indices
        return torch.as_tensor(current_pids, dtype=torch.int32)[chosen]

    first = _top(pids, idx, nfiltered_docs)
    all_centroids = torch.ones(
        centroid_scores.size(0), dtype=torch.bool, device=centroid_scores.device
    )
    return _top(first, all_centroids, int(nfiltered_docs) // 4)


def _decompress_residuals_py(
    pids,
    lengths,
    offsets,
    bucket_weights,
    reversed_bit_map,
    bucket_weight_combinations,
    binary_residuals,
    codes,
    centroids,
    dim,
    nbits,
):
    import torch

    pids = torch.as_tensor(pids, dtype=torch.int64)
    token_slices = []
    for pid in pids.tolist():
        start = int(offsets[pid])
        n = int(lengths[pid])
        if n > 0:
            token_slices.append(torch.arange(start, start + n))
    if not token_slices:
        return centroids.new_zeros((0, int(dim)), dtype=torch.float32)
    token_ids = torch.cat(token_slices)

    codes_ = codes[token_ids].long()
    residuals_ = binary_residuals[token_ids].long()
    centroids_ = centroids[codes_].to(torch.float32)
    residuals_ = reversed_bit_map[residuals_].long()
    residuals_ = bucket_weight_combinations[residuals_].long()
    residuals_ = residuals_.reshape(residuals_.shape[0], -1)
    residuals_ = bucket_weights[residuals_].to(torch.float32)
    centroids_.add_(residuals_)
    return centroids_


def _install_python_cpu_kernels(cls, kernels: dict[str, Any]) -> None:
    for name, fn in kernels.items():
        setattr(cls, name, staticmethod(fn))
    cls.loaded_extensions = True


def _ensure_colbert_cpu_extensions() -> None:
    """
    Stanford ColBERT CPU kernels are POSIX C++ (pthread + cl.exe).

    On Windows they do not compile. Bind equivalent PyTorch fallbacks so
    indexing and search can run without Visual Studio Build Tools.
    """
    import sys

    try:
        from colbert.modeling.colbert import ColBERT
        from colbert.search.index_storage import IndexScorer
        from colbert.search.strided_tensor import StridedTensor
    except ImportError:
        return

    if getattr(ColBERT, "_ENGRAMME_CPU_EXT_PATCH", False):
        return

    def wrap(cls, kernels: dict[str, Any], original):
        def try_load_torch_extensions(inner_cls, use_gpu):
            if hasattr(inner_cls, "loaded_extensions") or use_gpu:
                return
            if sys.platform.startswith("win"):
                print(
                    "Using Python ColBERT CPU kernels "
                    "(Windows cannot compile Stanford ColBERT C++ extensions)."
                )
                _install_python_cpu_kernels(inner_cls, kernels)
                return
            try:
                original(inner_cls, use_gpu)
            except Exception as exc:  # pragma: no cover - compile failures
                print(
                    f"ColBERT C++ extensions failed ({exc}); "
                    "using Python CPU kernels."
                )
                _install_python_cpu_kernels(inner_cls, kernels)

        cls.try_load_torch_extensions = classmethod(try_load_torch_extensions)

    wrap(
        ColBERT,
        {"segmented_maxsim": _segmented_maxsim_py},
        ColBERT.try_load_torch_extensions.__func__,
    )
    wrap(
        StridedTensor,
        {"segmented_lookup": _segmented_lookup_py},
        StridedTensor.try_load_torch_extensions.__func__,
    )
    wrap(
        IndexScorer,
        {
            "filter_pids": _filter_pids_py,
            "decompress_residuals": _decompress_residuals_py,
        },
        IndexScorer.try_load_torch_extensions.__func__,
    )
    ColBERT._ENGRAMME_CPU_EXT_PATCH = True


def _ensure_langchain_retrievers_shim() -> None:
    """
    RAGatouille 0.0.9 imports langchain.retrievers.*, removed in LangChain v1.

    Map that legacy path to langchain_core / langchain-classic so RAGatouille loads.
    """
    import sys
    import types

    if "langchain.retrievers.document_compressors.base" in sys.modules:
        return

    try:
        from langchain_core.documents import BaseDocumentCompressor
    except ImportError:
        try:
            from langchain_classic.retrievers.document_compressors.base import (
                BaseDocumentCompressor,
            )
        except ImportError:
            return

    base_mod = types.ModuleType("langchain.retrievers.document_compressors.base")
    base_mod.BaseDocumentCompressor = BaseDocumentCompressor
    dc_mod = types.ModuleType("langchain.retrievers.document_compressors")
    dc_mod.base = base_mod
    ret_mod = types.ModuleType("langchain.retrievers")
    ret_mod.document_compressors = dc_mod
    sys.modules["langchain.retrievers.document_compressors.base"] = base_mod
    sys.modules["langchain.retrievers.document_compressors"] = dc_mod
    sys.modules["langchain.retrievers"] = ret_mod


def _try_load_rag_pretrained_model():
    global _RAGATOUILLE_IMPORT_ERROR
    _ensure_langchain_retrievers_shim()
    _ensure_colbert_transformers_compat()
    try:
        from ragatouille import RAGPretrainedModel

        _ensure_colbert_cpu_extensions()
        _RAGATOUILLE_IMPORT_ERROR = None
        return RAGPretrainedModel
    except ImportError as exc:
        _RAGATOUILLE_IMPORT_ERROR = str(exc)
        return None


def ragatouille_available() -> bool:
    return _try_load_rag_pretrained_model() is not None


def ragatouille_import_error() -> str | None:
    if ragatouille_available():
        return None
    return _RAGATOUILLE_IMPORT_ERROR


def colbert_setup_hint() -> str:
    return (
        "ColBERT needs a working RAGatouille import:\n"
        "  pip install ragatouille\n"
        "If LangChain v1 breaks RAGatouille, also: pip install langchain-classic\n"
        "If loading fails with all_tied_weights_keys, use this repo's shim or pin "
        "transformers<5.\n"
        "On Windows, this repo uses Python ColBERT CPU kernels (no MSVC/cl.exe)."
    )


def _load_ragatouille():
    cls = _try_load_rag_pretrained_model()
    if cls is not None:
        return cls
    detail = _RAGATOUILLE_IMPORT_ERROR or "unknown import error"
    raise RuntimeError(
        f"ColBERT/RAGatouille failed to import ({detail}).\n{colbert_setup_hint()}"
    )


def build_colbert_index(
    texts: list[str],
    record_ids: list[str],
    cache_dir: Path,
    *,
    corpus_index_path: Path,
    model_name: str = DEFAULT_COLBERT_MODEL,
    force: bool = False,
) -> Any:
    """
    Build or load a PLAID ColBERT index over corpus metadata texts.

    Returns a RAGPretrainedModel with an active index.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "colbert_manifest.json"
    corpus_sha = sha256_file(corpus_index_path)

    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = Path(manifest.get("index_path", ""))
        if (
            index_path.exists()
            and manifest.get("corpus_index_sha256") == corpus_sha
            and manifest.get("model_name") == model_name
            and manifest.get("num_documents") == len(texts)
        ):
            RAGPretrainedModel = _load_ragatouille()
            print(f"Loading ColBERT index from {index_path} ...")
            return RAGPretrainedModel.from_index(str(index_path))

    RAGPretrainedModel = _load_ragatouille()
    print(
        f"Building ColBERT index ({model_name}) over {len(texts):,} docs ..."
    )
    rag = RAGPretrainedModel.from_pretrained(model_name)
    built_path = rag.index(
        collection=texts,
        document_ids=record_ids,
        index_name=DEFAULT_INDEX_NAME,
        overwrite_index="force" if force else True,
        split_documents=False,
        max_document_length=320,
        bsize=16,
    )

    manifest = {
        "model_name": model_name,
        "corpus_index_sha256": corpus_sha,
        "num_documents": len(texts),
        "index_path": str(built_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RAGPretrainedModel.from_index(str(built_path))


class ColBERTRetriever:
    def __init__(self, rag_model: Any):
        self._rag = rag_model

    def search(self, query: str, k: int) -> tuple[list[str], float]:
        started = time.perf_counter()
        results = self._rag.search(query, k=k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        ids: list[str] = []
        seen: set[str] = set()
        for row in results:
            doc_id = row.get("document_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            ids.append(doc_id)
            if len(ids) >= k:
                break

        return ids, elapsed_ms
