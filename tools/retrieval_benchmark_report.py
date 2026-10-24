"""Run retrieval benchmarks and render a README comparison table."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tools.corpus_qa import DEFAULT_RETRIEVAL_BENCHMARK_N, load_questions
from tools.dense_retrieval import DEFAULT_EMBED_MODEL, DEFAULT_TOP_K
from tools.colbert_retrieval import (
    DEFAULT_COLBERT_MODEL,
    colbert_setup_hint,
    ragatouille_available,
    ragatouille_import_error,
)
from tools.retrieval_cascade import (
    DEFAULT_SPLADE_MODEL,
    benchmark_retrieval_methods,
    build_pipeline,
)

README_MARKER_START = "<!-- RETRIEVAL_BENCHMARK_START -->"
README_MARKER_END = "<!-- RETRIEVAL_BENCHMARK_END -->"

DEFAULT_METHODS = ("dense", "bm25", "bm25_splade", "colbert", "cascade")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.1f}%"


def _ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def _bold_if(value: str, best: bool) -> str:
    return f"**{value}**" if best else value


def render_benchmark_table(
    summary: dict,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> str:
    """Markdown table; bold best mean recall@k and best (lowest) mean latency."""
    rows: list[tuple[str, float | None, float | None]] = []
    for name in methods:
        stats = summary.get("methods", {}).get(name, {})
        if stats.get("error"):
            rows.append((name, None, None))
            continue
        recall = stats.get("mean_recall_at_k")
        lat = (stats.get("latency_ms") or {}).get("mean_total")
        rows.append((name, recall, lat))

    valid_recalls = [r for _, r, _ in rows if r is not None]
    valid_lats = [lat for _, _, lat in rows if lat is not None]
    best_recall = max(valid_recalls) if valid_recalls else None
    best_lat = min(valid_lats) if valid_lats else None

    k = summary.get("k", "?")
    n = summary.get("questions", "?")
    created = summary.get("created_at_utc", "")

    lines = [
        f"Retrieval benchmark on **{n}** synthetic questions, recall@{k} "
        f"(corpus slice 10K). Updated: `{created}`.",
        "",
        "| Method | Mean recall@k | Mean latency (ms) |",
        "|--------|---------------|-------------------|",
    ]

    for name, recall, lat in rows:
        stats = summary.get("methods", {}).get(name, {})
        if stats.get("error"):
            err = stats["error"].replace("|", "/")[:60]
            lines.append(f"| `{name}` | _error_ | _{err}_ |")
            continue
        recall_s = _bold_if(_pct(recall), recall is not None and recall == best_recall)
        lat_s = _bold_if(_ms(lat), lat is not None and lat == best_lat)
        lines.append(f"| `{name}` | {recall_s} | {lat_s} |")

    lines.extend([
        "",
        "Bold = best in column (highest recall, lowest latency).",
        "",
        "Regenerate: `python -m tools.retrieval_benchmark_report --update-readme`",
    ])
    return "\n".join(lines)


def patch_readme(readme_path: Path, table_block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    block = f"{README_MARKER_START}\n{table_block}\n{README_MARKER_END}"

    if README_MARKER_START in text and README_MARKER_END in text:
        pattern = re.compile(
            re.escape(README_MARKER_START) + r".*?" + re.escape(README_MARKER_END),
            re.DOTALL,
        )
        text = pattern.sub(block, text)
    else:
        text = text.rstrip() + "\n\n## Retrieval benchmark\n\n" + block + "\n"

    readme_path.write_text(text, encoding="utf-8")


def run_benchmark(
    *,
    data_dir: Path,
    corpus_sample_n: int,
    qa_count: int,
    benchmark_n: int,
    k: int,
    methods: list[str],
    embed_model: str,
    splade_model: str,
    colbert_model: str,
    rebuild_dense: bool,
    rebuild_colbert: bool,
    rebuild_splade_cache: bool,
) -> dict:
    index_path = data_dir / f"corpus_index_{corpus_sample_n}.jsonl"
    qa_path = data_dir / f"synthetic_qa_{qa_count}.jsonl"
    if not index_path.exists() or not qa_path.exists():
        raise FileNotFoundError(
            f"Need {index_path} and {qa_path}. Run: python eval_enron.py "
            "(or python -m tools.corpus_qa) once to build corpus + questions."
        )

    questions = load_questions(qa_path)[:benchmark_n]
    print(
        f"Building retrieval pipeline for {methods} "
        f"({len(questions)} benchmark questions, k={k}) ..."
    )
    pipeline = build_pipeline(
        index_path,
        data_dir / f"dense_cache_{corpus_sample_n}",
        embed_model=embed_model,
        force_dense=rebuild_dense,
        splade_model=splade_model,
        colbert_model=colbert_model,
        force_colbert=rebuild_colbert,
        force_splade_cache=rebuild_splade_cache,
        methods=methods,
    )

    summary = benchmark_retrieval_methods(
        pipeline,
        questions,
        k=k,
        methods=methods,
    )
    summary["created_at_utc"] = utc_now()
    summary["dense_embed_model"] = embed_model
    summary["splade_model"] = splade_model
    summary["colbert_model"] = colbert_model
    return summary


def merge_benchmark_files(paths: list[Path]) -> dict:
    """Combine partial runs (one method per JSON file)."""
    merged: dict = {"methods": {}}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        merged.setdefault("k", data.get("k"))
        merged.setdefault("questions", data.get("questions"))
        merged["methods"].update(data.get("methods", {}))
        for key in ("dense_embed_model", "splade_model", "created_at_utc"):
            if data.get(key) and not merged.get(key):
                merged[key] = data[key]
    merged["created_at_utc"] = merged.get("created_at_utc") or utc_now()
    merged["merged_from"] = [str(p) for p in paths]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark dense vs bm25_splade vs cascade; print README table."
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Merge partial benchmark JSON files (after parallel runs).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data/engramme_test"))
    parser.add_argument("--corpus-sample-n", type=int, default=10_000)
    parser.add_argument("--corpus-qa-n", type=int, default=1000)
    parser.add_argument(
        "--retrieval-benchmark-n",
        type=int,
        default=DEFAULT_RETRIEVAL_BENCHMARK_N,
    )
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated: dense,bm25,bm25_splade,colbert,cascade "
            "(one method for parallel runs)."
        ),
    )
    parser.add_argument("--dense-embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--splade-model", default=DEFAULT_SPLADE_MODEL)
    parser.add_argument("--colbert-model", default=DEFAULT_COLBERT_MODEL)
    parser.add_argument("--rebuild-dense", action="store_true")
    parser.add_argument("--rebuild-colbert", action="store_true")
    parser.add_argument(
        "--rebuild-splade-cache",
        action="store_true",
        help="Force re-encode of cached SPLADE document vectors.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/engramme_test/retrieval_benchmark.json"),
    )
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()

    if args.merge:
        paths = [p.resolve() for p in args.merge]
        summary = merge_benchmark_files(paths)
        methods = tuple(summary.get("methods", {}).keys())
        table = render_benchmark_table(summary, methods)
        print(table)
        if args.update_readme:
            patch_readme(args.readme.resolve(), table)
            print(f"\nUpdated {args.readme}")
        return 0

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "colbert" in methods and not ragatouille_available():
        if set(methods) == {"colbert"}:
            detail = ragatouille_import_error()
            print("ERROR: ColBERT/RAGatouille is not usable.", file=sys.stderr)
            if detail:
                print(f"  {detail}", file=sys.stderr)
            print(colbert_setup_hint(), file=sys.stderr)
            return 1
        print(
            "Note: skipping colbert (install ragatouille for ColBERT benchmarks).",
            file=sys.stderr,
        )
        methods = [m for m in methods if m != "colbert"]

    try:
        summary = run_benchmark(
            data_dir=args.data_dir.resolve(),
            corpus_sample_n=args.corpus_sample_n,
            qa_count=args.corpus_qa_n,
            benchmark_n=args.retrieval_benchmark_n,
            k=args.retrieval_k,
            methods=methods,
            embed_model=args.dense_embed_model,
            splade_model=args.splade_model,
            colbert_model=args.colbert_model,
            rebuild_dense=args.rebuild_dense,
            rebuild_colbert=args.rebuild_colbert,
            rebuild_splade_cache=args.rebuild_splade_cache,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = render_benchmark_table(summary, tuple(methods))
    print(table)

    if args.update_readme:
        patch_readme(args.readme.resolve(), table)
        print(f"\nUpdated {args.readme}")

    print(f"\nJSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
