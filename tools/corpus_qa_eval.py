"""Run corpus Q&A evaluation (retrieval + Codex answering)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai_codex import Codex

from tools.corpus_qa import (
    DEFAULT_QA_COUNT,
    DEFAULT_RETRIEVAL_BENCHMARK_N,
    aggregate_corpus_qa_results,
    evidence_lines_for_question,
    load_questions,
    score_count_answer,
)
from tools.codex_corpus_qa import CORPUS_QA_MODEL, run_corpus_qa_one
from tools.dense_retrieval import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    aggregate_retrieval_metrics,
    recall_at_k,
)
from tools.colbert_retrieval import DEFAULT_COLBERT_MODEL, ragatouille_available
from tools.retrieval_cascade import (
    DEFAULT_SPLADE_MODEL,
    DEFAULT_SPLADE_MODEL_LITE,
    RETRIEVAL_METHODS,
    benchmark_retrieval_methods,
    build_pipeline,
    retrieve_for_qa,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evidence_lines_for_ids(index_path: Path, record_ids: list[str]) -> list[str]:
    id_set = set(record_ids)
    by_id: dict[str, str] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["record_id"]
            if rid not in id_set:
                continue
            senders = ", ".join(row.get("senders") or []) or "?"
            recipients = ", ".join(row.get("recipients") or []) or "?"
            date = row.get("date_utc") or "?"
            subject = (row.get("subject") or "").replace("\n", " ")[:120]
            by_id[rid] = (
                f"- id={rid} | date={date} | "
                f"from={senders} | to={recipients} | subject={subject}"
            )
    return [by_id[rid] for rid in record_ids if rid in by_id]


def run_corpus_qa_evaluation(
    *,
    index_path: Path,
    questions_path: Path,
    output_dir: Path,
    dense_cache_dir: Path,
    model: str = CORPUS_QA_MODEL,
    retrieval_k: int = DEFAULT_TOP_K,
    use_oracle_evidence: bool = False,
    limit: int | None = None,
    rebuild_dense: bool = False,
    embed_model: str = DEFAULT_EMBED_MODEL,
    splade_model: str = DEFAULT_SPLADE_MODEL,
    colbert_model: str = DEFAULT_COLBERT_MODEL,
    retrieval_method: str = "cascade",
    retrieval_benchmark_n: int = DEFAULT_RETRIEVAL_BENCHMARK_N,
    rebuild_colbert: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "corpus_qa_predictions.jsonl"
    metrics_path = output_dir / "corpus_qa_metrics.json"
    retrieval_metrics_path = output_dir / "corpus_qa_retrieval.json"
    benchmark_path = output_dir / "retrieval_benchmark.json"

    all_questions = load_questions(questions_path)
    codex_questions = all_questions if limit is None else all_questions[:limit]

    pipeline = None
    if not use_oracle_evidence:
        needed_methods = {retrieval_method}
        if retrieval_benchmark_n > 0:
            needed_methods.update(RETRIEVAL_METHODS.keys())
        if (
            "colbert" in needed_methods
            and retrieval_method != "colbert"
            and not ragatouille_available()
        ):
            needed_methods.discard("colbert")
        pipeline = build_pipeline(
            index_path,
            dense_cache_dir,
            embed_model=embed_model,
            force_dense=rebuild_dense,
            force_colbert=rebuild_colbert,
            splade_model=splade_model,
            colbert_model=colbert_model,
            methods=list(needed_methods),
        )

    benchmark_summary: dict[str, Any] | None = None
    if (
        pipeline is not None
        and retrieval_benchmark_n > 0
        and not use_oracle_evidence
    ):
        bench_qs = all_questions[:retrieval_benchmark_n]
        print(
            f"\n=== RETRIEVAL BENCHMARK ({len(bench_qs)} questions, k={retrieval_k}) ==="
        )
        benchmark_summary = benchmark_retrieval_methods(
            pipeline,
            bench_qs,
            k=retrieval_k,
            methods=[m for m in RETRIEVAL_METHODS if m in needed_methods],
        )
        benchmark_summary["created_at_utc"] = utc_now()
        benchmark_path.write_text(
            json.dumps(benchmark_summary, indent=2),
            encoding="utf-8",
        )
        dense_stats = benchmark_summary["methods"].get("dense", {})
        lat = dense_stats.get("latency_ms", {})
        print(
            "Dense-only latency (ms): "
            f"mean={lat.get('mean_total')}, p50={lat.get('p50_total')}, "
            f"p95={lat.get('p95_total')}"
        )
        cascade_stats = benchmark_summary["methods"].get("cascade", {})
        clat = cascade_stats.get("latency_ms", {})
        print(
            "Cascade latency (ms): "
            f"mean={clat.get('mean_total')}, p50={clat.get('p50_total')}"
        )
        print(f"Retrieval benchmark: {benchmark_path}")

    if retrieval_method not in RETRIEVAL_METHODS and not use_oracle_evidence:
        raise ValueError(
            f"Unknown retrieval method {retrieval_method!r}. "
            f"Choose from: {', '.join(RETRIEVAL_METHODS)}"
        )

    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        completed[row["question_id"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue

    results: list[dict[str, Any]] = []

    with Codex() as codex:
        for i, question in enumerate(codex_questions, start=1):
            qid = question["question_id"]
            gold_ids = set(question.get("relevant_record_ids") or [])
            gold_count = int(question["answer_count"])

            if qid in completed:
                print(f"[corpus QA {i}/{len(codex_questions)}] Reusing {qid}")
                results.append(completed[qid])
                continue

            print(f"[corpus QA {i}/{len(codex_questions)}] {question['question'][:80]}...")

            retrieval_timings: dict[str, float] = {}
            if use_oracle_evidence:
                retrieved_ids = question.get("relevant_record_ids") or []
            else:
                assert pipeline is not None
                retrieval = retrieve_for_qa(
                    pipeline,
                    question["question"],
                    retrieval_k,
                    retrieval_method,
                )
                retrieved_ids = retrieval.record_ids
                retrieval_timings = retrieval.timings_ms

            recall = recall_at_k(retrieved_ids, gold_ids)
            visible_gold = len(set(retrieved_ids) & gold_ids)

            if use_oracle_evidence:
                evidence_lines = evidence_lines_for_question(question, index_path)
            else:
                evidence_lines = evidence_lines_for_ids(index_path, retrieved_ids)

            row: dict[str, Any] = {
                "question_id": qid,
                "template": question.get("template"),
                "question": question["question"],
                "params": question.get("params"),
                "answer_count_gold": gold_count,
                "answer_count_visible_gold": visible_gold,
                "retrieval_k": retrieval_k,
                "retrieval_method": retrieval_method
                if not use_oracle_evidence
                else "oracle",
                "retrieved_record_ids": retrieved_ids,
                "retrieval_recall_at_k": recall,
                "retrieval_timings_ms": retrieval_timings,
                "evidence_count": len(evidence_lines),
                "model": model,
            }

            if not evidence_lines:
                row["status"] = "no_evidence"
                row["count_correct"] = False
            else:
                try:
                    codex_result = run_corpus_qa_one(
                        codex,
                        question,
                        evidence_lines,
                        model=model,
                    )
                    row["thread_id"] = codex_result.get("thread_id")
                    row["usage"] = codex_result.get("usage", {})
                    row["status"] = codex_result.get("status", "error")

                    if codex_result.get("status") == "ok":
                        pred_count = codex_result["prediction"]["count"]
                        row["prediction"] = codex_result["prediction"]
                        row["count_correct_full"] = score_count_answer(
                            pred_count, gold_count
                        )
                        row["count_correct_visible"] = score_count_answer(
                            pred_count, visible_gold
                        )
                        row["count_correct"] = row["count_correct_full"]
                    else:
                        row["error"] = codex_result.get("parse_error", "unknown")
                        row["count_correct"] = False
                        row["count_correct_full"] = False
                        row["count_correct_visible"] = False
                except Exception as exc:
                    row["status"] = "error"
                    row["error_type"] = type(exc).__name__
                    row["error"] = str(exc)[:1000]
                    row["count_correct"] = False

            results.append(row)
            write_jsonl(results_path, results)

    answer_metrics = aggregate_corpus_qa_results(results)
    retrieval_summary = aggregate_retrieval_metrics(results)

    qa_latencies = [
        r["retrieval_timings_ms"]["total_ms"]
        for r in results
        if r.get("retrieval_timings_ms", {}).get("total_ms") is not None
    ]
    if qa_latencies:
        retrieval_summary["qa_run_mean_retrieval_ms"] = sum(qa_latencies) / len(
            qa_latencies
        )

    full_correct = sum(1 for r in results if r.get("count_correct_full"))
    visible_correct = sum(1 for r in results if r.get("count_correct_visible"))
    scored = sum(1 for r in results if r.get("count_correct_full") is not None)

    metrics = {
        **answer_metrics,
        "retrieval": retrieval_summary,
        "count_accuracy_full_corpus": full_correct / scored if scored else 0.0,
        "count_accuracy_visible_in_top_k": visible_correct / scored
        if scored
        else 0.0,
        "corpus_qa_model": model,
        "retrieval_mode": "oracle" if use_oracle_evidence else retrieval_method,
        "dense_embed_model": embed_model,
        "splade_model": splade_model,
        "colbert_model": colbert_model,
        "retrieval_benchmark": benchmark_summary,
        "questions_path": str(questions_path),
        "index_path": str(index_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    retrieval_metrics_path.write_text(
        json.dumps(retrieval_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== CORPUS Q&A SUMMARY ===")
    print(f"Model: {model}")
    print(f"Questions: {len(results)}")
    print(f"Retrieval method for Codex: {metrics['retrieval_mode']}")
    print(
        "Mean retrieval recall@k: "
        f"{retrieval_summary.get('mean_recall_at_k')}"
    )
    if retrieval_summary.get("qa_run_mean_retrieval_ms") is not None:
        print(
            "Mean retrieval latency this run (ms): "
            f"{retrieval_summary['qa_run_mean_retrieval_ms']:.2f}"
        )
    print(
        "Count exact match (full corpus gold): "
        f"{metrics['count_accuracy_full_corpus']:.1%}"
    )
    print(
        "Count exact match (gold visible in top-k): "
        f"{metrics['count_accuracy_visible_in_top_k']:.1%}"
    )
    print(f"Predictions: {results_path}")
    print(f"Metrics:     {metrics_path}")

    return metrics


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate corpus Q&A with retrieval cascade."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data/engramme_test"))
    parser.add_argument("--corpus-sample-n", type=int, default=10_000)
    parser.add_argument("--corpus-qa-n", type=int, default=DEFAULT_QA_COUNT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/engramme_test"))
    parser.add_argument("--model", default=CORPUS_QA_MODEL)
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--retrieval-method", default="cascade")
    parser.add_argument("--retrieval-benchmark-n", type=int, default=DEFAULT_RETRIEVAL_BENCHMARK_N)
    parser.add_argument("--oracle-evidence", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-dense", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    index_path = data_dir / f"corpus_index_{args.corpus_sample_n}.jsonl"
    qa_path = data_dir / f"synthetic_qa_{args.corpus_qa_n}.jsonl"

    if not index_path.exists() or not qa_path.exists():
        print(
            "Missing corpus index or synthetic QA. Run eval_enron.py first, "
            "or: python -m tools.corpus_qa",
            file=sys.stderr,
        )
        return 2

    try:
        run_corpus_qa_evaluation(
            index_path=index_path,
            questions_path=qa_path,
            output_dir=args.output_dir.resolve(),
            dense_cache_dir=data_dir / f"dense_cache_{args.corpus_sample_n}",
            model=args.model,
            retrieval_k=args.retrieval_k,
            use_oracle_evidence=args.oracle_evidence,
            limit=args.limit,
            rebuild_dense=args.rebuild_dense,
            retrieval_method=args.retrieval_method,
            retrieval_benchmark_n=args.retrieval_benchmark_n,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
