"""Run corpus Q&A evaluation (retrieval + Codex answering)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from openai_codex import Codex

from tools.corpus_qa import (
    DEFAULT_QA_COUNT,
    aggregate_corpus_qa_results,
    evidence_lines_for_question,
    load_questions,
    score_count_answer,
)
from tools.codex_corpus_qa import CORPUS_QA_MODEL, run_corpus_qa_one
from tools.dense_retrieval import (
    DEFAULT_TOP_K,
    aggregate_retrieval_metrics,
    build_dense_index,
    recall_at_k,
    retrieve_top_k,
)


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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "corpus_qa_predictions.jsonl"
    metrics_path = output_dir / "corpus_qa_metrics.json"
    retrieval_metrics_path = output_dir / "corpus_qa_retrieval.json"

    questions = load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    dense_index = build_dense_index(
        index_path,
        dense_cache_dir,
        force=rebuild_dense,
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
        for i, question in enumerate(questions, start=1):
            qid = question["question_id"]
            gold_ids = set(question.get("relevant_record_ids") or [])
            gold_count = int(question["answer_count"])

            if qid in completed:
                print(f"[corpus QA {i}/{len(questions)}] Reusing {qid}")
                results.append(completed[qid])
                continue

            print(f"[corpus QA {i}/{len(questions)}] {question['question'][:80]}...")

            if use_oracle_evidence:
                retrieved_ids = question.get("relevant_record_ids") or []
            else:
                retrieved_ids = retrieve_top_k(
                    question["question"],
                    dense_index,
                    k=retrieval_k,
                )

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
                "retrieved_record_ids": retrieved_ids,
                "retrieval_recall_at_k": recall,
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
        "retrieval_mode": "oracle" if use_oracle_evidence else "dense",
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
    print(
        "Mean retrieval recall@k: "
        f"{retrieval_summary.get('mean_recall_at_k')}"
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

    parser = argparse.ArgumentParser(description="Evaluate corpus Q&A with dense retrieval.")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/engramme_test"))
    parser.add_argument("--corpus-sample-n", type=int, default=10_000)
    parser.add_argument("--corpus-qa-n", type=int, default=DEFAULT_QA_COUNT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/engramme_test"))
    parser.add_argument("--model", default=CORPUS_QA_MODEL)
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--oracle-evidence", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-dense", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    index_path = data_dir / f"corpus_index_{args.corpus_sample_n}.jsonl"
    qa_path = data_dir / f"synthetic_qa_{args.corpus_qa_n}.jsonl"

    if not index_path.exists() or not qa_path.exists():
        print(
            "Missing corpus index or synthetic QA. Run eval_enron.py with "
            "corpus QA enabled first, or: python -m tools.corpus_qa",
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
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
