"""Synthetic corpus Q&A with algorithmic ground truth for retrieval eval."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys

csv.field_size_limit(min(sys.maxsize, 100_000_000))
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tools.enron_parse import parse_email_metadata
from tools.download_tools import resolve_dataset_csv

DEFAULT_QA_COUNT = 1000
DEFAULT_QA_SEED = 42
DEFAULT_CORPUS_SAMPLE_N = 10_000
DEFAULT_CORPUS_SAMPLE_SEED = 42
MAX_CODEX_EVIDENCE = 200
INDEX_PROGRESS_EVERY = 50_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_date_human(iso_date: str) -> str:
    if not iso_date:
        return "unknown date"
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return iso_date[:10]


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    raw_sha256: str
    senders: tuple[str, ...]
    recipients: tuple[str, ...]
    date_utc: str
    subject: str


def build_corpus_index(csv_path: Path, index_path: Path) -> str:
    """Stream the CSV and write corpus_index.jsonl. Returns CSV sha256."""
    csv_path = csv_path.resolve()
    csv_digest = sha256_file(csv_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    indexed = 0

    print(f"Building corpus index from {csv_path} ...")

    with csv_path.open(
        "r", encoding="utf-8-sig", newline="", errors="replace"
    ) as src, index_path.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        fields = {
            name.strip().casefold(): name
            for name in reader.fieldnames
            if name
        }
        message_col = fields.get("message")
        if not message_col:
            raise ValueError(
                f"Expected a 'message' column; found {reader.fieldnames}"
            )
        id_col = fields.get("file")

        for row in reader:
            raw = row.get(message_col) or ""
            if not raw.strip():
                continue

            total += 1
            record_id = (row.get(id_col, "").strip() if id_col else "") or (
                f"row_{total}"
            )
            meta = parse_email_metadata(raw)
            entry = {
                "record_id": record_id,
                "raw_sha256": sha256_text(raw),
                "senders": meta["sender"],
                "recipients": meta["recipients"],
                "date_utc": meta["date_utc"],
                "subject": meta["subject"],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            indexed += 1

            if total % INDEX_PROGRESS_EVERY == 0:
                print(f"  indexed {indexed:,} emails (scanned {total:,})")

    print(f"Corpus index complete: {indexed:,} emails -> {index_path}")
    return csv_digest


def build_corpus_index_from_sample(
    sample_path: Path,
    index_path: Path,
) -> str:
    """Parse metadata for each row in a JSONL email sample."""
    sample_path = sample_path.resolve()
    sample_digest = sha256_file(sample_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building corpus index from sample {sample_path} ...")
    indexed = 0

    with sample_path.open(encoding="utf-8") as src, index_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            raw = item.get("raw_message") or ""
            if not raw.strip():
                continue

            meta = parse_email_metadata(raw)
            entry = {
                "record_id": item["record_id"],
                "raw_sha256": item.get("raw_sha256") or sha256_text(raw),
                "senders": meta["sender"],
                "recipients": meta["recipients"],
                "date_utc": meta["date_utc"],
                "subject": meta["subject"],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            indexed += 1

    print(f"Corpus index complete: {indexed:,} emails -> {index_path}")
    return sample_digest


def load_corpus_index(index_path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                CorpusRecord(
                    record_id=row["record_id"],
                    raw_sha256=row["raw_sha256"],
                    senders=tuple(row.get("senders") or []),
                    recipients=tuple(row.get("recipients") or []),
                    date_utc=row.get("date_utc") or "",
                    subject=row.get("subject") or "",
                )
            )
    return records


@dataclass
class InvertedIndex:
    records: list[CorpusRecord]
    by_sender: dict[str, list[int]]
    by_recipient: dict[str, list[int]]
    dated_indices: list[int]


def build_inverted_index(records: list[CorpusRecord]) -> InvertedIndex:
    by_sender: dict[str, list[int]] = {}
    by_recipient: dict[str, list[int]] = {}
    dated_indices: list[int] = []

    for i, rec in enumerate(records):
        for sender in rec.senders:
            by_sender.setdefault(sender, []).append(i)
        for recipient in rec.recipients:
            by_recipient.setdefault(recipient, []).append(i)
        if rec.date_utc:
            dated_indices.append(i)

    dated_indices.sort(key=lambda i: records[i].date_utc or "")
    return InvertedIndex(
        records=records,
        by_sender=by_sender,
        by_recipient=by_recipient,
        dated_indices=dated_indices,
    )


def _indices_before(index: InvertedIndex, date_iso: str) -> set[int]:
    dated = index.dated_indices
    records = index.records
    lo, hi = 0, len(dated)
    while lo < hi:
        mid = (lo + hi) // 2
        if (records[dated[mid]].date_utc or "") < date_iso:
            lo = mid + 1
        else:
            hi = mid
    return set(dated[:lo])


def _indices_after(index: InvertedIndex, date_iso: str) -> set[int]:
    dated = index.dated_indices
    records = index.records
    lo, hi = 0, len(dated)
    while lo < hi:
        mid = (lo + hi) // 2
        if (records[dated[mid]].date_utc or "") <= date_iso:
            lo = mid + 1
        else:
            hi = mid
    return set(dated[lo:])


def _set_to_ids(index: InvertedIndex, indices: set[int]) -> list[str]:
    return [index.records[i].record_id for i in sorted(indices)]


def _set_to_sha(index: InvertedIndex, indices: set[int]) -> list[str]:
    return [index.records[i].raw_sha256 for i in sorted(indices)]


FilterFn = Callable[[InvertedIndex, dict[str, str], random.Random], dict[str, Any] | None]


def _try_sent_by(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    senders = [s for s, ids in index.by_sender.items() if len(ids) >= 1]
    if not senders:
        return None
    sender = rng.choice(senders)
    hits = set(index.by_sender.get(sender, []))
    return {
        "template": "sent_by",
        "params": {"sender": sender},
        "question": f"How many emails were sent by {sender}?",
        "indices": hits,
    }


def _try_before(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    if not index.dated_indices:
        return None
    rec = index.records[rng.choice(index.dated_indices)]
    date_iso = rec.date_utc
    hits = _indices_before(index, date_iso)
    if not hits:
        return None
    return {
        "template": "before",
        "params": {"before": date_iso},
        "question": (
            "How many emails were sent before "
            f"{format_date_human(date_iso)}?"
        ),
        "indices": hits,
    }


def _try_after(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    if not index.dated_indices:
        return None
    rec = index.records[rng.choice(index.dated_indices)]
    date_iso = rec.date_utc
    hits = _indices_after(index, date_iso)
    if not hits:
        return None
    return {
        "template": "after",
        "params": {"after": date_iso},
        "question": (
            f"How many emails were sent after {format_date_human(date_iso)}?"
        ),
        "indices": hits,
    }


def _try_received_by(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    recipients = [r for r, ids in index.by_recipient.items() if len(ids) >= 1]
    if not recipients:
        return None
    recipient = rng.choice(recipients)
    hits = set(index.by_recipient.get(recipient, []))
    return {
        "template": "received_by",
        "params": {"recipient": recipient},
        "question": f"How many emails were received by {recipient}?",
        "indices": hits,
    }


def _try_sent_by_and_before(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    senders = [s for s, ids in index.by_sender.items() if len(ids) >= 2]
    if not senders or not index.dated_indices:
        return None
    sender = rng.choice(senders)
    date_iso = index.records[rng.choice(index.dated_indices)].date_utc
    hits = set(index.by_sender.get(sender, [])) & _indices_before(
        index, date_iso
    )
    if not hits:
        return None
    return {
        "template": "sent_by_and_before",
        "params": {"sender": sender, "before": date_iso},
        "question": (
            f"How many emails were sent by {sender} before "
            f"{format_date_human(date_iso)}?"
        ),
        "indices": hits,
    }


def _try_sent_by_and_after(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    senders = [s for s, ids in index.by_sender.items() if len(ids) >= 2]
    if not senders or not index.dated_indices:
        return None
    sender = rng.choice(senders)
    date_iso = index.records[rng.choice(index.dated_indices)].date_utc
    hits = set(index.by_sender.get(sender, [])) & _indices_after(
        index, date_iso
    )
    if not hits:
        return None
    return {
        "template": "sent_by_and_after",
        "params": {"sender": sender, "after": date_iso},
        "question": (
            f"How many emails were sent by {sender} after "
            f"{format_date_human(date_iso)}?"
        ),
        "indices": hits,
    }


def _try_sent_by_or_received_by(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    addresses = set(index.by_sender) | set(index.by_recipient)
    addresses = [a for a in addresses if "@" in a]
    if len(addresses) < 2:
        return None
    x, y = rng.sample(addresses, 2)
    hits = set(index.by_sender.get(x, [])) | set(index.by_recipient.get(y, []))
    if not hits:
        return None
    return {
        "template": "sent_by_or_received_by",
        "params": {"sender": x, "recipient": y},
        "question": (
            f"How many emails were sent by {x} or received by {y}?"
        ),
        "indices": hits,
    }


def _try_received_by_and_before(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    recipients = [r for r, ids in index.by_recipient.items() if len(ids) >= 2]
    if not recipients or not index.dated_indices:
        return None
    recipient = rng.choice(recipients)
    date_iso = index.records[rng.choice(index.dated_indices)].date_utc
    hits = set(index.by_recipient.get(recipient, [])) & _indices_before(
        index, date_iso
    )
    if not hits:
        return None
    return {
        "template": "received_by_and_before",
        "params": {"recipient": recipient, "before": date_iso},
        "question": (
            f"How many emails were received by {recipient} before "
            f"{format_date_human(date_iso)}?"
        ),
        "indices": hits,
    }


def _try_sent_by_and_received_by(
    index: InvertedIndex, _: dict[str, str], rng: random.Random
) -> dict[str, Any] | None:
    candidates: list[tuple[str, str]] = []
    for i, rec in enumerate(index.records):
        if not rec.senders or not rec.recipients:
            continue
        for sender in rec.senders:
            for recipient in rec.recipients:
                if sender != recipient:
                    candidates.append((sender, recipient))

    if not candidates:
        return None

    sender, recipient = rng.choice(candidates)
    hits = {
        i
        for i in index.by_sender.get(sender, [])
        if recipient in index.records[i].recipients
    }
    if not hits:
        return None
    return {
        "template": "sent_by_and_received_by",
        "params": {"sender": sender, "recipient": recipient},
        "question": (
            f"How many emails were sent by {sender} and received by "
            f"{recipient}?"
        ),
        "indices": hits,
    }


TEMPLATE_GENERATORS: list[FilterFn] = [
    _try_sent_by,
    _try_before,
    _try_after,
    _try_received_by,
    _try_sent_by_and_before,
    _try_sent_by_and_after,
    _try_sent_by_or_received_by,
    _try_received_by_and_before,
    _try_sent_by_and_received_by,
]


def generate_synthetic_questions(
    index: InvertedIndex,
    n: int,
    seed: int,
    corpus_index_sha256: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 200
    template_i = 0

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        generator = TEMPLATE_GENERATORS[template_i % len(TEMPLATE_GENERATORS)]
        template_i += 1

        draft = generator(index, {}, rng)
        if draft is None:
            continue

        key = json.dumps(
            {"template": draft["template"], "params": draft["params"]},
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)

        indices: set[int] = draft["indices"]
        answer_count = len(indices)
        qid = f"qa_{len(questions):04d}"

        questions.append({
            "question_id": qid,
            "template": draft["template"],
            "question": draft["question"],
            "params": draft["params"],
            "answer_count": answer_count,
            "relevant_record_ids": _set_to_ids(index, indices),
            "relevant_sha256": _set_to_sha(index, indices),
            "corpus_index_sha256": corpus_index_sha256,
            "codex_eval_eligible": answer_count <= MAX_CODEX_EVIDENCE,
            "created_at_utc": utc_now(),
        })

    if len(questions) < n:
        raise RuntimeError(
            f"Only generated {len(questions)}/{n} unique questions "
            f"after {attempts} attempts."
        )

    return questions


def write_questions(path: Path, questions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in questions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_corpus_sample(
    data_dir: Path,
    csv_path: Path,
    corpus_sample_n: int,
    corpus_seed: int,
    force_regenerate: bool = False,
) -> Path:
    """Reservoir-sample emails into corpus_sample_{n}.jsonl."""
    sample_path = data_dir / f"corpus_sample_{corpus_sample_n}.jsonl"
    sample_manifest_path = data_dir / f"corpus_sample_{corpus_sample_n}_manifest.json"
    csv_digest = sha256_file(csv_path.resolve())

    need_sample = force_regenerate or not sample_path.exists()
    if not need_sample and sample_manifest_path.exists():
        manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
        need_sample = (
            manifest.get("source_csv_sha256") != csv_digest
            or manifest.get("requested_sample_size") != corpus_sample_n
            or manifest.get("seed") != corpus_seed
        )

    if need_sample:
        from tools.enron_sample import reservoir_sample, write_sample_jsonl

        sample, total = reservoir_sample(csv_path, corpus_sample_n, corpus_seed)
        if len(sample) < 1:
            raise RuntimeError("Corpus sample is empty.")
        write_sample_jsonl(sample_path, sample)
        sample_manifest_path.write_text(
            json.dumps(
                {
                    "created_at_utc": utc_now(),
                    "source_csv": str(csv_path),
                    "source_csv_sha256": csv_digest,
                    "requested_sample_size": corpus_sample_n,
                    "actual_sample_size": len(sample),
                    "nonempty_source_rows_scanned": total,
                    "seed": corpus_seed,
                    "sample_file": str(sample_path),
                    "sample_sha256": sha256_file(sample_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Corpus sample saved: {sample_path} ({len(sample):,} emails)")
    else:
        print(f"Reusing corpus sample: {sample_path}")

    return sample_path


def ensure_synthetic_qa_dataset(
    data_dir: Path,
    csv_path: Path | None = None,
    corpus_sample_n: int = DEFAULT_CORPUS_SAMPLE_N,
    corpus_sample_seed: int = DEFAULT_CORPUS_SAMPLE_SEED,
    qa_count: int = DEFAULT_QA_COUNT,
    seed: int = DEFAULT_QA_SEED,
    force_regenerate: bool = False,
) -> tuple[Path, Path]:
    """Build 10K (or N) corpus index + synthetic QA if missing."""
    data_dir = data_dir.resolve()
    raw_dir = data_dir / "raw"
    sample_path = data_dir / f"corpus_sample_{corpus_sample_n}.jsonl"
    index_path = data_dir / f"corpus_index_{corpus_sample_n}.jsonl"
    index_manifest_path = data_dir / f"corpus_index_{corpus_sample_n}_manifest.json"
    qa_path = data_dir / f"synthetic_qa_{qa_count}.jsonl"
    qa_manifest_path = data_dir / f"synthetic_qa_{qa_count}_manifest.json"

    if csv_path is None:
        csv_path = resolve_dataset_csv(data_dir)

    ensure_corpus_sample(
        data_dir,
        csv_path,
        corpus_sample_n,
        corpus_sample_seed,
        force_regenerate=force_regenerate,
    )
    sample_digest = sha256_file(sample_path)

    need_index = force_regenerate or not index_path.exists()
    if not need_index and index_manifest_path.exists():
        manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        need_index = (
            manifest.get("corpus_sample_sha256") != sample_digest
            or manifest.get("corpus_sample_size") != corpus_sample_n
        )

    if need_index:
        build_corpus_index_from_sample(sample_path, index_path)
        index_manifest = {
            "created_at_utc": utc_now(),
            "corpus_sample_file": str(sample_path),
            "corpus_sample_sha256": sample_digest,
            "corpus_sample_size": corpus_sample_n,
            "index_file": str(index_path),
            "index_sha256": sha256_file(index_path),
        }
        index_manifest_path.write_text(
            json.dumps(index_manifest, indent=2),
            encoding="utf-8",
        )
    else:
        print(f"Reusing corpus index: {index_path}")

    index_sha = sha256_file(index_path)

    need_qa = force_regenerate or not qa_path.exists()
    if not need_qa and qa_manifest_path.exists():
        qa_manifest = json.loads(qa_manifest_path.read_text(encoding="utf-8"))
        need_qa = (
            qa_manifest.get("corpus_index_sha256") != index_sha
            or qa_manifest.get("requested_count") != qa_count
            or qa_manifest.get("seed") != seed
            or qa_manifest.get("corpus_sample_size") != corpus_sample_n
        )

    if need_qa:
        print(f"Generating {qa_count} synthetic corpus questions ...")
        records = load_corpus_index(index_path)
        inverted = build_inverted_index(records)
        questions = generate_synthetic_questions(
            inverted, qa_count, seed, index_sha
        )
        write_questions(qa_path, questions)

        by_template = Counter(q["template"] for q in questions)
        qa_manifest = {
            "created_at_utc": utc_now(),
            "seed": seed,
            "requested_count": qa_count,
            "actual_count": len(questions),
            "corpus_sample_size": corpus_sample_n,
            "corpus_sample_seed": corpus_sample_seed,
            "corpus_index_sha256": index_sha,
            "questions_file": str(qa_path),
            "questions_sha256": sha256_file(qa_path),
            "template_counts": dict(sorted(by_template.items())),
            "codex_eval_eligible": sum(
                1 for q in questions if q.get("codex_eval_eligible")
            ),
        }
        qa_manifest_path.write_text(
            json.dumps(qa_manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Synthetic QA saved: {qa_path}")
    else:
        print(f"Reusing synthetic QA: {qa_path}")

    return qa_path, index_path


def evidence_lines_for_question(
    question: dict[str, Any],
    index_path: Path,
) -> list[str]:
    """Metadata lines for oracle Codex context (retrieval upper bound)."""
    id_set = set(question.get("relevant_record_ids") or [])
    if not id_set:
        return []

    lines: list[str] = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["record_id"] not in id_set:
                continue
            senders = ", ".join(row.get("senders") or []) or "?"
            recipients = ", ".join(row.get("recipients") or []) or "?"
            date = row.get("date_utc") or "?"
            subject = (row.get("subject") or "").replace("\n", " ")[:120]
            lines.append(
                f"- id={row['record_id']} | date={date} | "
                f"from={senders} | to={recipients} | subject={subject}"
            )

    lines.sort()
    return lines


def score_count_answer(predicted: Any, gold: int) -> bool:
    try:
        return int(predicted) == gold
    except (TypeError, ValueError):
        return False


def aggregate_corpus_qa_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    scored = [r for r in results if r.get("count_correct") is not None]
    correct = sum(1 for r in scored if r.get("count_correct") is True)
    latencies = [
        r["usage"]["latency_seconds"]
        for r in results
        if r.get("status") == "ok"
        and r.get("usage", {}).get("latency_seconds") is not None
    ]

    by_template: dict[str, dict[str, int]] = {}
    for row in results:
        template = row.get("template", "unknown")
        bucket = by_template.setdefault(
            template, {"total": 0, "correct": 0}
        )
        bucket["total"] += 1
        if row.get("count_correct"):
            bucket["correct"] += 1

    template_accuracy = {
        name: {
            **stats,
            "accuracy": stats["correct"] / stats["total"]
            if stats["total"]
            else 0.0,
        }
        for name, stats in sorted(by_template.items())
    }

    return {
        "created_at_utc": utc_now(),
        "items_requested": total,
        "items_ok": ok,
        "items_failed": total - ok,
        "count_exact_match": correct,
        "items_scored": len(scored),
        "items_skipped_large_gold_set": sum(
            1 for r in results if r.get("status") == "skipped_large_gold_set"
        ),
        "count_accuracy": correct / len(scored) if scored else 0.0,
        "template_accuracy": template_accuracy,
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "median": sorted(latencies)[len(latencies) // 2]
            if latencies
            else None,
        },
        "note": (
            "Codex QA uses oracle metadata retrieval (all relevant emails). "
            "Use relevant_record_ids for dense retrieval evaluation separately."
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build corpus index and synthetic Q&A ground truth."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/engramme_test"),
    )
    parser.add_argument("--count", type=int, default=DEFAULT_QA_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_QA_SEED)
    parser.add_argument(
        "--corpus-sample-n",
        type=int,
        default=DEFAULT_CORPUS_SAMPLE_N,
    )
    parser.add_argument(
        "--corpus-sample-seed",
        type=int,
        default=DEFAULT_CORPUS_SAMPLE_SEED,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        ensure_synthetic_qa_dataset(
            args.data_dir,
            corpus_sample_n=args.corpus_sample_n,
            corpus_sample_seed=args.corpus_sample_seed,
            qa_count=args.count,
            seed=args.seed,
            force_regenerate=args.force,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
