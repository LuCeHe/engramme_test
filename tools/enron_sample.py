"""Reservoir sampling over the Enron email CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 100_000_000))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def reservoir_sample(
    csv_path: Path,
    n: int,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    """
    Uniform reservoir sample without loading the whole CSV into memory.

    Each sampled record contains the source row ID and raw email message.
    """
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    total = 0

    print(f"Scanning CSV: {csv_path}")

    with csv_path.open(
        "r", encoding="utf-8-sig", newline="", errors="replace"
    ) as f:
        reader = csv.DictReader(f)

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

            record_id = (
                row.get(id_col, "").strip()
                if id_col else ""
            )
            if not record_id:
                record_id = f"row_{total}"

            item = {
                "record_id": record_id,
                "raw_message": raw,
                "raw_sha256": sha256_text(raw),
            }

            total += 1

            if len(sample) < n:
                sample.append(item)
            else:
                j = rng.randrange(total)
                if j < n:
                    sample[j] = item

            if total % 100_000 == 0:
                print(f"  scanned {total:,} nonempty emails")

    if total < n:
        print(f"Warning: only {total} nonempty emails found.")

    sample.sort(key=lambda x: x["record_id"])
    return sample, total


def write_sample_jsonl(sample_path: Path, sample: list[dict[str, str]]) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
