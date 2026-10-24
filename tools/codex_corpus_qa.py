"""Run corpus Q&A items through Codex."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai_codex import Codex, Sandbox

from tools.codex_runner import BASE_INSTRUCTIONS, DEFAULT_REASONING_EFFORT, _usage_from_turn

CORPUS_QA_MODEL = "gpt-5.6-luna"


def build_corpus_qa_prompt(question: str, evidence_lines: list[str]) -> str:
    evidence_block = "\n".join(evidence_lines)
    return f"""
Answer an aggregate question about the email corpus using ONLY the evidence
emails listed below. Each evidence row is one email's metadata.

Question:
{question}

Evidence emails ({len(evidence_lines)}):
{evidence_block}

Return one JSON object with exactly:
- count: integer (how many evidence emails match the question)
- brief_reason: one short sentence explaining the count

Count only from the evidence list. Return JSON only, no Markdown fences.
""".strip()


def parse_count_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain a JSON object")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    if "count" not in parsed:
        raise ValueError("JSON object missing 'count'")
    return parsed


def run_corpus_qa_one(
    codex: Codex,
    question_row: dict[str, Any],
    evidence_lines: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    model_id = model or CORPUS_QA_MODEL
    prompt = build_corpus_qa_prompt(question_row["question"], evidence_lines)

    thread = codex.thread_start(
        model=model_id,
        config={"model_reasoning_effort": DEFAULT_REASONING_EFFORT},
        sandbox=Sandbox.read_only,
        base_instructions=BASE_INSTRUCTIONS,
        ephemeral=True,
    )

    started = time.perf_counter()
    result = thread.run(prompt)
    elapsed = time.perf_counter() - started

    output: dict[str, Any] = {
        "model": model_id,
        "thread_id": thread.id,
        "raw_response": result.final_response,
        "usage": _usage_from_turn(result, elapsed),
    }

    try:
        parsed = parse_count_response(result.final_response or "")
        output["prediction"] = parsed
        output["status"] = "ok"
    except (ValueError, json.JSONDecodeError) as exc:
        output["status"] = "parse_error"
        output["parse_error"] = str(exc)

    return output
