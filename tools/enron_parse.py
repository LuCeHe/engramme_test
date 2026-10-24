"""Parse Enron email messages for metadata extraction."""

from __future__ import annotations

import re
from datetime import timezone
from email import policy
from email.parser import Parser
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def normalize_addresses(value: str) -> list[str]:
    addresses = getaddresses([value or ""])
    return sorted({
        address.strip().casefold()
        for _, address in addresses
        if address and "@" in address
    })


def normalize_date(value: str) -> str:
    if not value:
        return ""

    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return ""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return ""


def parse_email_metadata(raw: str) -> dict[str, Any]:
    """Parse headers only (fast path for corpus indexing)."""
    msg = Parser(policy=policy.default).parsestr(raw or "")
    return {
        "sender": normalize_addresses(str(msg.get("From", ""))),
        "recipients": normalize_addresses(
            str(msg.get("To", "")) + ", " + str(msg.get("Cc", ""))
        ),
        "date_utc": normalize_date(str(msg.get("Date", ""))),
        "subject": str(msg.get("Subject", "")).strip(),
    }


def parse_email(raw: str) -> dict[str, Any]:
    """Parse an RFC-style email message into metadata and body."""
    msg = Parser(policy=policy.default).parsestr(raw or "")

    body_parts = []
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() != "text":
                    continue
                if part.get_content_disposition() == "attachment":
                    continue
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        body_parts.append(content)
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(
                            payload.decode("utf-8", errors="replace")
                        )
        else:
            content = msg.get_content()
            if isinstance(content, str):
                body_parts.append(content)
    except Exception:
        pass

    body = "\n".join(body_parts).strip()
    if not body:
        body = raw or ""

    meta = parse_email_metadata(raw)
    return {
        **meta,
        "sender_raw": str(msg.get("From", "")),
        "recipients_raw": str(msg.get("To", "")),
        "cc_raw": str(msg.get("Cc", "")),
        "date_raw": str(msg.get("Date", "")),
        "subject_raw": str(msg.get("Subject", "")),
        "body": body,
    }
