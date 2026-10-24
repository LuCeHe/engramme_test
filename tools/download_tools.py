
"""Download and extract the Kaggle Enron email dataset."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "wcukierski/enron-email-dataset"


def find_dataset_csv(raw_dir: Path) -> Path:
    """Return the largest CSV under raw_dir, preferring emails.csv."""
    candidates = [p for p in raw_dir.rglob("*.csv") if p.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"No CSV found under {raw_dir}. Dataset extraction may have failed."
        )

    candidates.sort(
        key=lambda p: (
            p.name.casefold() == "emails.csv",
            p.stat().st_size,
        ),
        reverse=True,
    )
    return candidates[0]


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP only when all member paths remain inside destination."""
    root = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()

            if target != root and root not in target.parents:
                raise ValueError(
                    f"Unsafe path in dataset archive: {member.filename}"
                )

        archive.extractall(destination)


def _kaggle_credentials_configured() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "access_token").is_file() or (
        kaggle_dir / "kaggle.json"
    ).is_file()


def _download_from_kaggle_cli(raw_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(raw_dir),
    ]

    print(f"Downloading Kaggle dataset {DATASET} ...")
    result = subprocess.run(command, text=True, capture_output=True)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "kaggle CLI exited with a non-zero status")


def _download_from_kaggle(raw_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        _download_from_kaggle_cli(raw_dir)
        return

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading Kaggle dataset {DATASET} ...")
    api.dataset_download_files(
        DATASET,
        path=str(raw_dir),
        unzip=False,
        quiet=False,
    )


def download_enron_dataset(
    data_dir: str | Path = "../data/engramme_test",
) -> Path:
    """
    Ensure the Enron dataset is downloaded and extracted.

    Returns the path to the extracted CSV.

    Expected layout:
        data_dir/
            raw/
                enron-email-dataset.zip
                emails.csv
    """
    data_dir = Path(data_dir).resolve()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # First-run behavior: reuse an already extracted dataset.
    existing_csvs = list(raw_dir.rglob("*.csv"))
    if existing_csvs:
        return find_dataset_csv(raw_dir)

    zip_path = raw_dir / "enron-email-dataset.zip"

    if not zip_path.exists():
        if not _kaggle_credentials_configured():
            kaggle_dir = Path.home() / ".kaggle"
            raise RuntimeError(
                "Kaggle credentials are missing. Use one of:\n"
                "  - kaggle auth login\n"
                "  - set KAGGLE_API_TOKEN (from kaggle.com/settings/api)\n"
                "  - set KAGGLE_USERNAME and KAGGLE_KEY\n"
                f"  - {kaggle_dir / 'access_token'} or "
                f"{kaggle_dir / 'kaggle.json'}"
            )

        try:
            _download_from_kaggle(raw_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Kaggle dataset download failed for {DATASET}."
            ) from exc

        # Kaggle may choose its own archive filename.
        archives = list(raw_dir.glob("*.zip"))
        if not archives:
            raise FileNotFoundError(
                f"Download completed but no ZIP archive was found in {raw_dir}"
            )

        downloaded_zip = max(archives, key=lambda p: p.stat().st_size)
        if downloaded_zip != zip_path:
            downloaded_zip.rename(zip_path)

    print(f"Extracting dataset into {raw_dir} ...")
    _safe_extract(zip_path, raw_dir)

    csv_path = find_dataset_csv(raw_dir)
    print(f"Dataset ready: {csv_path}")
    return csv_path
