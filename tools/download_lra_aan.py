#!/usr/bin/env python3
"""Reliably download the three official AAN TSV files with HTTP resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import get_hf_file_metadata


REPOSITORY = "OpenNLPLab/lra"
REVISION = "6cb133c44f406da743661488df9e977b87d98775"
FILES = (
    "new_aan_pairs.train.tsv",
    "new_aan_pairs.eval.tsv",
    "new_aan_pairs.test.tsv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/lra")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.data_root) / "opennlplab" / "data" / "aan"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in FILES:
        target = destination / filename
        partial = target.with_suffix(target.suffix + ".partial")
        url = (
            f"{args.endpoint.rstrip('/')}/datasets/{REPOSITORY}/resolve/"
            f"{REVISION}/data/aan/{quote(filename)}"
        )
        token = os.environ.get("HF_TOKEN")
        if token:
            # Authenticate only the short resolver request. Curl receives the
            # resulting signed CDN URL, so the long-lived process arguments
            # and progress log never contain the access token.
            url = get_hf_file_metadata(url, token=token).location
        if not target.is_file():
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--continue-at",
                    "-",
                    "--retry",
                    "20",
                    "--retry-delay",
                    "5",
                    "--retry-all-errors",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
            os.replace(partial, target)
        records.append({"filename": filename, "bytes": target.stat().st_size})
    manifest = {
        "schema": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "transport_endpoint": args.endpoint,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    (destination / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
