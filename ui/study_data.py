"""Export the current transactional collaborative-study log to JSONL."""

from __future__ import annotations

import argparse

from .study_store import StudyStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Current SQLite database path")
    parser.add_argument("--namespace", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("output", help="Destination JSONL path")
    args = parser.parse_args()
    store = StudyStore(args.db, namespace=args.namespace)
    count = store.export_jsonl(args.output)
    print(f"Exported {count} collaborative-study events.")


if __name__ == "__main__":
    main()
