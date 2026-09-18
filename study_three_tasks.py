"""Shared, versioned three-task study contract for both domains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PATH = Path(__file__).resolve().parent / "configs" / "three_task_study.json"
PROTOCOL: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
VERSION: str = PROTOCOL["version"]


def task(domain: str, number: int) -> dict[str, Any]:
    tasks = PROTOCOL["tasks"][domain]
    if not 1 <= number <= len(tasks):
        raise ValueError(f"Unknown {domain} task: {number}")
    result = tasks[number - 1]
    if result["id"] != number:
        raise ValueError(f"Out-of-order {domain} task configuration")
    return result


def permits(domain: str, number: int, group: str, *, review: bool = False) -> bool:
    field = "review_groups" if review else "explanation_groups"
    return group in task(domain, number)[field]
