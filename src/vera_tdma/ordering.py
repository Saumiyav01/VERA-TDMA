"""Deterministic, case-sensitive natural ordering for node identifiers."""

import re
from collections.abc import Iterable
from typing import TypeVar

_PARTS = re.compile(r"(\d+)")
T = TypeVar("T", bound=str)


def natural_sort_key(identifier: str) -> tuple[tuple[int, str | int, int], ...]:
    """Return a case-sensitive natural-sort key for an identifier."""
    if not isinstance(identifier, str):
        raise TypeError("node identifiers must be strings")

    key: list[tuple[int, str | int, int]] = []
    for part in _PARTS.split(identifier):
        if not part:
            continue
        if part.isdecimal():
            key.append((1, int(part), len(part)))
        else:
            key.append((0, part, 0))
    return tuple(key)


def natural_sorted(identifiers: Iterable[T]) -> list[T]:
    """Sort identifiers naturally while preserving their original spelling."""
    return sorted(identifiers, key=natural_sort_key)
