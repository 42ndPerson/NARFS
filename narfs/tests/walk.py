"""Tree helpers for structure tests."""
from __future__ import annotations

from narfs.blocks import walk


def of_type(tree, cls) -> list:
    """Every node of type `cls` in the tree (shared nodes once)."""
    return [n for n in walk(tree) if isinstance(n, cls)]


def named(tree, name: str) -> list:
    return [n for n in walk(tree) if getattr(n, "name", "") == name]
