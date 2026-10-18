"""Tree walking for structure tests."""
from __future__ import annotations

from dataclasses import fields, is_dataclass


def iter_blocks(block):
    """Every block in the tree, parents before children."""
    yield block
    for f in fields(block):
        v = getattr(block, f.name)
        items = v if isinstance(v, tuple) else (v,)
        for x in items:
            if is_dataclass(x) and not isinstance(x, type) and type(x).__module__ == "narfs.blocks":
                yield from iter_blocks(x)


def of_type(block, cls):
    return [b for b in iter_blocks(block) if isinstance(b, cls)]
