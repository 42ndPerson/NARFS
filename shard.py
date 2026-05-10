from __future__ import annotations
from dataclasses import dataclass

from model import Model


@dataclass(frozen=True)
class Shard:
    model: Model
    dp: int  # data parallel rank
    pp: int  # pipeline parallel rank
    tp: int  # tensor parallel rank
    ep: int = 0  # expert parallel rank (0 = no EP / degenerate single-rank EP group)
