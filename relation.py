from __future__ import annotations
from enum import Enum


class Relation(Enum):
    EXCLUSIVE = "exclusive"
    CONCURRENT = "concurrent"


class NetworkPhase(Enum):
    STEADY = "steady"      # repeats throughout a stage; bandwidth-shared contention
    BOUNDARY = "boundary"  # one-shot at stage edges; flow-time contention
