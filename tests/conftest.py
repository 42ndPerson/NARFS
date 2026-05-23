"""Put the project root on sys.path so tests can use absolute imports
(`from topology.hardware import HardwareSpec`, `from operators.gemm import Gemm`,
etc.) without requiring the project to be installed as a package.
"""
from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
