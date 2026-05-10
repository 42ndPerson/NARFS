from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SubMeshes:
    submeshes: list[Mesh] # list of submeshes, each submesh is a subset of the full mesh that can run model(s) independently

@dataclass
class Mesh:
    host_ids: list[int] # list of host IDs in the mesh
    num_devices_per_host: int # number of devices per host