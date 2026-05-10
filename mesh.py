from __future__ import annotations
from dataclasses import dataclass

from topo import Device


@dataclass
class SubMeshes:
    submeshes: list[Mesh]  # subsets of the full mesh, each runs model(s) independently


@dataclass
class Mesh:
    """A flat list of Devices a model is mapped onto.

    Devices carry their own (host_id, gpu_id) so the mesh no longer needs
    to encode host grouping or per-host device count separately.
    """
    devices: list[Device]
