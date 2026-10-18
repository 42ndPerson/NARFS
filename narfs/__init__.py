"""NARFS v2: an analytical RLHF placement simulator.

Design summary (see narfs/README.md for the full description):

- Once per optimization run: architecture descriptions, the component
  library, cost tables, the network resource index and framework profiles.
- Per candidate (inside the optimizer loop): levels 1-3 of placement,
  an early feasibility check, expansion into a small block tree, and
  evaluation by closed-form rules with array kernels for the network.
- Contention between concurrent blocks uses the fluid model.
"""
