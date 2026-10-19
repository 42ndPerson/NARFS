"""NARFS: an analytical RLHF placement simulator.

Five layers own five kinds of knowledge:

    L0  physics: compute cost, the tiered network router, the communication
        library model and host cost (narfs.costs, narfs.network)
    L1  the core evaluator: eight node types, the fluid bound, memo and
        explain (narfs.blocks, narfs.evaluate)
    L2  the layout layer: degrees, device mapping, handles and layout
        transfers (narfs.parallel)
    L3  lowerings: components, schedules, overlap, variation and the
        generation engine
    L4  the problem: RL flows, models, profiles and the workload

Only the evaluator turns structure into time, and only the layout layer reads
the device mapping.
"""
