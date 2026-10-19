# NARFS

NARFS is an analytical simulator for RLHF placement search. For one placement
of an RLHF job's model calls on a GPU cluster, it predicts the time of one RL
iteration and checks whether the placement fits in memory. An external
optimizer calls it in a loop, so it is deterministic and fast.

The code is in [narfs/](narfs/); [narfs/README.md](narfs/README.md) describes
the layers and the API.

```bash
pip install -r narfs/requirements.txt
python -m pytest narfs/tests
```

The earlier simulator (`topology/`, `operators/`, `schemes/`, `mapping/`,
`simulation/`, `validation/` and the root modules) was removed from the tree.
It is still in the git history.
