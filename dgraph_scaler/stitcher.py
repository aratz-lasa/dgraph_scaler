import random
from typing import List

import networkx as nx

from dgraph_scaler import mpi


def stitch_samples(samples: List[nx.MultiDiGraph], bridges_amount: int):
    local_stitching(samples, bridges_amount)
    distributed_stitching(samples, bridges_amount)


def local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    samples_amount = len(samples)
    if samples_amount < 2:
        return
    for i, sample in enumerate(samples):
        if i == samples_amount - 1:
            destination = samples[0]
        else:
            destination = samples[i + 1]
        tails = random.sample(sample.nodes, k=bridges_amount//2)
        heads = random.sample(destination.nodes, k=bridges_amount//2)
        sample.add_edges_from(zip(tails, heads))


def distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    raw_samples = [list(s.nodes) for s in samples]
    my_remote_heads = [random.sample(random.choice(raw_samples), k=bridges_amount//2) for _ in range(mpi.size)]

    tails = [random.sample(random.choice(raw_samples), k=bridges_amount//2) for _ in range(mpi.size)]
    remote_heads = mpi.comm.alltoall(my_remote_heads)
    for i in range(mpi.size):
        if i != mpi.rank:
            samples[0].add_edges_from(zip(remote_heads[i], tails[i]))

    mpi.comm.barrier()
