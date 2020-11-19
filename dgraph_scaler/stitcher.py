import random
from typing import List

import networkx as nx

from dgraph_scaler import mpi


def stitch_samples(samples: List[nx.MultiDiGraph], bridges_amount: int):
    local_stitching(samples, bridges_amount)
    distributed_stitching(samples, bridges_amount)


def local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    samples_amount = len(samples)
    for i, sample in enumerate(samples):
        if i == samples_amount - 1:
            destination = samples[0]
        else:
            destination = samples[i + 1]
        tails = random.sample(sample.nodes, k=bridges_amount)
        heads = random.sample(destination.nodes, k=bridges_amount)
        sample.add_edges_from(zip(tails, heads))


def distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    my_remote_heads = random.sample(samples[0].nodes, k=bridges_amount)  # Head that other nodes will connect to
    tails = random.sample(samples[0].nodes, k=bridges_amount)
    dest = mpi.size - 1 if mpi.rank == 0 else mpi.rank - 1
    source = 0 if mpi.rank == mpi.size - 1 else mpi.rank + 1
    mpi.comm.bsend(my_remote_heads, dest=dest, tag=mpi.Tags.STITCHING)
    remote_heads = mpi.comm.recv(source=source, tag=mpi.Tags.STITCHING)
    samples[0].add_edges_from(zip(tails, remote_heads))

    mpi.comm.barrier()
