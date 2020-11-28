import random
from typing import List

import networkx as nx

from dgraph_scaler.mpi_util import GraphMPI
from dgraph_scaler import util


def stitch_samples(samples: List[nx.MultiDiGraph], bridges_percent: float, mpi: GraphMPI):
    bridges_amount = int(samples[0].number_of_nodes() * bridges_percent)
    # bridges_amount = int(min(samples[0].number_of_nodes(), samples[-1].number_of_nodes())*bridges_percent)
    local_stitching(samples, bridges_amount)
    distributed_stitching(samples, bridges_amount, mpi)


def local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    samples_amount = len(samples)
    if samples_amount < 2:
        return
    tails = [[util.choices(list(sample.nodes), k=bridges_amount) for _ in range(len(samples))] for sample in samples]
    heads = [[util.choices(list(sample.nodes), k=bridges_amount) for _ in range(len(samples))] for sample in samples]
    for i, sample in enumerate(samples):
        for j in range(len(samples)):
            sample.add_edges_from(zip(tails[i][j], heads[j][i]))


def distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int, mpi: GraphMPI):
    raw_samples = []
    for sample in samples:
        raw_samples.extend(sample.nodes)
    random.shuffle(raw_samples)
    my_remote_heads = [util.choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]

    tails = [util.choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]
    remote_heads = mpi.comm.alltoall(my_remote_heads)
    for i in range(mpi.size):
        bridges_amount_i = min(len(remote_heads[i]), len(tails[i]))
        samples[0].add_edges_from(zip(remote_heads[i][:bridges_amount_i], tails[i][:bridges_amount_i]))

    mpi.comm.barrier()
