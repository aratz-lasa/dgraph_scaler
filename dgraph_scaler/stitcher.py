import random
from enum import Enum
from typing import List

import networkx as nx

from dgraph_scaler import mpi
from dgraph_scaler.util import choices


class StitchType(Enum):
    ALL_TO_ALL = 0
    RING = 1

    @staticmethod
    def parse_type(raw_type: str):
        if raw_type == "all-to-all":
            return StitchType.ALL_TO_ALL
        elif raw_type == "ring":
            return StitchType.RING
        else:
            raise ValueError("Invalid sticthing type")


def stitch_samples(samples: List[nx.MultiDiGraph], bridges_percent: float, stitch_type: StitchType):
    bridges_amount = int(samples[0].number_of_nodes() * bridges_percent)
    # bridges_amount = int(min(samples[0].number_of_nodes(), samples[-1].number_of_nodes())*bridges_percent)
    local_stitching(samples, bridges_amount, stitch_type)
    distributed_stitching(samples, bridges_amount, stitch_type)


def local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int, stitch_type: StitchType):
    if stitch_type == StitchType.ALL_TO_ALL:
        all_to_all_local_stitching(samples, bridges_amount)
    elif stitch_type == StitchType.RING:
        ring_local_stitching(samples, bridges_amount)
    else:
        raise ValueError("Invalid Stitching type")


def distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int, stitch_type: StitchType):
    if stitch_type == StitchType.ALL_TO_ALL:
        all_to_all_distributed_stitching(samples, bridges_amount)
    elif stitch_type == StitchType.RING:
        ring_distributed_stitching(samples, bridges_amount)
    else:
        raise ValueError("Invalid Stitching type")


def all_to_all_local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    samples_amount = len(samples)
    if samples_amount < 2:
        return
    tails = [[choices(list(sample.nodes), k=bridges_amount) for _ in range(len(samples))] for sample in samples]
    heads = [[choices(list(sample.nodes), k=bridges_amount) for _ in range(len(samples))] for sample in samples]
    for i, sample in enumerate(samples):
        for j in range(len(samples)):
            if i != j:
                sample.add_edges_from(zip(tails[i][j], heads[j][i]))


def all_to_all_distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    raw_samples = []
    for sample in samples:
        raw_samples.extend(sample.nodes)
    random.shuffle(raw_samples)
    my_remote_heads = [choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]

    tails = [choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]
    remote_heads = mpi.comm.alltoall(my_remote_heads)
    for i in range(mpi.size):
        if i != mpi.rank:
            bridges_amount_i = min(len(remote_heads[i]), len(tails[i]))
            samples[0].add_edges_from(zip(remote_heads[i][:bridges_amount_i], tails[i][:bridges_amount_i]))


def ring_local_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    samples_amount = len(samples)
    if samples_amount < 2:
        return
    tails = [[choices(list(sample.nodes), k=bridges_amount) for _ in range(2)] for sample in samples]
    heads = [[choices(list(sample.nodes), k=bridges_amount) for _ in range(2)] for sample in samples]
    for i, sample in enumerate(samples):
        j = 0 if i == len(samples) - 1 else i + 1
        if i != j:
            sample.add_edges_from(zip(tails[i][0], heads[j][1]))


def ring_distributed_stitching(samples: List[nx.MultiDiGraph], bridges_amount: int):
    raw_samples = []
    for sample in samples:
        raw_samples.extend(sample.nodes)
    random.shuffle(raw_samples)
    my_remote_heads = [choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]

    tails = [choices(raw_samples, k=bridges_amount) for _ in range(mpi.size)]
    remote_heads = mpi.comm.alltoall(my_remote_heads)
    for i in range(mpi.size):
        if i != mpi.rank:
            bridges_amount_i = min(len(remote_heads[i]), len(tails[i]))
            samples[0].add_edges_from(zip(remote_heads[i][:bridges_amount_i], tails[i][:bridges_amount_i]))
