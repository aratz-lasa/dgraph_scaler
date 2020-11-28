from typing import List

import networkx as nx

from dgraph_scaler.mpi_util import GraphMPI, Tags


def merge_samples(samples: List[nx.MultiDiGraph], output_file: str, mpi: GraphMPI):
    if mpi.rank == 0:
        merge_samples_master(samples, output_file, mpi)
    else:
        merge_samples_follower(samples, mpi)


def merge_samples_master(samples: List[nx.MultiDiGraph], output_file: str, mpi: GraphMPI):
    with open(output_file, "w") as file:
        for sample in samples:
            file.writelines(map(lambda e: f"{e[0]} {e[1]}\n", sample.edges))
        for node in range(mpi.size):
            if node != mpi.rank:
                edges = mpi.comm.recv(source=node, tag=Tags.MERGE)
                file.writelines(edges)


def merge_samples_follower(samples: List[nx.MultiDiGraph], mpi: GraphMPI):
    edges = []
    for sample in samples:
        edges.extend(map(lambda e: f"{e[0]} {e[1]}\n", sample.edges))
    mpi.comm.send(edges, dest=0, tag=Tags.MERGE)
