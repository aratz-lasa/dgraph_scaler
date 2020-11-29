from typing import List

import networkx as nx

from dgraph_scaler import mpi


def merge_samples(samples: List[nx.MultiDiGraph], output_file: str, nfs: bool):
    if nfs:
        merge_nfs(samples, output_file)
    else:
        merge_centralized(samples, output_file)


def merge_nfs(samples: List[nx.MultiDiGraph], output_file: str):
    for i, sample in enumerate(samples):
        nx.write_edgelist(sample, f"{output_file}.{mpi.rank}.{i}.txt", data=False)


def merge_centralized(samples: List[nx.MultiDiGraph], output_file: str):
    if mpi.rank == 0:
        merge_samples_master(samples, output_file)
    else:
        merge_samples_follower(samples)


def merge_samples_master(samples: List[nx.MultiDiGraph], output_file: str):
    with open(f"{output_file}.txt", "w") as file:
        for sample in samples:
            file.writelines(map(lambda e: f"{e[0]} {e[1]}\n", sample.edges))
        for node in range(mpi.size):
            if node != mpi.rank:
                edges = mpi.comm.recv(source=node, tag=mpi.Tags.MERGE)
                file.writelines(edges)


def merge_samples_follower(samples: List[nx.MultiDiGraph]):
    edges = []
    for sample in samples:
        edges.extend(map(lambda e: f"{e[0]} {e[1]}\n", sample.edges))
    mpi.comm.send(edges, dest=0, tag=mpi.Tags.MERGE)
