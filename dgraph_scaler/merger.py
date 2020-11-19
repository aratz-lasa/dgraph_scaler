from typing import List

import networkx as nx

from dgraph_scaler import mpi


def dump_samples(samples: List[nx.MultiDiGraph], output_file: str):
    for i, sample in enumerate(samples):
        nx.write_edgelist(sample, f"{output_file}.{mpi.rank}.{i}.edges", data=False)