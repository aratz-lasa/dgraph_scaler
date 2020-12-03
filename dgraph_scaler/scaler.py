import time
from math import ceil

import networkx as nx

from dgraph_scaler import distributor, sampler, util, mpi, stitcher, merger
from dgraph_scaler.stitcher import StitchType


def scale(input_file, output_file, scale_factor, bridges=0.1, factor_size=0.5, precision=0.95, connect=False,
          stitching_type="all-to-all", merge_nfs=False, verbose=True):
    if verbose and mpi.rank == 0:
        print("""
  ______  ______ _______  _____  _     _      _______ _______ _______        _______  ______
 |  ____ |_____/ |_____| |_____] |_____|      |______ |       |_____| |      |______ |_____/
 |_____| |    \_ |     | |       |     |      ______| |_____  |     | |_____ |______ |    \_ █0.1█
""")
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
        print(
            "Scale factor: {}. Bridges: {}%. Precision: {}%. Factor size: {}.Connect: {}. Stitching:{}. NFS: {}".format(
                scale_factor,
                bridges * 100,
                precision * 100,
                factor_size,
                connect,
                stitching_type,
                merge_nfs), )
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
        print()

    total_t = time.time()
    # Step X: Parse sticthing type and check if valid
    stitching_type = StitchType.parse_type(stitching_type)

    # Step X: Read distribute edges and load graph
    loading_t = time.time()
    edges, partition_map, total_nodes = distributor.distribute_edges(input_file)
    graph = util.load_graph_from_edges(edges)
    edges = None  # Free memory
    if verbose and mpi.rank == 0:
        print("Loading time:", round(time.time() - loading_t, 2), "seconds")
        print("=================================")
    # Step X: Calculate weights (how many % of the nodes to sample) for each node
    nodes_amount = mpi.comm.alltoall([graph.number_of_nodes()] * mpi.size)
    weights = list(map(lambda a: ceil(a / sum(nodes_amount) * 100) / 100.0, nodes_amount))
    # Step X: Split factor into sample rounds
    factor_size = min(factor_size, scale_factor)
    factors = [factor_size for _ in range(int(scale_factor / factor_size))]
    remaining_factor = scale_factor - round(sum(factors), 2)
    if remaining_factor:
        factors.append(remaining_factor)
    # Step X: Run distributed sampling
    samples = []
    for i, factor in enumerate(factors):
        sampling_t = time.time()
        samples.append(sampler.sample(graph, int(total_nodes * factor), weights[mpi.rank], partition_map, precision))
        if verbose and mpi.rank == 0:
            print("Sampling time {}/{}:".format(i + 1, len(factors)), round(time.time() - sampling_t, 2), "seconds")
    if verbose and mpi.rank == 0:
        print("=================================")
    # Step X: Connect the graph
    if connect:
        connecting_t = time.time()
        for sample in samples:
            sample.add_edges_from(nx.k_edge_augmentation(nx.Graph(sample), 1))
        if verbose and mpi.rank == 0:
            print("Connecting time:", round(time.time() - connecting_t, 2), "seconds")
            print("=================================")
    # Step X: Rename vertices
    relabeling_t = time.time()
    util.relabel_samples(samples)
    if verbose and mpi.rank == 0:
        print("Relabeling time:", round(time.time() - relabeling_t, 2), "seconds")
        print("=================================")
    # Step X: Stitch samples locally and distributively
    stitching_t = time.time()
    stitcher.stitch_samples(samples, bridges, stitching_type)
    if verbose and mpi.rank == 0:
        print("Stiching time:", round(time.time() - stitching_t, 2), "seconds")
        print("=================================")
    # Step X: Merge distributed samples into master file
    dumping_t = time.time()
    merger.merge_samples(samples, output_file, merge_nfs)
    if verbose and mpi.rank == 0:
        print("Dumping time:", round(time.time() - dumping_t, 2), "seconds")
        print("=================================")

    if verbose and mpi.rank == 0:
        print("Total time:", round(time.time() - total_t, 2), "seconds")
