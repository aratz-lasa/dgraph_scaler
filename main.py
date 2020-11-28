import time
from math import ceil

import click
import networkx as nx
from mpi4py import MPI

from dgraph_scaler import distributor, sampler, util, mpi_util, stitcher, merger

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
@click.option('-b', '--bridges', default=0.1, type=float)
@click.option('-fs', '--factor-size', default=0.5, type=float)
@click.option('-p', '--precision', default=0.95, type=float)
@click.option('-c', '--connect', is_flag=True, )
def distributed_sampling(input_file, output_file, scale_factor, bridges, factor_size, precision, connect):
    mpi = mpi_util.GraphMPI(comm, rank, size)

    total_t = time.time()

    # Step X: Read distribute edges and load graph
    loading_t = time.time()
    edges, partition_map, total_nodes = distributor.distribute_edges(input_file, mpi)
    graph = util.load_graph_from_edges(edges)
    edges = None  # Free memory
    if mpi.rank == 0:
        print("Loading time:", time.time() - loading_t)
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
        samples.append(
            sampler.sample_graph(graph, int(total_nodes * factor), weights[mpi.rank], partition_map, precision, mpi))
        if mpi.rank == 0:
            print("Sampling time", (i + 1), "/", len(factors), ":", time.time() - sampling_t)
    # Step X: Connect the graph
    if connect:
        connecting_t = time.time()
        for sample in samples:
            sample.add_edges_from(nx.k_edge_augmentation(nx.Graph(sample), 1))
        if mpi.rank == 0:
            print("Connecting time:", time.time() - connecting_t)
    # Step X: Rename vertices
    relabeling_t = time.time()
    util.relabel_samples(samples)
    if mpi.rank == 0:
        print("Relabeling time:", time.time() - relabeling_t)
    # Step X: Stitch samples locally and distributively
    stitching_t = time.time()
    stitcher.stitch_samples(samples, bridges, mpi)
    if mpi.rank == 0:
        print("Stiching time:", time.time() - stitching_t)
    # Step X: Merge distributed samples into master file
    dumping_t = time.time()
    merger.merge_samples(samples, output_file, mpi)
    if mpi.rank == 0:
        print("Dumping time:", time.time() - dumping_t)

    if mpi.rank == 0:
        print("Total time:", time.time() - total_t)


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
>  python3.6 main.py datasets/ordered/facebook.edges samples/facebook 2.5
"""
