import random
from math import ceil
from typing import List

import networkx as nx
from mpi4py import MPI

from dgraph_scaler import mpi
from dgraph_scaler.typing import Ownerships, Ownership, Edge
from dgraph_scaler.util import PartitionMap

PICKLE_SET_OVERHEAD = 31
PICKLE_BIG_INT_OVERHEAD = 25


def sample(graph: nx.MultiDiGraph, factor: float, partition_map: PartitionMap) -> nx.MultiDiGraph:
    # Step 1: Local random edges sampling
    sample = local_edge_sampling(graph, factor)
    # Step 2: Distributed induction
    distributed_induction(graph, sample, partition_map)
    return sample
    pass  # todo


def local_edge_sampling(graph: nx.MultiDiGraph, factor: float) -> nx.MultiDiGraph:
    candidate_edges = list(graph.edges)
    subgraph = nx.MultiDiGraph()
    sample_nodes_amount = int(graph.number_of_nodes() * factor)
    nodes_sampled = subgraph.number_of_nodes()
    while nodes_sampled < sample_nodes_amount:
        subgraph.add_edges_from(random.choices(candidate_edges, k=ceil((sample_nodes_amount - nodes_sampled) / 2)))
        nodes_sampled = subgraph.number_of_nodes()
    return subgraph


def distributed_induction(graph: nx.MultiDiGraph, sample: nx.MultiDiGraph, partition_map: PartitionMap):
    # Step 1: Determine ownerships
    ownerships = [set() for _ in range(mpi.size)]
    for vertex in sample.nodes:
        for owner in partition_map.get_owners(vertex):
            ownerships[owner].add(vertex)
    # Step 2: Distribute ownerships
    distribute_ownerships(ownerships, sample)
    # Step 3: Induction on non-sampled edges
    # Step 3.1: Get non-sampled edges non-owned nodes
    edge_queries = [[] for _ in range(mpi.size)]
    for edge in filter(lambda e: not sample.has_edge(*e), graph.edges):
        owners = partition_map.get_owners(edge[1])
        edge_queries[random.choice(owners)].append(edge)  # Select only one of the owners randomly
    # Step 3.2: Resolve induction of owned nodes
    for edge in edge_queries[mpi.rank]:
        if sample.has_node(edge[0]) and sample.has_node(edge[1]):
            sample.add_edge(*edge)
    # Step 3.3: Query each node's owner for
    query_inductions(sample, edge_queries, ownerships[mpi.rank])


def distribute_ownerships(ownerships, sample):
    buf = MPI.Alloc_mem(
        sample.number_of_nodes() * PICKLE_BIG_INT_OVERHEAD + PICKLE_SET_OVERHEAD * mpi.size + MPI.BSEND_OVERHEAD * mpi.size)
    MPI.Attach_buffer(buf)

    received_ownerships = 0
    for node in range(mpi.size):
        if node != mpi.rank:
            mpi.comm.bsend(ownerships[node], dest=node, tag=mpi.Tags.OWNERSHIPS)
        received = True
        while received:
            received = receive_ownerships(ownerships)
            if received:
                received_ownerships += 1
    for _ in range(mpi.size - 1 - received_ownerships):
        receive_ownerships_blocking(ownerships)

    mpi.comm.barrier()

    MPI.Detach_buffer()
    MPI.Free_mem(buf)


def query_inductions(sample: nx.MultiDiGraph, edge_queries: List[List[Edge]], ownership: Ownership):
    pass  # TODO


def receive_ownerships_blocking(ownerships: Ownerships):
    ownership = mpi.comm.recv(source=MPI.ANY_SOURCE, tag=mpi.Tags.OWNERSHIPS)
    ownerships[mpi.rank] = ownerships[mpi.rank].union(ownership)


def receive_ownerships(ownerships: Ownerships):
    flag = mpi.comm.iprobe(source=MPI.ANY_SOURCE, tag=mpi.Tags.OWNERSHIPS)
    if flag:
        ownership = mpi.comm.recv(source=MPI.ANY_SOURCE, tag=mpi.Tags.OWNERSHIPS)
        ownerships[mpi.rank] = ownerships[mpi.rank].union(ownership)
        return True
    return False
