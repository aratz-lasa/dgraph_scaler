import random
from functools import reduce
from math import ceil
from typing import List

import networkx as nx

from dgraph_scaler import mpi
from dgraph_scaler.typing import Ownership, Edge
from dgraph_scaler.util import PartitionMap


def sample(graph: nx.MultiDiGraph, factor: float, partition_map: PartitionMap) -> nx.MultiDiGraph:
    # Step 1: Local random edges sampling
    sample = local_edge_sampling(graph, factor)
    # Step 2: Distributed induction
    distributed_induction(graph, sample, partition_map)
    return sample


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
    remote_ownerships = mpi.comm.alltoall(ownerships)
    ownerships[mpi.rank] = ownerships[mpi.rank] | reduce(lambda o1, o2: o1 | o2, remote_ownerships)
    # Step 3: Induction on non-sampled edges
    # Step 3.1: Get non-sampled edges non-owned nodes
    edge_queries = [[] for _ in range(mpi.size)]
    for edge in filter(lambda e: not sample.has_edge(*e), graph.edges):
        owners = partition_map.get_owners(edge[1])
        edge_queries[random.choice(owners)].append(edge)  # Select only one of the owners randomly
    # Step 3.2: Resolve induction of owned nodes
    my_ownership = ownerships[mpi.rank]
    for edge in edge_queries[mpi.rank]:
        if edge[0] in my_ownership and edge[1] in my_ownership:
            sample.add_edge(*edge)
    edge_queries[mpi.rank].clear()
    # Step 3.3: Query each node's owner for
    query_inductions(sample, edge_queries, ownerships[mpi.rank])


def query_inductions(sample: nx.MultiDiGraph, edge_queries: List[List[Edge]], ownership: Ownership):
    remote_queries = mpi.comm.alltoall(edge_queries)
    answers = [list(filter(lambda e: e[1] in ownership, q)) for q in remote_queries]
    remote_answers = mpi.comm.alltoall(answers)
    for remote_answer in remote_answers:
        sample.add_edges_from(remote_answer)
    mpi.comm.barrier()
