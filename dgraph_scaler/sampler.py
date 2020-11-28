import random
from functools import reduce
from math import ceil
from typing import List, Set, Tuple

import networkx as nx

from dgraph_scaler.mpi_util import GraphMPI
from dgraph_scaler.typing import Ownership, Edge, Vertex
from dgraph_scaler.util import PartitionMap


def sample_graph(graph: nx.MultiDiGraph, total_nodes: int, weight: float, partition_map: PartitionMap,
                 precision: float, mpi: GraphMPI) -> nx.MultiDiGraph:
    ownership_lens = [0] * mpi.size
    ownerships = [set() for _ in range(mpi.size)]
    sample = nx.MultiDiGraph()
    candidate_edges = list(graph.edges)
    while sum(ownership_lens) < total_nodes * precision:
        # Step 1: Local random edges sampling
        sub_sample, candidate_edges = local_edge_sampling(candidate_edges, (total_nodes - sum(ownership_lens)) * weight)
        # Step 2: Calculate ownerships
        ownership_lens = distribute_ownerships(sub_sample, ownerships, partition_map, mpi)
        sub_sample.add_edges_from(sample.edges)
        sample = sub_sample  # Inefficient! In order to avoid seg fault
    # Step 2: Distributed induction
    distributed_induction(graph, sample, partition_map, ownerships[mpi.rank], mpi)
    return sample


def local_edge_sampling(candidate_edges: List[Vertex], nodes_amount: int) -> Tuple[nx.MultiDiGraph, List[Vertex]]:
    subgraph = nx.MultiDiGraph()
    nodes_sampled = 0
    while nodes_sampled < nodes_amount:
        subgraph.add_edges_from(random.choices(candidate_edges, k=ceil((nodes_amount - nodes_sampled) / 2)))
        nodes_sampled = subgraph.number_of_nodes()
    candidate_edges = list(filter(lambda e: not subgraph.has_edge(*e), candidate_edges))
    return subgraph, candidate_edges


def distribute_ownerships(sample: nx.MultiDiGraph, ownerships: List[Set[Vertex]], partition_map: PartitionMap, mpi: GraphMPI) -> List[
    int]:
    for vertex in sample.nodes:
        for owner in partition_map.get_owners(vertex):
            ownerships[owner].add(vertex)
    # Step 2: Distribute ownerships
    remote_ownerships = mpi.comm.alltoall(ownerships)
    ownerships[mpi.rank] = ownerships[mpi.rank] | reduce(lambda o1, o2: o1 | o2, remote_ownerships)
    ownership_lens = mpi.comm.alltoall([len(ownerships[mpi.rank])] * mpi.size)
    return ownership_lens


def distributed_induction(graph: nx.MultiDiGraph, sample: nx.MultiDiGraph, partition_map: PartitionMap,
                          ownership: Set[Vertex], mpi: GraphMPI):
    # Step 1: Induction on non-sampled edges
    # Step 1.1: Get non-sampled edges non-owned nodes
    edge_queries = [[] for _ in range(mpi.size)]
    for edge in filter(lambda e: not sample.has_edge(*e) and sample.has_node(e[0]), graph.edges):
        owners = partition_map.get_owners(edge[1])
        edge_queries[random.choice(owners)].append(edge)  # Select only one of the owners randomly
    # Step 1.2: Resolve induction of owned nodes
    for edge in edge_queries[mpi.rank]:
        if edge[1] in ownership:
            sample.add_edge(*edge)
    edge_queries[mpi.rank].clear()
    # Step 1.3: Query each node's owner for
    query_inductions(sample, edge_queries, ownership, mpi)


def query_inductions(sample: nx.MultiDiGraph, edge_queries: List[List[Edge]], ownership: Ownership, mpi: GraphMPI):
    remote_queries = mpi.comm.alltoall(edge_queries)
    answers = [list(filter(lambda e: e[1] in ownership, q)) for q in remote_queries]
    remote_answers = mpi.comm.alltoall(answers)
    for remote_answer in remote_answers:
        sample.add_edges_from(remote_answer)
    mpi.comm.barrier()
