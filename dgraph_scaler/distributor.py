import math
from typing import List, Tuple

from dgraph_scaler.mpi import rank, size, comm, NodeType, Tags
from dgraph_scaler.typing import RawEdge
from dgraph_scaler.util import PartitionMap


def distribute_edges(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    if rank == NodeType.MASTER:
        return distribute_edges_leader(input_file)
    else:
        return distribute_edges_follower()


def distribute_edges_leader(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    with open(input_file) as file:
        raw_map = [None] * size
        total_edges_amount = int(file.readline().rstrip())
        extra_edges = total_edges_amount % size
        edges_buffer = [None] * (total_edges_amount // size + 1)

        for i in range(total_edges_amount // size + 1 if extra_edges else total_edges_amount // 2):
            edges_buffer[i] = file.readline().rstrip()
        raw_map[NodeType.MASTER] = ((int(edges_buffer[0].split()[0]), int(edges_buffer[i].split()[0])))

        for follower in range(1, size):
            edges_amount = total_edges_amount // size + 1 if follower < extra_edges else total_edges_amount // size
            for i in range(edges_amount):
                edges_buffer[i] = file.readline().rstrip()
            raw_map[follower] = (int(edges_buffer[0].split()[0]), int(edges_buffer[i].split()[0]))
            comm.send(edges_buffer[:i + 1], follower, Tags.INITIAL_EDGES)

        new_map = fill_map_gaps(raw_map)

        for follower in range(1, size):
            comm.send(new_map, follower, Tags.PARTITION_MAP)

        return edges_buffer[:i + 1], PartitionMap(new_map)


def distribute_edges_follower() -> Tuple[List[RawEdge], PartitionMap]:
    edges = comm.recv(source=NodeType.MASTER, tag=Tags.INITIAL_EDGES)
    raw_map = comm.recv(source=NodeType.MASTER, tag=Tags.PARTITION_MAP)
    return edges, PartitionMap(raw_map)


def fill_map_gaps(raw_map):
    new_map = []
    prev_last = 0
    for i, limit in enumerate(raw_map):
        first, last = limit
        if first > prev_last:
            first = prev_last + 1
        if i == len(raw_map) - 1:
            last = math.inf
        new_map.append((first, last))
        prev_last = last
    return new_map
