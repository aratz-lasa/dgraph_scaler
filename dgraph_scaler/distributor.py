import math
from typing import List, Tuple

from dgraph_scaler.mpi import rank, size, comm
from dgraph_scaler.typing import RawEdge
from dgraph_scaler.util import PartitionMap


def distribute_edges(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    return distribute_edges_leader(input_file)


def distribute_edges_leader(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    with open(input_file) as file:
        total_edges_amount = int(file.readline().rstrip())
        edges_amount = [total_edges_amount // size] * size
        for i in range(total_edges_amount % size):
            edges_amount[i] += 1
        forward_amount = sum(edges_amount[:rank])
        for _ in range(forward_amount):
            file.readline()
        edges_buffer = [None] * (total_edges_amount // size + 1)
        for i in range(edges_amount[rank]):
            edges_buffer[i] = file.readline().rstrip()

    my_mapping = (int(edges_buffer[0].split()[0]), int(edges_buffer[i].split()[0]))

    raw_map = comm.alltoall([my_mapping] * (size))
    raw_map = fill_map_gaps(raw_map)
    return edges_buffer[:i + 1], PartitionMap(raw_map)


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
