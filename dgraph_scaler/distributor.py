from typing import List, Tuple

from dgraph_scaler.mpi import rank, size, comm, NodeType, Tags
from dgraph_scaler.typing import RawEdge, PartitionMap


def distribute_edges(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    if rank == NodeType.MASTER:
        return distribute_edges_leader(input_file)
    else:
        return distribute_edges_follower()


def distribute_edges_leader(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    with open(input_file) as file:
        partition_map = []
        total_edges_amount = int(file.readline().rstrip())
        extra_edges = total_edges_amount % size
        edges_buffer = [None] * (total_edges_amount // size + 1)

        for follower in range(1, size):
            edges_amount = total_edges_amount // size + 1 if follower < extra_edges else total_edges_amount // size
            for i in range(edges_amount):
                edges_buffer[i] = file.readline().rstrip()
            partition_map.append(int(edges_buffer[i].split()[0]))
            comm.send(edges_buffer[:i + 1], follower, Tags.INITIAL_EDGES)

        for i in range(total_edges_amount // size + 1 if extra_edges else total_edges_amount // 2):
            edges_buffer[i] = file.readline().rstrip()
        partition_map.append(int(edges_buffer[i].split()[0]))

        for follower in range(1, size):
            comm.send(partition_map, follower, Tags.PARTITION_MAP)

        return edges_buffer[:i + 1], partition_map


def distribute_edges_follower() -> Tuple[List[RawEdge], PartitionMap]:
    edges = comm.recv(source=NodeType.MASTER, tag=Tags.INITIAL_EDGES)
    map = comm.recv(source=NodeType.MASTER, tag=Tags.PARTITION_MAP)
    return edges, map
