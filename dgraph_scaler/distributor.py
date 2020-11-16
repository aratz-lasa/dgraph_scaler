from typing import List

from dgraph_scaler.mpi import rank, size, comm, NodeType, Tags
from dgraph_scaler.typing import RawEdge


def distribute_edges(input_file: str) -> List[RawEdge]:
    if rank == NodeType.MASTER:
        return distribute_edges_leader(input_file)
    else:
        return distribute_edges_follower()


def distribute_edges_leader(input_file: str) -> List[RawEdge]:
    with open(input_file) as file:
        total_edges_amount = int(file.readline().rstrip())
        extra_edges = total_edges_amount % size
        edges_buffer = [None] * (total_edges_amount // size + 1 if extra_edges else total_edges_amount // 2)
        for follower in range(1, size):
            edges_amount = total_edges_amount // size + 1 if follower < extra_edges else total_edges_amount // size
            for i in range(edges_amount):
                edges_buffer[i] = file.readline().rstrip()
            comm.send(edges_buffer, follower, Tags.INITIAL_EDGES)
        for i in range(total_edges_amount // size + 1 if extra_edges else total_edges_amount // 2):
            edges_buffer[i] = file.readline().rstrip()
        return edges_buffer


def distribute_edges_follower() -> List[RawEdge]:
    return comm.recv(source=NodeType.MASTER, tag=Tags.INITIAL_EDGES)
