from typing import List, Tuple

from dgraph_scaler.mpi import rank, size, comm, NodeType, Tags
from dgraph_scaler.typing import RawEdge, PartitionMap


def distribute_edges(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    if rank == NodeType.MASTER:
        return distribute_edges_leader(input_file)
    else:
        return distribute_edges_follower()


def distribute_edges_leader(input_file: str) -> Tuple[List[RawEdge], PartitionMap]:
    # TODO: get a more fair partitioining
    map = []
    with open(input_file) as file:
        total_edges_amount = int(file.readline().rstrip())
        edges = [None] * (total_edges_amount // size) * 1.5
        nodes = [None] * (total_edges_amount // size) * 1.5
        total_i = 0
        partition_amount = total_edges_amount // size
        for node in range(size):
            i = -1
            while total_i < total_edges_amount and (i < partition_amount or nodes[i] == nodes[max(0, i - 1)]):
                total_i += 1
                edges[i] = file.readline().rstrip()
                nodes[i] = edges[i].split()[0]
            map.append(int(nodes[i - 1]))
            comm.send(edges[:i], node, Tags.INITIAL_EDGES)

        for follower in range(1, size):
            comm.send(map, follower, Tags.PARTITION_MAP)

        return edges, map


def distribute_edges_follower() -> Tuple[List[RawEdge], PartitionMap]:
    edges = comm.recv(source=NodeType.MASTER, tag=Tags.INITIAL_EDGES)
    map = comm.recv(source=NodeType.MASTER, tag=Tags.PARTITION_MAP)
    return edges, map
