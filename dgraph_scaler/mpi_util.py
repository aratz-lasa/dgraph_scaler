from enum import IntEnum


class GraphMPI:
    def __init__(self, comm, rank, size):
        self.comm = comm
        self.rank = rank
        self.size = size


class NodeType(IntEnum):
    MASTER = 0
    FOLLOWER = 1


class Tags(IntEnum):
    INITIAL_EDGES = 0
    PARTITION_MAP = 1
    OWNERSHIPS = 2
    EDGE_QUERY = 3
    EDGE_RESPONSE = 4
    STITCHING = 5
    MERGE = 6


PICKLE_SET_OVERHEAD = 31
PICKLE_LIST_OVERHEAD = 6
PICKLE_TUPLE_OVERHEAD = 3
PICKLE_BIG_INT_OVERHEAD = 25
MAX_NODES = 200000
