from enum import IntEnum, auto

from mpi4py import MPI


class NodeType(IntEnum):
    MASTER = 0
    FOLLOWER = 1


class Tags(IntEnum):
    INITIAL_EDGES = auto()
    PARTITION_MAP = auto()
    OWNERSHIPS = auto()
    EDGE_QUERY = auto()
    EDGE_RESPONSE = auto()
    STITCHING = auto()
    MERGE = auto()


PICKLE_SET_OVERHEAD = 31
PICKLE_LIST_OVERHEAD = 6
PICKLE_TUPLE_OVERHEAD = 3
PICKLE_BIG_INT_OVERHEAD = 25
MAX_NODES = 200_000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

buf = MPI.Alloc_mem(MAX_NODES * PICKLE_BIG_INT_OVERHEAD + PICKLE_SET_OVERHEAD * size + MPI.BSEND_OVERHEAD * size)
MPI.Attach_buffer(buf)
