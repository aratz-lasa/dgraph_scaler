from enum import IntEnum, auto

from mpi4py import MPI


class NodeType(IntEnum):
    MASTER = 0
    FOLLOWER = 1


class Tags(IntEnum):
    INITIAL_EDGES = auto()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()