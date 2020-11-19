import itertools
import string
from typing import List

import networkx as nx

from dgraph_scaler.typing import RawEdge, Vertex


class PartitionMap:
    def __init__(self, partition_map):
        self.partition_map = partition_map

    def is_owner(self, machine, vertex):
        first, last = self.partition_map[machine]
        return first <= vertex <= last

    def get_owners(self, vertex) -> List[Vertex]:
        return [i for i, l in enumerate(self.partition_map) if l[0] <= vertex <= l[1]]

    def __repr__(self):
        return repr(self.partition_map)


def load_graph_from_edges(edges: List[RawEdge]) -> nx.MultiDiGraph:
    return nx.parse_edgelist(edges, nodetype=int, create_using=nx.MultiDiGraph)


def relabel_samples(samples):
    prefixes = [l for l in string.ascii_lowercase]
    i = 2
    while len(prefixes) < len(samples):
        prefixes.extend(map(lambda n: "".join(n), itertools.permutations(string.ascii_lowercase, i)))
        i += 1
    for i in range(len(samples)):
        prefix = prefixes[i]
        samples[i] = nx.relabel_nodes(samples[i], lambda n: prefix + str(n))
