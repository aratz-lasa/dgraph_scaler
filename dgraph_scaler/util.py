import itertools
import random
import string
from bisect import bisect as _bisect
from itertools import accumulate as _accumulate, repeat as _repeat
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
        edges = map(lambda e: (prefix + str(e[0]), prefix + str(e[1])), list(samples[i].edges))
        samples[i] = g = nx.MultiDiGraph()
        g.add_edges_from(edges)


def choices(population, weights=None, *, cum_weights=None, k=1):
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0  # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0  # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]
