from typing import List

import networkx as nx

from dgraph_scaler.typing import RawEdge


def load_graph_from_edges(edges: List[RawEdge]) -> nx.DiGraph:
    return nx.parse_edgelist(edges, nodetype=int, create_using=nx.DiGraph)
