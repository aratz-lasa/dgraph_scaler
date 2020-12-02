import os

import click
import networkx as nx
import snap


@click.command()
@click.argument("input_path")
@click.option("-e", "--extension", default=None, help="Extension of files that contain edges")
@click.option("-nx", "--use-networkx", is_flag=True)
def load_analyze_properties(input_path, extension, use_networkx):
    if use_networkx:
        analyze_with_networkx(extension, input_path)
    else:
        analyze_with_sanppy(extension, input_path)


def analyze_with_sanppy(extension, input_path):
    id_map = {}
    next_id = 0
    graph = snap.PNGraph.New()
    files_read = 0
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if not extension or file.endswith(extension):
                files_read += 1
                next_id = load_graph_from_file(os.path.join(input_path, file), graph, id_map, next_id)
    else:
        load_graph_from_file(input_path, graph, id_map, next_id)
    print("Nodes: {}".format(snap.CntNonZNodes(graph)))
    print("Edges: {}".format(snap.CntUniqDirEdges(graph)))
    print("Connected: {}".format(snap.IsConnected(graph)))
    print("Average degree: {}".format(snap.CntNonZNodes(graph) / snap.CntUniqDirEdges(graph)))
    print("Average clustering coefficient: {}".format(snap.GetClustCf(graph)))
    ef_diam_l, ef_diam_h, diam, sp = snap.GetBfsEffDiamAll(graph, 10, True)
    print("Average shortest-path: {}".format(sp))
    print("Diameter: {}".format(diam))
    print("Density: {}".format(snap.CntUniqDirEdges(graph)/(snap.CntNonZNodes(graph)*snap.CntNonZNodes(graph))))
    # TODO: a more efficient diameter: print(f"Diameter: {nx.diameter(undir_graph)}")


def load_graph_from_file(input_path, graph, id_map, next_id):
    with open(input_path) as file:
        for line in file.readlines():
            edge = line.rstrip().split()
            n1 = id_map.get(edge[0], next_id)
            if n1 == next_id:
                id_map[edge[0]] = n1
            next_id += 1
            try:
                n2 = id_map.get(edge[1], next_id)
            except IndexError:
                n2 = id_map.get(edge[0], next_id)
            if n2 == next_id:
                id_map[edge[1]] = n2
            next_id += 1
            if not graph.IsNode(n1):
                graph.AddNode(n1)
            if not graph.IsNode(n2):
                graph.AddNode(n2)
            graph.AddEdge(n1, n2)
    return next_id


def analyze_with_networkx(extension, input_path):
    graph = nx.MultiDiGraph()
    files_read = 0
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if not extension or file.endswith(extension):
                files_read += 1
                graph = nx.compose(graph, nx.read_edgelist(os.path.join(input_path, file), create_using=nx.MultiDiGraph))
    else:
        graph = nx.read_edgelist(input_path, create_using=nx.MultiDiGraph)

    undir_graph = nx.MultiGraph(graph)
    unmul_graph = nx.DiGraph(graph)
    print("Nodes: {}".format(graph.number_of_nodes()))
    print("Edges: {}".format(graph.number_of_edges()))
    print("Connected: {}".format(nx.is_connected(undir_graph)))
    print("Average degree: {}".format(graph.number_of_edges() / graph.number_of_nodes()))
    print("Average shortest-path: {}".format(nx.average_shortest_path_length(graph)))
    print("Average clustering coefficient: {}".format(nx.average_clustering(unmul_graph)))
    print("SCC: {}".format(nx.number_strongly_connected_components(graph)))
    print("WCC: {}".format(nx.number_weakly_connected_components(graph)))
    print("Density: {}".format(nx.density(undir_graph)))
    # TODO: a more efficient diameter: print(f"Diameter: {nx.diameter(undir_graph)}")


if __name__ == "__main__":
    load_analyze_properties()

"""
Command example:
> python3.6 scripts/analyze_properties.py samples/facebook
"""
