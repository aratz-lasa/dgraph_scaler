import os

import click
import networkx as nx
import snap


@click.command()
@click.argument("input_folder")
@click.option("-e", "--extension", default=None, help="Extension of files that contain edges")
@click.option("-nx", "--use-networkx", is_flag=True)
def load_analyze_properties(input_folder, extension, use_networkx):
    if use_networkx:
        analyze_with_networkx(extension, input_folder)
    else:
        analyze_with_sanppy(extension, input_folder)


def analyze_with_sanppy(extension, input_folder):
    id_map = {}
    next_id = 0
    graph = snap.PNGraph.New()
    files_read = 0
    for file in os.listdir(input_folder):
        if not extension or file.endswith(extension):
            files_read += 1
            with open(os.path.join(input_folder, file)) as file:
                file.readline()  # Read nodes amount
                file.readline()  # Read edges amount
                for line in file.readlines():
                    edge = line.rstrip().split()
                    n1 = id_map.get(edge[0], next_id)
                    if n1 == next_id:
                        id_map[edge[0]] = n1
                    next_id += 1
                    n2 = id_map.get(edge[1], next_id)
                    if n2 == next_id:
                        id_map[edge[1]] = n2
                    next_id += 1
                    if not graph.IsNode(n1):
                        graph.AddNode(n1)
                    if not graph.IsNode(n2):
                        graph.AddNode(n2)
                    graph.AddEdge(n1, n2)
    print(f"Nodes: {snap.CntNonZNodes(graph)}")
    print(f"Edges: {snap.CntUniqDirEdges(graph)}")
    print(f"Connected: {snap.IsConnected(graph)}")
    print(f"Average degree: {snap.CntNonZNodes(graph) / snap.CntUniqDirEdges(graph)}")
    print(f"Average clustering coefficient: {snap.GetClustCf(graph)}")
    ef_diam_l, ef_diam_h, diam, sp = snap.GetBfsEffDiamAll(graph, 10, True)
    print(f"Average shortest-path: {sp}")
    print(f"Diameter: {diam}")
    print(f"Effective diameter: {ef_diam_l} - {ef_diam_h}")
    # TODO: a more efficient diameter: print(f"Diameter: {nx.diameter(undir_graph)}")


def analyze_with_networkx(extension, input_folder):
    graph = nx.MultiDiGraph()
    files_read = 0
    for file in os.listdir(input_folder):
        if not extension or file.endswith(extension):
            files_read += 1
            graph = nx.compose(graph, nx.read_edgelist(os.path.join(input_folder, file), create_using=nx.MultiDiGraph))
    undir_graph = nx.MultiGraph(graph)
    unmul_graph = nx.DiGraph(graph)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Connected: {nx.is_connected(undir_graph)}")
    print(f"Average degree: {graph.number_of_edges() / graph.number_of_nodes()}")
    print(f"Average shortest-path: {nx.average_shortest_path_length(graph)}")
    print(f"Average clustering coefficient: {nx.average_clustering(unmul_graph)}")
    print(f"SCC: {nx.number_strongly_connected_components(graph)}")
    print(f"WCC: {nx.number_weakly_connected_components(graph)}")
    print(f"Density: {nx.density(undir_graph)}")
    # TODO: a more efficient diameter: print(f"Diameter: {nx.diameter(undir_graph)}")


if __name__ == "__main__":
    load_analyze_properties()

"""
Command example:
> python3.6 scripts/analyze_properties.py samples/facebook
"""
