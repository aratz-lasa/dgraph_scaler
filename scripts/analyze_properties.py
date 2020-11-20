import os

import click
import networkx as nx


@click.command()
@click.argument("input_folder")
@click.option("-e", "--extension", default=None, help="Extension of files that contain edges")
def load_analyze_properties(input_folder, extension):
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
    print(f"Average degree: {graph.number_of_edges()/ graph.number_of_nodes()}")
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
