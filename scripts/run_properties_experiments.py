import csv
import inspect
import os
import shutil
import sys

import click
import snap

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dgraph_scaler import scaler, mpi


@click.command()
@click.argument("input_file")
@click.argument("results_folder")
@click.argument("output_file")
@click.argument("measurements", type=int)
@click.option("-s", "--scaling-factors", type=float, multiple=True)
@click.option("-e", "--extension", default=None, help="Extension of files that contain edges")
@click.option("-a", "--append", is_flag=True)
@click.option('-b', '--bridges', default=0.1, type=float)
@click.option('-fs', '--factor-size', default=0.5, type=float)
@click.option('-p', '--precision', default=0.95, type=float)
@click.option('-c', '--connect', is_flag=True, )
@click.option('-nfs', '--merge-nfs', is_flag=True, )
@click.option('-v', '--verbose', is_flag=True)
def run_properties_experiments(input_file, results_folder, output_file, measurements, scaling_factors, extension,
                               append, bridges, factor_size, precision, connect, merge_nfs, verbose):
    if mpi.rank == 0:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            clean_folder(results_folder)
        file = open(output_file, "a" if append else "w")
        csv_writer = csv.DictWriter(file, fieldnames=get_headers())
        if not append:
            csv_writer.writeheader()
    for factor in scaling_factors:
        for i in range(measurements):
            if mpi.rank == 0:
                print(f"Running experiment {factor} - {i + 1}/{measurements}")
            scaler.scale(input_file, results_folder, factor, bridges, factor_size, precision, connect, merge_nfs,
                         verbose)
            if mpi.rank == 0:
                row = get_properties_with_sanppy(extension, results_folder)
                row[csv_writer.fieldnames[0]] = factor
                csv_writer.writerow(row)
                file.flush()
                clean_folder(results_folder)

    if mpi.rank == 0:
        file.close()


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_headers():
    return ["Scaling-factor"] + get_property_names()


def get_property_names():
    return ["Nodes", "Edges", "Connected", "Average-degree", "Average-clustering", "Average-sp",
            "Diameter", "Density"]


def get_properties_with_sanppy(extension, input_folder):
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
    ef_diam_l, ef_diam_h, diam, sp = snap.GetBfsEffDiamAll(graph, 10, True)
    properties = [snap.CntNonZNodes(graph), snap.CntUniqDirEdges(graph),
                  snap.IsConnected(graph), snap.CntNonZNodes(graph) / snap.CntUniqDirEdges(graph),
                  snap.GetClustCf(graph), sp, diam,
                  snap.CntUniqDirEdges(graph) / (snap.CntNonZNodes(graph) * snap.CntNonZNodes(graph))]
    return dict(zip(get_property_names(), properties))


if __name__ == "__main__":
    run_properties_experiments()

"""
Command example:
> python3.6 scripts/analyze_properties.py samples/facebook
"""
