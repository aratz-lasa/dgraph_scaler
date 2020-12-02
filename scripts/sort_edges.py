import os

import click


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("vertices_amount")
@click.argument("edges_amount")
def order_merge_edges(input_file, output_file, vertices_amount, edges_amount):
    with open(input_file) as file:
        edges = file.readlines()
    with open(output_file, "w") as file:
        file.write(f"{vertices_amount}\n")
        file.write(f"{edges_amount}\n")
        edges.sort(key=lambda line: int(line.split()[0]))
        file.writelines(edges)


if __name__ == "__main__":
    order_merge_edges()

"""
Command example:
> python3.6 scripts/sort_edges.py datasets/facebook datasets/ordered/facebook.edges -e ".edges"
"""
