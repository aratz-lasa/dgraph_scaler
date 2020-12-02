import os

import click


@click.command()
@click.argument("input_folder")
@click.argument("output_file")
@click.option("-e", "--extension", default=None, help="Extension of files that contain edges")
@click.option("-na", "--nodes-amount", default=None, type=int)
def order_merge_edges(input_folder, output_file, extension, nodes_amount):
    merged_lines = []
    files_read = 0
    for file in os.listdir(input_folder):
        if not extension or file.endswith(extension):
            files_read += 1
            print(f"Loading {os.path.join(input_folder, file)}")
            with open(os.path.join(input_folder, file)) as file:
                merged_lines.extend(file.readlines())
            print(f"Loading {os.path.join(input_folder, file)}")
    with open(output_file, "w") as file:
        if nodes_amount:
            file.write(f"{nodes_amount}\n")
        file.write(f"{len(merged_lines)}\n")
        merged_lines.sort(key=lambda line: int(line.split()[0]))
        file.writelines(merged_lines)


if __name__ == "__main__":
    order_merge_edges()

"""
Command example:
> python3.6 scripts/sort_merge_edges.py datasets/facebook datasets/ordered/facebook.edges -e ".edges"
"""
