import os

import click


@click.command()
@click.argument("input_folder")
@click.argument("output_file")
@click.option("-e", "--extension", default=".*", help="Extension of files that contain edges")
def order_merge_edges(input_folder, output_file, extension):
    merged_lines = []
    files_amount = len(list(filter(lambda f: f.endswith(extension), os.listdir(input_folder))))
    files_read = 0
    for file in os.listdir(input_folder):
        if file.endswith(extension):
            files_read+=1
            print(f"Loading {os.path.join(input_folder, file)}. {files_read}/{files_amount}")
            with open(os.path.join(input_folder, file)) as file:
                merged_lines.extend(file.readlines())
    with open(output_file, "w") as file:
        file.write(f"{len(merged_lines)}\n")
        merged_lines.sort(key=lambda line: int(line.split()[0]))
        file.writelines(merged_lines)


if __name__ == "__main__":
    order_merge_edges()

"""
Command example:
> python3.6 scripts/sort_merge_edges.py datasets/facebook datasets/ordered/facebook.edges -e ".edges"
"""