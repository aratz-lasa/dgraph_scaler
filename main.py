import time

import click

from dgraph_scaler import distributor, sampler, util, mpi, stitcher, merger


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
@click.option('-bn', '--bridges-number', default=1, type=int)
def distributed_sampling(input_file, output_file, scale_factor, bridges_number):
    start_time = time.time()

    # Step 1: Read distribute edges and load graph
    edges, partition_map = distributor.distribute_edges(input_file)
    graph = util.load_graph_from_edges(edges)
    edges = None  # Free memory
    # Step 2: Split factor into sample rounds
    factors = [0.5 for _ in range(int(scale_factor) * 2)] + [scale_factor - int(scale_factor)]
    # Step 3: Run dsitributed sampling
    samples = []
    for i, factor in enumerate(factors):
        if mpi.rank == 0:
            print(f"Sampling {i}/{len(factors)}")
        samples.append(sampler.sample(graph, factor, partition_map))
    # Step 4: Rename vertices
    util.relabel_samples(samples)
    # Step 5: Stitch samples locally and distributively
    stitcher.stitch_samples(samples, bridges_number)
    # Step 5: Merge distributed samples into master file
    merger.dump_samples(samples, output_file)

    print(time.time() - start_time)


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
>  python3.6 main.py datasets/ordered/facebook.edges samples/facebook 2.5
"""
