import time

import click

from dgraph_scaler import distributor, sampler, util, mpi, stitcher, merger


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
@click.option('-bn', '--bridges-number', default=1, type=int)
@click.option('-fs', '--factor-size', default=0.5, type=float)
def distributed_sampling(input_file, output_file, scale_factor, bridges_number, factor_size):
    total_t = time.time()

    # Step 1: Read distribute edges and load graph
    loading_t = time.time()
    edges, partition_map = distributor.distribute_edges(input_file)
    graph = util.load_graph_from_edges(edges)
    edges = None  # Free memory
    if mpi.rank == 0:
        print("Loading time:", time.time() - loading_t)
    # Step 2: Split factor into sample rounds
    factors = [factor_size for _ in range(int(scale_factor / factor_size))]
    remaining_factor = scale_factor - round(sum(factors), 2)
    if remaining_factor:
        factors.append(remaining_factor)
    # Step 3: Run distributed sampling
    samples = []
    for i, factor in enumerate(factors):
        sampling_t = time.time()
        samples.append(sampler.sample(graph, factor, partition_map))
        if mpi.rank == 0:
            print(f"Sampling time {i + 1}/{len(factors)}:", time.time() - sampling_t)
    # Step 4: Rename vertices
    relabeling_t = time.time()
    util.relabel_samples(samples)
    if mpi.rank == 0:
        print(f"Relabeling time:", time.time() - relabeling_t)
    # Step 5: Stitch samples locally and distributively
    stitching_t = time.time()
    stitcher.stitch_samples(samples, bridges_number)
    if mpi.rank == 0:
        print(f"Stiching time:", time.time() - stitching_t)
    # Step 5: Merge distributed samples into master file
    dumping_t = time.time()
    merger.dump_samples(samples, output_file)
    if mpi.rank == 0:
        print(f"Dumping time:", time.time() - dumping_t)

    if mpi.rank == 0:
        print(time.time() - total_t)


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
>  python3.6 main.py datasets/ordered/facebook.edges samples/facebook 2.5
"""
