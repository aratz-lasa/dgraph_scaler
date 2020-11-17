import click

from dgraph_scaler import distributor, sampler, util, mpi


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
def distributed_sampling(input_file, output_file, scale_factor):
    # Step 1: Read distribute edges and load graph
    edges, partition_map = distributor.distribute_edges(input_file)
    print(f"{mpi.rank} {partition_map}")
    graph = util.load_graph_from_edges(edges)
    edges = None  # Free memory
    # Step 2: Split factor into sample rounds
    factors = [0.5 for _ in range(int(scale_factor) * 2)] + [scale_factor - int(scale_factor)]
    # Step 3: Run dsitributed sampling
    samples = []
    for factor in factors:
        samples.append(sampler.sample(graph, factor, partition_map))
    # Step 4: Locally stitch samples
    # Step 5: Merge distributed samples into master file
    pass


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
> python3.6 main.py datasets/ordered/facebook.edges samples/facebook.sample 2.5
"""
