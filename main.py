from dgraph_scaler import distributor
import click


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor")
def distributed_sampling(input_file, output_file, scale_factor):
    # TODO
    # Step 1: Read distribute edges
    edges = distributor.distribute_edges(input_file)
    # Step 2: Split factor into sample rounds
    # Step 3: Run dsitributed sampling
        # Step 3.0: While rounds
            # Step 3.1: Run local random edge sampling
            # Step 3.2: Run distrbuted induction
    # Step 4: Locally stitch samples
    # Step 5: Merge distributed samples into master file
    pass


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
> python3.6 main.py datasets/ordered/facebook.edges samples/facebook.sample 2.5
"""
