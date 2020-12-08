import click

from dgraph_scaler import scaler


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
@click.option('-b', '--bridges', default=0.1, type=float)
@click.option('-fs', '--sampling-factor', default=0.5, type=float)
@click.option('-p', '--precision', default=0.95, type=float)
@click.option('-c', '--connect', is_flag=True, )
@click.option('-st', '--stitching-type', default="all-to-all", type=str)
@click.option('-nfs', '--merge-nfs', is_flag=True, )
@click.option('-v', '--verbose', is_flag=True)
def distributed_sampling(*args, **kwargs):
    scaler.scale(*args, **kwargs)


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
>  python3.6 main.py datasets/ordered/facebook.edges samples/facebook 2.5
"""
