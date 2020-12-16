import click

from dgraph_scaler import scaler


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.argument("scale_factor", type=float)
@click.option('-b', '--bridges',
              help="Edges amount to interconnect samples. It is specified in percentage. e.g. 0.1 for 10%.",
              default=0.1, type=float)
@click.option('-fs', '--sampling-factor',
              help="The percentage of vertices that will contain every sample. It is specified in percentage. e.g. 0.5 for 50%.",
              default=0.5, type=float)
@click.option('-p', '--precision',
              help="The minimum percentage of vertices that will be sampled compared to the expected amount. It is specified in percentage. 0.95 for 95%.",
              default=0.95, type=float)
@click.option('-c', '--connect', is_flag=True, )
@click.option('-st', '--stitching-type', help="Stitching topology used for connecting samples.", default="all-to-all",
              type=str)
@click.option('-nfs', '--merge-nfs', help="Flag indicating if the output will be splitted in multiple files.",
              is_flag=True, )
@click.option('-v', '--verbose', help="Flag for printing program progress.", is_flag=True)
def distributed_sampling(*args, **kwargs):
    scaler.scale(*args, **kwargs)


if __name__ == "__main__":
    distributed_sampling()

"""
Command example:
>  python3.6 main.py datasets/ordered/facebook.edges samples/facebook 2.5
"""
