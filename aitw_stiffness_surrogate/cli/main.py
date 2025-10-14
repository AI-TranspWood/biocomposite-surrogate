import click


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(writable=True))
@click.argument('lower_bound_percentage', type=click.FloatRange(0.0, 1.0))
@click.argument('upper_bound_percentage', type=click.FloatRange(0.0, 1.0))
def aitw_stiffness(
        input_file,
        output_file,
        lower_bound_percentage,
        upper_bound_percentage
    ):
    """Command line interface for the AI surrogate model for stiffness prediction of wood."""
    # Import this inside to avoid running tensorflow loading for CLI help or sourcing
    from .. import dataLoaders as dl
    from .. import surrogate

    quantiles = [ lower_bound_percentage, upper_bound_percentage]

    input = dl.loadInput( input_file )

    model = surrogate.createArchitecture( inputSize = 18, outputSize = 5, layerSize = 24)
    estimate = surrogate.computeEstimate( input, model, quantiles = quantiles)

    dl.saveResult( estimate, output_file)

    click.echo(f'Stiffness estimates saved to {output_file}')
