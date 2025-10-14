import click

from .. import dataLoaders as dl
from .. import surrogate


@click.command()
@click.argument(
    'input_file', type=click.Path(exists=True),
    help='Path to the input file containing the features for stiffness prediction'
)
@click.argument(
    'output_file', type=click.Path(),
    help='Path to the output file where the stiffness estimates will be saved'
)
@click.argument(
    'lower_bound_percentage', type=click.FloatRange(0.0, 1.0),
    help='Lower bound percentage of the uncertainty estimate'
)
@click.argument(
    'upper_bound_percentage', type=click.FloatRange(0.0, 1.0),
    help='Upper bound percentage of the uncertainty estimate'
)
def aitw_stiffness(
        input_file,
        output_file,
        lower_bound_percentage,
        upper_bound_percentage
    ):
    """Command line interface for the AI surrogate model for stiffness prediction of wood."""
    quantiles = [ lower_bound_percentage, upper_bound_percentage]

    input = dl.loadInput( input_file )

    model = surrogate.createArchitecture( inputSize = 18, outputSize = 5, layerSize = 24)
    estimate = surrogate.computeEstimate( input, model, quantiles = quantiles)

    dl.saveResult( estimate, output_file)
