"""AiiDA CLI commands for managing custom image data types for MatDBForge workflows."""

import click
from aiida.cmdline.params import arguments, types
from aiida.cmdline.utils import decorators
from aiida.plugins import DataFactory

# Load the data class
ImagePNGData = DataFactory('mdb.img-png')


@click.group('mdb.img-png')
def cli():
    """PNG format images resulting from the MatDBForge workflow."""
    pass


@cli.command('show')
@arguments.NODE(type=types.NodeParamType(sub_classes=('aiida.data:mdb.img-png',)))
@decorators.with_dbenv()
def show(node):
    """
    View the image with the default image viewer.

    Parameters
    ----------
    node : ImagePNGData
        The AiiDA node containing the image to display.
    """
    click.echo(f'Opening image from node <{node.pk}>...')
    node.display_image()
    # click.launch(temp_path)
