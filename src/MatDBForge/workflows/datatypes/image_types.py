"""Custom data types for image handling in MatDBForge."""
import os
import subprocess as sb
import tempfile

from aiida.orm import SinglefileData


class ImagePNGData(SinglefileData):
    """A new data type that contains an image in PNG format."""

    def __init__(self, filepath, **kwargs):
        """Construct a new instance and set the contents to that of the file.

        :param file: an absolute filepath of the file to wrap
        """
        super().__init__(**kwargs)

        # Get the filename from the absolute path
        filename = os.path.basename(filepath)

        # Store the file in the repository under the given filename
        self.put_object_from_file(filepath, filename)

        # Store in the attributes what the filename is
        self.base.attributes.set('filename', filename)

    def display_image(self):
        """Return the content of the single file stored for this data node.

        :return: the content of the file as a string
        """
        filename = self.base.attributes.get('filename')
        image_data_bytes = self.get_object_content(filename, mode='rb')

        # Write the bytes to a temporary file and return the path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_data_bytes)

            # For Linux systems, use xdg-open to open the image
            sb.call(['xdg-open', temp_file.name])
