"""
Very simple class that holds the pixbuf along with some additional
information.
"""

import tempfile                          # <remove>
import base64                            # <remove>
from .utils import base64_to_pixbuf  # <remove>

class ImageObj:
    """Simple class to hold an image object."""
    def __init__(self, pixbuf, base64_enc):

        if base64_enc:
            self.__base64 = base64_enc
            pixbuf = base64_to_pixbuf(base64_enc)
        else:
            self.__base64 = None

        self.__pixbuf = pixbuf
        self.__size = (pixbuf.get_width(), pixbuf.get_height())

    def pixbuf(self):
        """Return the pixbuf."""
        return self.__pixbuf

    def size(self):
        """Return the size of the image."""
        return self.__size

    def encode_base64(self):
        """Encode the image to base64."""
        with tempfile.NamedTemporaryFile(delete = True) as temp:
            self.__pixbuf.savev(temp.name, "png", [], [])
            with open(temp.name, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64

    def base64(self):
        """Return the base64 encoded image."""
        if self.__base64 is None:
            self.__base64 = self.encode_base64()
        return self.__base64
