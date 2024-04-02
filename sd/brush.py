"""Class for different brushes."""

class Brush:
    """Base class for brushes."""
    def __init__(self):
        self.__outline = [ ]
        self.__outline_l = [ ]
        self.__outline_r = [ ]
        self.__coords = [ ]

    def recalc(self, coords):
        """Recalculate brush outline."""
        self.__outline = coords
