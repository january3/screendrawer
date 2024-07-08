"""A class which holds a set of transformations."""
import logging                                                    # <remove>

log = logging.getLogger(__name__)                                 # <remove>
log.setLevel(logging.DEBUG)                                        # <remove>

# ---------- trafos ----------------------
def trafos_apply(coords, trafos):
    """Apply transformations to a set of coordinates."""

    for trafo in trafos:
        trafo_type, trafo_args = trafo
        if trafo_type == "rotate":
            coords = coords_rotate(coords, trafo_args[2], (trafo_args[0], trafo_args[1]))
        elif trafo_type == "resize":
            #coords = transform_coords(coords, (0, 0, w, h), (0, 0, w * trafo_args[2], h * trafo_args[3]))
            coords = [ (trafo_args[0] + (p[0] - trafo_args[0]) * trafo_args[2],
                        trafo_args[1] + (p[1] - trafo_args[1]) * trafo_args[3]) for p in coords ]
        elif trafo_type == "move":
            coords = [ (p[0] + trafo_args[0], p[1] + trafo_args[1]) for p in coords ]
        elif trafo_type == "dummy":
            pass
        else:
                log.error("unknown trafo %s", trafo_type)

    return coords

def trafos_reverse(coords, trafos):
    """Reverse transformations on a set of coordinates."""

    for trafo in reversed(trafos):
        trafo_type, trafo_args = trafo
        if trafo_type == "rotate":
            coords = coords_rotate(coords, -trafo_args[2], (trafo_args[0], trafo_args[1]))
        elif trafo_type == "resize":
            coords = transform_coords(coords, (0, 0, w, h), (0, 0, w / trafo_args[2], h / trafo_args[3]))
        elif trafo_type == "move":
            coords = [ (p[0] - trafo_args[0], p[1] - trafo_args[1]) for p in coords ]
        elif trafo_type == "dummy":
            pass
        else:
            log.error("unknown trafo %s", trafo_type)

    return coords

def trafos_on_cairo(cr, trafos):
    """Apply transformations to a cairo context."""

    for i, trafo in enumerate(reversed(trafos)):

        trafo_type, trafo_args = trafo

        if trafo_type == "rotate":
            cr.translate(trafo_args[0], trafo_args[1])
            cr.rotate(trafo_args[2])
            cr.translate(-trafo_args[0], -trafo_args[1])
        elif trafo_type == "resize":
            cr.translate(trafo_args[0], trafo_args[1])
            cr.scale(trafo_args[2], trafo_args[3])
            cr.translate(-trafo_args[0], -trafo_args[1])
        elif trafo_type == "move":
            cr.translate(trafo_args[0], trafo_args[1])
        elif trafo_type == "dummy":
            pass
        else:
            log.error("unknown trafo %s [%d]", trafo_type, i)

class Trafo():
    """
    Class to hold a set of transformations.

    Attributes:
        trafo (list): The list of transformations.
    """

    def __init__(self, trafo = None):

        log.debug("initializing with %s", trafo)

        if not trafo:
            self.__trafo = []
        else:
            assert isinstance(trafo, list)
            self.__trafo = trafo

    def n(self):
        """Return the number of transformations."""
        return len(self.__trafo)

    def add_trafo(self, trafo, merge = True):
        """Add a transformation to the list."""
        prev_trafo_type = self.__trafo[-1][0] if self.__trafo else None

        if merge and prev_trafo_type == "move" and trafo[0] == "move":
            x, y = self.__trafo.pop()[1]
            dx, dy = trafo[1]
            self.__trafo.append(("move", (x + dx, y + dy)))
            return

        if merge and prev_trafo_type == "rotate" and trafo[0] == "rotate":
            x0, y0, a0 = self.__trafo.pop()[1]
            x1, y1, a1 = trafo[1]
            if x0 == x1 and y0 == y1:
                self.__trafo.append(("rotate", (x1, y1, a0 + a1)))
                return

        self.__trafo.append(trafo)

    def pop_trafo(self):
        """Remove the last transformation from the list."""
        return self.__trafo.pop()

    def trafos(self):
        """Return the transformations."""
        return self.__trafo

    def apply(self, coords):
        """Apply the transformations to the coordinates."""

        return trafos_apply(coords, self.__trafo)

    def apply_reverse(self, coords):
        """Apply the reverse transformations to the coordinates."""

        return trafos_reverse(coords, self.__trafo)

    def transform_context(self, cr):
        """Transform the cairo context."""

        trafos_on_cairo(cr, self.__trafo)

    def __str__(self):
        return f"Trafo({self.__trafo})"

    def __repr__(self):
        return self.__str__()
