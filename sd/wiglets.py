from .pen import Pen               # <remove>

## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, type, coords):
        self.wiglet_type   = type
        self.coords = coords

    def draw(self, cr):
        raise NotImplementedError("draw method not implemented")

    def event_update(self, x, y):
        raise NotImplementedError("event_update method not implemented")

    def event_finish(self):
        raise NotImplementedError("event_finish method not implemented")

class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, coords, pen):
        super().__init__("transparency", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")

        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_transparency = pen.transparency
        print("initial transparency:", self._initial_transparency)

    def draw(self, cr):
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        #print("changing transparency", dx)
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.pen.transparency = max(0, min(1, self._initial_transparency + dx/500))
        #print("new transparency:", self.pen.transparency)

    def event_finish(self):
        pass

class WigletLineWidth(Wiglet):
    """Wiglet for changing the line width."""
    """directly operates on the pen of the object"""
    def __init__(self, coords, pen):
        super().__init__("line_width", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")
        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_width = pen.line_width

    def draw(self, cr):
        cr.set_source_rgb(*self.pen.color)
        draw_dot(cr, *self.coords, self.pen.line_width)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing line width", dx)
        self.pen.line_width = max(1, min(60, self._initial_width + dx/20))
        return True

    def event_finish(self):
        pass

## ---------------------------------------------------------------------


