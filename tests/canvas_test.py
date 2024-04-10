# Description: Test the canvas module
from unittest import mock
from unittest.mock import patch, MagicMock

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
import cairo                                  # <remove>

from sd.canvas import Canvas


def test_canvas():
    """Test the canvas module"""
    width, height = 100, 100

    state   = MagicMock()
    state.bg_color.return_value = (1, 1, 1)
    state.transparency.return_value = .5
    state.show_grid.return_value = False
    state.get_win_size.return_value = (width, height)

    dm      = MagicMock()
    wiglets = MagicMock()

    # just test whether we can create the object
    canvas = Canvas(state, dm, wiglets)
    assert canvas is not None
    assert isinstance(canvas, Canvas)

    surface = cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1)
    cr = cairo.Context(surface)

    canvas.draw_bg(cr, (0, 0))
    canvas.draw_bg(cr, (10, 10))

    state.show_grid.return_value = True
    canvas.draw_bg(cr, (0, 0))
    canvas.draw_bg(cr, (10, 10))

