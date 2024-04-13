import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.state import State, Setter

def test_state():
    """Test the state."""

    app = MagicMock()
    gom = MagicMock()
    gom.page.return_value = 42
    cursor = MagicMock()
    bus = MagicMock()

    state = State(app, bus, gom, cursor)

    assert state.current_page() == 42, "Failed to get current page"

    state.mode("draw")
    assert state.mode() == 'draw', "Failed to set mode"

    state.modified(True)
    assert state.modified() == True, "Failed to set modified"

    assert state.cursor() == cursor, "Failed to get cursor"

    state.current_obj(42)
    assert state.current_obj() == 42, "Failed to set current object"

    state.current_obj_clear()
    assert state.current_obj() == None, "Failed to clear current object"

    state.hover_obj(42)
    assert state.hover_obj() == 42, "Failed to set hover object"

    state.hover_obj_clear()
    assert state.hover_obj() == None, "Failed to clear hover object"

    state.show_grid()
    state.toggle_grid()

    state.show_wiglets(True)
    assert state.show_wiglets() == True, "Failed to set show wiglets"
    state.toggle_wiglets()
    assert state.show_wiglets() == False, "Failed to toggle show wiglets"

    assert state.alpha() == 0, "Failed to get transparency"
    state.cycle_background()
    assert state.alpha() == 0.5, "Failed to get transparency"
    state.cycle_background()
    assert state.alpha() == 1, "Failed to get transparency"
    state.cycle_background()
    assert state.alpha() == 0, "Failed to get transparency"
    state.alpha(0.5)
    assert state.alpha() == 0.5, "Failed to set transparency"

    assert state.outline() == False, "Failed to get outline"
    state.outline_toggle()
    assert state.outline() == True, "Failed to toggle outline"

    assert state.bg_color() == (.8, .75, .65), "Failed to get bg color"
    state.bg_color((1, 1, 1))
    assert state.bg_color() == (1, 1, 1), "Failed to set bg color"

    assert state.hidden() == False, "Failed to get hidden"
    state.hidden(True)
    assert state.hidden() == True, "Failed to set hidden"

def test_state_backcalls():
    """Test the state."""
    app = MagicMock()
    gom = MagicMock()
    gom.page.return_value = 42
    cursor = MagicMock()
    bus = MagicMock()

    app.get_size.return_value = (140, 410)
    

    state = State(app, bus, gom, cursor)
    assert state.get_win_size() == (140, 410), "Failed to get window size"
    state.queue_draw()
    assert app.queue_draw.called, "Failed to queue draw"

@patch('sd.state.Pen')
def test_state_pen(mock_pen_class):
    """Test the state."""

    app = MagicMock()
    gom = MagicMock()
    gom.page.return_value = 42
    cursor = MagicMock()
    bus = MagicMock()

    mock_pen_instance = MagicMock()
    mock_pen_instance.color = (1, 0, 0)
    mock_pen_class.return_value = mock_pen_instance

    state = State(app, bus, gom, cursor)
    assert mock_pen_class.call_count == 2, "Failed to create state correctly"
    pen = state.pen()
    assert pen == mock_pen_instance, "Failed to get pen"
    state.apply_pen_to_bg()
    assert state.bg_color() == (1, 0, 0), "Failed to apply pen to bg color"


@patch('sd.state.Pen')
def test_setter(mock_pen_class):
    """Test the setter class"""

    app = MagicMock()
    gom = MagicMock()
    gom.page.return_value = 42
    selection = MagicMock()
    gom.selection.return_value = selection

    cursor = MagicMock()
    state = MagicMock()

    pen = MagicMock()
    pen.color = (1, 0, 0)
    mock_pen_class.return_value = pen

    state.pen.return_value = pen
    
    obj = MagicMock()
    obj.type = "text"
    obj.pen = MagicMock()
    obj.pen.font_size = "xxx"
    obj.strlen.return_value = 0
    state.current_obj.return_value = obj

    setter = Setter(app, gom, cursor, state)

    setter.set_font("42")
    assert state.pen.called, "Failed to set font"
    assert pen.font_set_from_description.called_with("42"), "Failed to set font"

    setter.set_brush("34")
    assert state.pen.called, "Failed to set brush"
    assert pen.brush_type.called_with("34"), "Failed to set brush"

    setter.set_color("12")
    assert state.pen.called, "Failed to set color"
    assert pen.color_set.called_with("12"), "Failed to set color"

    setter.stroke_change("122")
    assert state.current_obj.called, "Failed to change stroke"
    assert obj.stroke_change.called_with("122"), "Failed to change stroke"
    assert pen.font_size == "xxx", "Failed to change stroke"

    setter.clear()
    assert selection.clear.called, "Failed to clear selection"
    assert state.current_obj_clear.called, "Failed to clear current object"
    assert gom.remove_all.called, "Failed to remove all objects"
    assert state.queue_draw.called, "Failed to queue draw"

