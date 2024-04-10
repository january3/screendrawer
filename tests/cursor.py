import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.cursor import CursorManager

@patch('sd.cursor.Gdk.Cursor.new_from_name')
@patch('sd.cursor.Gdk.Cursor.new_from_pixbuf')
def test_cursor_manager(mock_nfp, mock_nfn):
    """Test the cursor manager"""

    retva = MagicMock()
    mock_nfp.return_value = retva
    mock_nfn.return_value = retva

    window = MagicMock()
    window.get_display.return_value = MagicMock()
    win = MagicMock()
    win.set_cursor.return_value = MagicMock()
    window.get_window.return_value = win

    cm = CursorManager(window)
    assert cm is not None, "Creating cursor manager failed"

    cm.update_pos(10, 10)
    assert cm.pos() == (10, 10), "Failed to update cursor position"

    cm.default("hand")
    win.set_cursor.assert_called_once()
    assert cm._CursorManager__current_cursor == "hand", "Failed to set default cursor"
    assert cm._CursorManager__default_cursor == "hand", "Failed to set default cursor"

    cm.set("move")
    cm.set("move")
    win.set_cursor.assert_called()
    assert cm._CursorManager__current_cursor == "move", "Failed to set default cursor"
    assert cm._CursorManager__default_cursor == "hand", "Failed to set default cursor"

    cm.revert()
    win.set_cursor.assert_called()
    assert cm._CursorManager__current_cursor == "hand", "Failed to set default cursor"
    assert cm._CursorManager__default_cursor == "hand", "Failed to set default cursor"
    cm.revert()
