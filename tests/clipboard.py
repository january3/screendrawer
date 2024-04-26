from unittest import mock
from unittest.mock import patch, MagicMock

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk, Gdk # <remove>

from sd.clipboard import Clipboard
from sd.drawable_group import DrawableGroup # <remove>
from sd.utils import img_object_copy # <remove>

def test_create_clipboard():

    cl = Clipboard()
    assert cl is not None, "Creating clipboard failed"
    assert isinstance(cl, Clipboard), "Creating clipboard failed"
    test_txt = "Test text"
    cl.set_text(test_txt)

    ctype, ctext = cl.get_content()
    assert ctype == "text" and ctext == test_txt, "Clipboard does not work"

    selection = MagicMock()
    selection.type = "text"
    selection.to_string.return_value = test_txt
    cl.copy_content(selection)

    ctype, ctext = cl.get_content()
    assert ctype == "internal", "Clipboard does not work"
    assert isinstance(ctext, DrawableGroup), "Expected DrawableGroup, got something else"
