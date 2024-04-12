import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.history import History

def test_history():

    cmd = MagicMock()
    cmd.undo.return_value = 42
    cmd.redo.return_value = 43

    history = History()

    history.add(cmd)

    assert history.length() == 1

    assert history.undo() == 42
    assert cmd.undo.call_count == 1
    assert history.length() == 0

    assert history.redo() == 43
    assert cmd.redo.call_count == 1
    assert history.length() == 1
