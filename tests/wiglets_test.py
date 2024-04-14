import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.wiglets import *

def _make_event(ctrl = False, shift = False, alt = False, double = False, mode = "move", pos = (0, 0)):

    # preparing the event magic
    ev = MagicMock()
    ev.mode.return_value = mode
    ev.double.return_value = double
    corner_obj = MagicMock()
    corner_obj.bbox.return_value = (0, 0, 10, 10)
    ev.corner.return_value = (corner_obj, "upper-left")

    ev.ctrl.return_value = ctrl
    ev.shift.return_value = shift
    ev.alt.return_value = alt
    ev.pos.return_value = pos

    return ev

@patch('sd.wiglets.ResizeCommand', return_value=None)
def test_resize(cmd_class):

    cmd = MagicMock()
    cmd_class.return_value = cmd

    bus = MagicMock()

    selection = MagicMock()
    selection.set.return_value = True

    gom = MagicMock()
    gom.selection.return_value = selection

    state = MagicMock()
    cursor = MagicMock()
    state.cursor.return_value = cursor

    rr = WigletResizeRotate(bus, gom, state)
    assert bus.on.call_count == 3, "Failed to call on"

    ev = _make_event()
    rr.on_click(ev)

    assert ev.mode.call_count == 1, "Failed to call mode"
    assert bus.emit.call_count == 1, "Failed to call emit"
    # get call arguments to resize_command
    assert cmd_class.call_count == 1, "Failed to call resize_command"
    assert selection.set.call_count == 1, "Failed to call set"

    ev2 = _make_event(pos = (5, 5))
    rr.on_move(ev2)
    assert cmd.event_update.call_count == 1, "Failed to call resize_command update on move"
    cmd.event_update.assert_called_with(*ev2.pos.return_value)
    assert bus.emit.call_count == 2, "Failed to call emit on move"

    ev3 = _make_event(pos = (15, 15))
    rr.on_release(ev3)
    assert cmd.event_update.call_count == 2, "Failed to call resize_command update on move"
    cmd.event_update.assert_called_with(*ev3.pos.return_value)
    assert bus.emit.call_count == 3, "Failed to call emit on move"
    assert cursor.revert.call_count == 1, "Failed to call cursor revert"
    gom.command_append.assert_called_with([cmd])


@patch('sd.wiglets.RotateCommand', return_value=None)
def test_rotate(cmd_class):

    cmd = MagicMock()
    cmd_class.return_value = cmd

    bus = MagicMock()

    selection = MagicMock()
    selection.set.return_value = True

    gom = MagicMock()
    gom.selection.return_value = selection

    state = MagicMock()
    cursor = MagicMock()
    state.cursor.return_value = cursor

    rr = WigletResizeRotate(bus, gom, state)
    assert bus.on.call_count == 3, "Failed to call on"

    ev = _make_event(shift=True, ctrl=True)
    rr.on_click(ev)

    assert ev.mode.call_count == 1, "Failed to call mode"
    assert bus.emit.call_count == 1, "Failed to call emit"
    # get call arguments to resize_command
    assert cmd_class.call_count == 1, "Failed to call resize_command"
    assert selection.set.call_count == 1, "Failed to call set"

    ev2 = _make_event(pos = (5, 5))
    rr.on_move(ev2)
    assert cmd.event_update.call_count == 1, "Failed to call resize_command update on move"
    cmd.event_update.assert_called_with(*ev2.pos.return_value)
    assert bus.emit.call_count == 2, "Failed to call emit on move"

    ev3 = _make_event(pos = (15, 15))
    rr.on_release(ev3)
    assert cmd.event_update.call_count == 2, "Failed to call resize_command update on move"
    cmd.event_update.assert_called_with(*ev3.pos.return_value)
    assert bus.emit.call_count == 3, "Failed to call emit on move"
    assert cursor.revert.call_count == 1, "Failed to call cursor revert"
    gom.command_append.assert_called_with([cmd])


    
