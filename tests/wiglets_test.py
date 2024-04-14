import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.wiglets import *

def _make_event(ctrl = False, shift = False, alt = False, double = False, mode = "move", pos = (0, 0),
                hover = False, corner = True):

    # preparing the event magic
    ev = MagicMock()
    ev.mode.return_value = mode
    ev.double.return_value = double

    if corner:
        corner_obj = MagicMock()
        corner_obj.bbox.return_value = (0, 0, 10, 10)
        ev.corner.return_value = (corner_obj, "upper-left")
    else:
        ev.corner.return_value = (None, None)

    if hover:
        hover_obj = MagicMock()
        hover_obj.bbox.return_value = (0, 0, 10, 10)
        ev.hover.return_value = hover_obj
    else:
        ev.hover.return_value = None

    ev.ctrl.return_value = ctrl
    ev.shift.return_value = shift
    ev.alt.return_value = alt
    ev.pos.return_value = pos

    ev.event = MagicMock()
    ev.event.x = pos[0]
    ev.event.y = pos[1]

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


def test_hover():    

    bus = MagicMock()
    state = MagicMock()
    cursor = MagicMock()
    state.cursor.return_value = cursor

    hover = WigletHover(bus, state)
    assert bus.on.call_count == 1, "Failed to call on"

    ev = _make_event()

    hover.on_move(ev)
    assert cursor.set.called_with("upper-left"), "Failed to call cursor set"
    assert bus.emit.call_count == 1, "Failed to call emit"

    ev = _make_event(hover = True, corner = False)
    hover.on_move(ev)
    assert cursor.set.called_with("move"), "Failed to call cursor set"
    assert bus.emit.call_count == 2, "Failed to call emit"
    assert state.hover_obj.called, "Failed to call hover_obj"

@patch('sd.wiglets.MoveCommand', return_value=None)
def test_move(cmd_class):

    cmd = MagicMock()
    cmd_class.return_value = cmd

    bus = MagicMock()

    selection = MagicMock()
    selection.set.return_value = True

    gom = MagicMock()
    gom.selection.return_value = selection

    state = MagicMock()
    state.selection.return_value = selection
    state.gom.return_value = gom
    state.get_win_size.return_value = (100, 100)

    cursor = MagicMock()
    state.cursor.return_value = cursor

    move = WigletMove(bus, state)
    assert bus.on.call_count == 3, "Failed to call on"

    ev = _make_event(hover = True, corner = False)
    move.on_click(ev)
    assert bus.emit.call_count == 1, "Failed to call emit"
    assert state.selection.call_count == 1, "Failed to call set"
    assert ev.hover.called, "Failed to call hover"
    assert cmd_class.called, "Failed to call move_command"
    assert cursor.set.called_with("grabbing"), "Failed to call cursor set"

    ev2 = _make_event(pos = (5, 5))
    move.on_move(ev2)
    assert cmd.event_update.called_with(5, 5), "Failed to call move_command update"
    assert bus.emit.call_count == 2, "Failed to call emit on move"

    ev3 = _make_event(pos = (15, 15))
    move.on_release(ev3)
    assert cmd.event_update.called_with(15, 15), "Failed to call move_command update"
    assert bus.emit.call_count == 3, "Failed to call emit on move"
    assert bus.emit.call_with("queue_draw"), "Failed to call emit queue_draw"
    assert state.get_win_size.called, "Failed to call get_win_size"
    assert gom.command_append.called, "Failed to call command_append"
    assert cursor.revert.called, "Failed to call cursor revert"

