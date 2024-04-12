import pytest
# mock
from unittest.mock import patch, MagicMock
from sd.bus import Bus


def test_bus():
    """Test the bus class."""

    listener = MagicMock()
    listener.return_value = True

    bus = Bus()
    assert bus, "Bus instance not created."

    bus.on("event", listener)
    assert bus.emit("event"), "Listener not called."

    bus.off("event", listener)
    assert not bus.emit("event"), "Listener called."

    listener.return_value = False
    bus.on("event", listener)
    assert not bus.emit("event"), "Listener called."

def test_bus_exclusive():
    """Test the bus class with exclusive events."""

    listener = MagicMock()
    listener.return_value = True
    listener2 = MagicMock()
    listener.return_value = True

    bus = Bus()
    assert bus, "Bus instance not created."
    bus.on("event", listener)
    bus.on("event", listener2)

    reply = bus.emit("event", exclusive = True)
    assert reply, "No listener called."
    assert listener.called, "Listener 1 not called."
    assert not listener2.called, "Listener 2 called."
