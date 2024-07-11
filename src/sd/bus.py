"""An event bus class for dispatching events between objects."""
import logging                                                   # <remove>
import traceback                                                 # <remove>
from sys import exc_info                                         # <remove>
log = logging.getLogger(__name__)                                # <remove>
log.setLevel(logging.INFO)                                       # <remove>

class Bus:
    """A simple event bus for dispatching events between objects."""

    def __init__(self):
        self.__listeners = {}

    def on(self, event, listener, priority = 0):
        """Add a listener for an event."""
        if listener is None:
            raise ValueError("Listener cannot be None")

        if not callable(listener):
            raise ValueError("Listener must be callable")

        if event is None:
            raise ValueError("Event cannot be None")

        if event not in self.__listeners:
            self.__listeners[event] = []

        self.__listeners[event].append((listener, priority))
        self.__listeners[event].sort(key = lambda x: -x[1])

    def off(self, event, listener):
        """Remove a listener for an event."""
        if event in self.__listeners:
            self.__listeners[event][:] = [x for x in self.__listeners[event] if x[0] != listener]

    def call(self, listener, event, include_event, args, kwargs):
        """Call the listener with the specified arguments."""
        try:
            if include_event:
                ret = listener(event, *args, **kwargs)
            else:
                ret = listener(*args, **kwargs)
        except Exception: #pylint: disable=broad-except
            ret = None
            exc_type, exc_value, exc_traceback = exc_info()
            log.warning("Traceback:")
            traceback.print_tb(exc_traceback)
            log.warning("Exception value: %s", exc_value)
            log.error("Exception type: %s", exc_type)
            log.warning("Error while dispatching signal %s to %s:", event, listener)
        return ret

    def emit_once(self, event, *args, **kwargs):
        """Emit an exclusive event - stops dispatching if a listener returns a truthy value."""

        return self.emit(event, True, *args, **kwargs)

    def emit_mult(self, event, *args, **kwargs):
        """Emit a non-exclusive event - dispatches to all listeners regardless of return value."""

        return self.emit(event, False, *args, **kwargs)

    def emit(self, event, exclusive = False, *args, **kwargs):
        """
        Dispatch an event to all listeners.

        Exclusive events will stop dispatching if a listener returns a truthy value.
        """

        log.debug("emitting event %s exclusive=%s with %s and %s",
                  event, exclusive, args, kwargs)

        # completely ignore events that have no listeners
        if not event in self.__listeners:
            return False

        # call the promiscous listeners first, but they don't stop the event
        for listener, _ in self.__listeners.get('*', []):
            ret = self.call(listener, event, True, args, kwargs)

        caught = False
        for listener, _ in self.__listeners.get(event, []):
            ret = self.call(listener, event, False, args, kwargs)
            if ret:
                caught = True
                if exclusive:
                    return ret

        return caught
