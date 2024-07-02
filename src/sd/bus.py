"""An event bus class for dispatching events between objects."""
import logging                                                   # <remove>
import traceback                                                 # <remove>
from sys import exc_info                                         # <remove>
log = logging.getLogger(__name__)                                # <remove>
log.setLevel(logging.INFO)

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
    
    def emit(self, event, exclusive = False, *args, **kwargs):
        """
        Dispatch an event to all listeners.

        Exclusive events will stop dispatching if a listener returns a truthy value.
        """

        log.debug(f"emitting event {event} with {args} and {kwargs}")
        caught = False
        if event in self.__listeners:
            for listener, _ in self.__listeners[event]:
                #print("event", event, "calling", listener, "with", args, kwargs)
                try:
                    ret = listener(*args, **kwargs)
                except Exception as e:
                    ret = None
                    exc_type, exc_value, exc_traceback = exc_info()
                    log.warning(f"Exception type: {exc_type}")
                    log.warning(f"Exception value:{exc_value}")
                    log.warning("Traceback:")
                    traceback.print_tb(exc_traceback)
                    log.warning(f"Error while dispatching signal {event} to {listener}: {e}")

                if ret:
                    #print("Event", event, "caught by:", listener)
                    caught = True
                    if exclusive:
                        return ret
        return caught
