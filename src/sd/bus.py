"""An event bus class for dispatching events between objects."""

class Bus:
    """A simple event bus for dispatching events between objects."""
    
    def __init__(self):
        self.__listeners = {}
    
    def on(self, event, listener, priority = 0):
        """Add a listener for an event."""
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
        caught = False
        if event in self.__listeners:
            for listener, _ in self.__listeners[event]:
                #print("event", event, "calling", listener)
                ret = listener(*args, **kwargs)
                if ret:
                    caught = True
                    if exclusive:
                        return ret
        return caught
