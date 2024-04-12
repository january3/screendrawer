"""An event bus class for dispatching events between objects."""

class Bus:
    """A simple event bus for dispatching events between objects."""
    
    def __init__(self):
        self.listeners = {}
    
    def on(self, event, listener, priority = 0):
        """Add a listener for an event."""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append((listener, priority))
        self.listeners[event].sort(key = lambda x: -x[1])
    
    def off(self, event, listener):
        """Remove a listener for an event."""
        if event in self.listeners:
            self.listeners[event].remove(listener)
    
    def emit(self, event, exclusive = False, *args):
        """
        Dispatch an event to all listeners.

        Exclusive events will stop dispatching if a listener returns a truthy value.
        """
        caught = False
        if event in self.listeners:
            for listener, _ in self.listeners[event]:
                #print("event", event, "calling", listener)
                ret = listener(*args)
                if ret:
                    caught = True
                    if exclusive:
                        return ret
        return caught
