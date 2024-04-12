"""An event bus class for dispatching events between objects."""

class Bus:
    """A simple event bus for dispatching events between objects."""
    
    def __init__(self):
        self.listeners = {}
    
    def on(self, event, listener):
        """Add a listener for an event."""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(listener)
    
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
            for listener in self.listeners[event]:
                #print("calling", listener)
                ret = listener(*args)
                if ret:
                    caught = True
                    if exclusive:
                        return ret
        return caught
