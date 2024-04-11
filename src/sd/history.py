"""Module for history tracking."""

class History:
    """
    Class for history tracking.

    Keeps track of the undo / redo stacks.
    """

    def __init__(self):
        self.__history = []
        self.__redo = []

    def length(self):
        """Return the number of items in the history."""
        return len(self.__history)

    def add(self, item):
        """Add item to history."""
        self.__history.append(item)
        self.__redo = []

    def undo(self):
        """Undo the last action."""
        if not self.__history:
            return None

        cmd = self.__history.pop()
        ret = cmd.undo()
        self.__redo.append(cmd)
        return ret

    def redo(self):
        """Redo the last action."""
        if not self.__redo:
            return None

        cmd = self.__redo.pop()
        ret = cmd.redo()
        self.__history.append(cmd)
        return ret
