"""Module for history tracking."""
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

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
        log.debug(f"appending {item.type()}")
        self.__history.append(item)
        self.__redo = []

    def undo(self):
        """Undo the last action."""
        if not self.__history:
            return None

        log.debug(f"undoing {self.__history[-1].type()}")

        cmd = self.__history.pop()
        ret = cmd.undo()
        self.__redo.append(cmd)
        return ret

    def redo(self):
        """Redo the last action."""
        if not self.__redo:
            return None

        cmd = self.__redo.pop()
        log.debug(f"redoing {cmd.type()}")
        ret = cmd.redo()
        self.__history.append(cmd)
        return ret
