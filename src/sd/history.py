"""Module for history tracking."""
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

class History:
    """
    Class for history tracking.

    Keeps track of the undo / redo stacks.
    """

    def __init__(self, bus):
        self.__history = []
        self.__redo = []
        self.__bus = bus
        self.__cur_page = "None"
        bus.on("page_changed",   self.set_page)
        bus.on("history_redo",   self.redo)
        bus.on("history_undo",   self.undo)
        bus.on("history_append", self.add)

    def length(self):
        """Return the number of items in the history."""
        return len(self.__history)

    def set_page(self, page):
        """Set the current page."""
        log.debug(f"setting page to {page}")
        self.__cur_page = page

    def add(self, cmd):
        """Add item to history."""
        log.debug(f"appending {cmd.type()} on page={self.__cur_page}")
        self.__history.append({'cmd': cmd, 'page': self.__cur_page})
        self.__redo = []

    def undo(self):
        """Undo the last action."""
        if not self.__history:
            return None

        item = self.__history.pop()
        cmd = item['cmd']
        log.debug(f"undoing {cmd.type()}")
        ret = cmd.undo()
        self.__redo.append(item)
        if item['page'] != self.__cur_page:
            self.__bus.emit("page_set", False, item['page'])
        return ret

    def redo(self):
        """Redo the last action."""
        if not self.__redo:
            return None

        item = self.__redo.pop()
        cmd = item['cmd']
        log.debug(f"redoing {cmd.type()}")
        ret = cmd.redo()
        self.__history.append(item)
        if item['page'] != self.__cur_page:
            self.__bus.emit("page_set", False, item['page'])
        return ret
