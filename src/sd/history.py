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
        bus.on("page_changed",     self.set_page)
        bus.on("history_redo",     self.redo)
        bus.on("history_undo",     self.undo)
        bus.on("history_undo_cmd", self.undo_command)
        bus.on("history_append",   self.add)
        bus.on("history_remove",   self.history_remove)

    def length(self):
        """Return the number of items in the history."""
        return len(self.__history)

    def set_page(self, page):
        """Set the current page."""
        log.debug("setting page to %s", page)
        self.__cur_page = page

    def add(self, cmd):
        """Add item to history."""
        log.debug("appending {cmd.type()} on page={self.__cur_page} new cmd hash=%s", cmd.hash())

        oldcmd  = self.__history[-1]['cmd']  if self.__history else None
        oldpage = self.__history[-1]['page'] if self.__history else None
        log.debug("oldcmd hash={oldcmd.hash() if oldcmd else None} oldpage=%s", oldpage)

        if oldcmd and oldpage == self.__cur_page:
            if oldcmd == cmd or oldcmd > cmd:
                log.debug("cmd hash=%s", cmd.hash() if cmd else None)
                log.debug("merging commands, {cmd.type()} and %s", oldcmd.type())
                self.__history.pop()
                cmd = oldcmd + cmd
                log.debug("new command hash=%s", cmd.hash())

        self.__history.append({'cmd': cmd, 'page': self.__cur_page})
        self.__redo = []

    def history_remove(self, cmd):
        """Remove an item from the history."""
        log.debug("removing %s from history", cmd.type())
        n = len(self.__history)
        self.__history = [ item for item in self.__history if item['cmd'] != cmd ]
        if n == len(self.__history):
            log.warning("could not remove %s from history", cmd.type())

    def undo_command(self, cmd):
        """
        Undo a specific command.

        Dangerous! Use with caution. Make sure that the command does not
        have any side effects.
        """
        log.debug("undoing specific command, type %s", cmd.type())
        if not self.__history:
            log.warning("Nothing to undo")
            return None

        if self.__history[-1]['cmd'] == cmd:
            return self.undo()

        log.warning("Undoing cmd %s which is not the last in history, beware", cmd.type())

        for i, item in enumerate(self.__history):
            if item['cmd'] == cmd:
                self.__history.pop(i)
                self.__redo.append(item)
                if item['page'] != self.__cur_page:
                    self.__bus.emit("page_set", False, item['page'])
                return cmd.undo()
        log.warning("Command %s not found in  history", cmd.type())
        return None

    def undo(self):
        """Undo the last action."""
        if not self.__history:
            log.warning("Nothing to undo")
            return None

        item = self.__history.pop()
        cmd = item['cmd']
        log.debug("undoing %s", cmd.type())
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
        log.debug("redoing %s", cmd.type())
        ret = cmd.redo()
        self.__history.append(item)
        if item['page'] != self.__cur_page:
            self.__bus.emit("page_set", False, item['page'])
        return ret
