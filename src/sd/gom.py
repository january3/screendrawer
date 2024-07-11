"""
Graphics Object Manager is the top level for handling the graphic objects.
The hierarchy is GOM -> Page -> Layer. Layer does all the footwork, Page
handles Layers, and GOM handles Pages.
"""

import logging                             # <remove>
from .commands import CommandGroup         # <remove>
from .commands_obj import RemoveCommand    # <remove>
from .page import Page                     # <remove>

log = logging.getLogger(__name__)          # <remove>


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects - in practice, to manage pages.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self, bus):

        # private attr
        self.__bus        = bus
        self.__page = None
        self.page_set(Page())
        self.__add_bus_listeners()

    def __add_bus_listeners(self):
        """Add listeners to the bus."""
        self.__bus.on("next_page", self.next_page)
        self.__bus.on("prev_page", self.prev_page)
        self.__bus.on("insert_page", self.insert_page)
        self.__bus.on("page_set", self.page_set)
        self.__bus.on("delete_page", self.delete_page)

    def page_set(self, page):
        """Set the current page."""
        self.__bus.emit("page_changed", False, page)
        self.__page = page
        self.__page.activate(self.__bus)

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def insert_page(self):
        """Insert a new page."""
        curpage, cmd = self.__page.insert()
        self.__bus.emit_once("history_append", cmd)
        self.page_set(curpage)

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        curpage, cmd = self.__page.delete()

        if curpage == self.page():
            return

        self.page_set(curpage)
        self.__bus.emit("history_append", True, cmd)

    def page(self):
        """Return the current page."""
        return self.__page

    def set_page_number(self, n):
        """Choose page number n."""
        tot_n = self.number_of_pages()
        if n < 0 or n >= tot_n:
            return
        cur_n = self.current_page_number()

        if n == cur_n:
            return
        if n > cur_n:
            for _ in range(n - cur_n):
                self.next_page()
        else:
            for _ in range(cur_n - n):
                self.prev_page()

    def number_of_pages(self):
        """Return the total number of pages."""
        p = self.start_page()

        n = 1
        while p.next(create = False):
            n += 1
            p = p.next(create = False)
        return n

    def current_page_number(self):
        """Return the current page number."""
        p = self.__page
        n = 0
        while p.prev() != p:
            n += 1
            p = p.prev()
        return n

    def start_page(self):
        """Return the first page."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()
        return p

    def objects(self):
        """Return the list of objects."""
        return self.__page.layer().objects()

    def selection(self):
        """Return the selection object."""
        return self.__page.layer().selection()

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        log.debug("GOM: setting n=%d objects", len(objects))
        self.__page.layer().objects(objects)

    def set_pages(self, pages):
        """Set the content of pages."""
        self.__page = Page()
        self.__page.import_page(pages[0])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.import_page(p)
        self.__page.activate(self.__bus)

    def get_all_pages(self):
        """Return all pages."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()

        pages = [ ]

        while p:
            pages.append(p)
            p = p.next(create = False)

        return pages

    def export_pages(self):
        """Export all pages."""

        pages = [ p.export() for p in self.get_all_pages() ]
        return pages

    def selected_objects(self):
        """Return the selected objects."""
        return self.__page.layer().selection().objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__page.layer().selection().is_empty():
            return
        cmd = RemoveCommand(self.__page.layer().selection().objects,
                                            self.__page.layer().objects())
        self.__bus.emit("history_append", True, cmd)
        self.__page.layer().selection().clear()

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order!
        cmd = CommandGroup(command_list[::-1])
        self.__bus.emit("history_append", True, cmd)
