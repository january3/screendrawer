"""
This module contains the Page class, which is a container for objects.
"""
from .drawable import SelectionObject # <remove>


class Page:
    """
    A page is a container for objects.
    """
    def __init__(self, prev = None):
        self.__prev    = prev
        self.__next = None
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)

    def next(self, create = True):
        """
        Return the next page.

        If create is True, create a new page if it doesn't exist.
        """
        if not self.__next and create:
            print("Creating new page")
            self.__next = Page(self)
        return self.__next

    def prev(self):
        """Return the previous page."""
        return self.__prev or self # can't go beyond first page

    def next_set(self, page):
        """Set the next page."""
        self.__next = page

    def prev_set(self, page):
        """Set the previous page."""
        self.__prev = page

    def delete(self):
        """Delete the page and create links between prev and next pages."""
        if not self.__prev and not self.__next:
            print("only one page remaining")
            return self

        if self.__prev:
            self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        return self.__prev or self.__next

    def objects(self, objects = None):
        """Return or set the list of objects on the page."""
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        if obj in self.__objects:
            self.__objects.remove(obj)
