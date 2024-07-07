## ---------------------------------------------------------------------
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>

## ------------------------ logging
from gi.repository import Gtk, Gdk                               # <remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>
log.setLevel(logging.DEBUG)                                       # <remove>

from .drawable_group import ClipboardGroup # <remove>
from .utils import img_object_copy # <remove>

class Clipboard:
    """
    Class to handle clipboard operations. Basically, it handles internally
    the clipboard within the app, but sets the gtk clipboard if necessary.

    Atrributes:
        clipboard_owner (bool): True if the app owns the clipboard.
        clipboard (unspecified): The clipboard content.

    """

    def __init__(self, gtk_clipboard=None):
        self.__gtk_clipboard = gtk_clipboard or Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.__gtk_clipboard.connect('owner-change', self.on_clipboard_owner_change)
        self.clipboard_owner = False
        self.clipboard = None

    def on_clipboard_owner_change(self, clipboard, event):
        """Handle clipboard owner change events."""

        log.debug("Owner change (%s), removing internal clipboard, reason: %s", clipboard, event.reason)
        if self.clipboard_owner:
            self.clipboard_owner = False
        else:
            self.clipboard = None
        return True

    def set_text(self, text):
        """
        Set text to clipboard and store it.

        Note:
            This is like external copy, so the internal clipboard is set to None.
        """
        self.clipboard_owner = False
        self.clipboard = None
        self.__gtk_clipboard.set_text(text, -1)
        self.__gtk_clipboard.store()

    def get_content(self):
        # internal paste
        if self.clipboard:
            log.debug("Pasting content internally")
            return "internal", self.clipboard

        # external paste
        clipboard = self.__gtk_clipboard
        clip_text = clipboard.wait_for_text()

        if clip_text:
            return "text", clip_text

        clip_img = clipboard.wait_for_image()
        if clip_img:
            return "image", clip_img
        return None, None

    def __make_copies(objects):
        """Make copies of objects."""
        copies = []

        for obj in objects:
            copies.append(obj.duplicate())

        return copies


    def copy_content(self, selection, cut = False):
        """
        Copy internal content: object or objects from current selection.

        Args:
            selection (Drawable): The selection to copy, either a group or
                                  a single object.
        """
        clipboard = self.__gtk_clipboard

        # don't like the code below
        if selection.length() == 1:
            sel = selection.objects[0]
        else:
            sel = selection

        if sel.type == "text":
            text = sel.to_string()
            log.debug("Copying text %s", text)
            # just copy the text
            clipboard.set_text(text, -1)
            clipboard.store()
        elif sel.type == "image":
            log.debug("Copying image")
            # simply copy the image into clipboard
            clipboard.set_image(sel.image())
            clipboard.store()
        else:
            log.debug("Copying another object")
            # draw a little image and copy it to clipboard
            img_copy = img_object_copy(sel)
            clipboard.set_image(img_copy)
            clipboard.store()

        log.debug("Setting internal clipboard")
        self.clipboard = ClipboardGroup(selection.objects[:], cut = cut)
        self.clipboard_owner = True
