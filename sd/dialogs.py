import gi # <remove>
import os # <remove>
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib # <remove>

## ---------------------------------------------------------------------
class help_dialog(Gtk.Dialog):
    def __init__(self, parent):
        super().__init__(title="Help", transient_for=parent, flags=0)
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)

        parent_size = parent.get_size()
        self.set_default_size(max(parent_size[0] * 0.8, 400), max(parent_size[1] * 0.9, 300))
        self.set_border_width(10)

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_min_content_width(380)
        scrolled_window.set_min_content_height(280)
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_hexpand(True)  # Allow horizontal expansion
        scrolled_window.set_vexpand(True)  # Allow vertical expansion

        help_text = f"""

<span font="24"><b>screendrawer</b></span>

Draw on the screen with Gnome and Cairo. Quick and dirty.

<b>(Help not complete yet.)</b>

<span font_family="monospace">
<b>Mouse:</b>

<b>All modes:</b>                                 <b>Move mode:</b>
shift-click:  Enter text mode              click: Select object   Resizing: click in corner
                                           move: Move object      Rotating: ctrl-shift-click in corner
ctrl-click:   Change line width            ctrl-a: Select all
ctrl-shift-click: Change transparency

Moving object to left lower screen corner deletes it.

<b>Shortcut keys:</b>

<b>Drawing modes:</b> (simple key, when not editing a text)

<b>d:</b> Draw mode (pencil)                 <b>m, |SPACE|:</b> Move mode (move objects around, copy and paste)
<b>t:</b> Text mode (text entry)             <b>b:</b> Box mode  (draw a rectangle)
<b>c:</b> Circle mode (draw an ellipse)      <b>e:</b> Eraser mode (delete objects with a click)
<b>p:</b> Polygon mode (draw a polygon)      <b>i:</b> Color pIcker mode (pick a color from the screen)

<b>Works always:</b>                                                                  <b>Move mode only:</b>
<b>With Ctrl:</b>              <b>Simple key (not when entering text)</b>                    <b>With Ctrl:</b>             <b>Simple key (not when entering text)</b>
Ctrl-q: Quit            x, q: Exit                                             Ctrl-c: Copy content   Tab: Next object
Ctrl-e: Export drawing  h, F1, ?: Show this help dialog                        Ctrl-v: Paste content  Shift-Tab: Previous object
Ctrl-l: Clear drawing   l: Clear drawing                                       Ctrl-x: Cut content    Shift-letter: quick color selection e.g. Shift-r for red
Ctrl-i: insert image                                                                                  |Del|: Delete selected object
Ctrl-z: undo            |Esc|: Finish text input                                                      g, u: group, ungroup                           
Ctrl-y: redo            |Enter|: New line (in text mode)                                 

Ctrl-k: Select color                     f: fill with current color
Ctrl-plus, Ctrl-minus: Change text size  o: toggle outline
Ctrl-b: Cycle background transparency
Ctrl-p: switch pens

</span>

The state is saved in / loaded from `{savefile}` so you can continue drawing later.
You might want to remove that file if something goes wrong.
        """
        label = Gtk.Label()
        label.set_markup(help_text)
        label.set_justify(Gtk.Justification.LEFT)
        label.set_line_wrap(True)
        scrolled_window.add(label)

        box = self.get_content_area()
        box.pack_start(scrolled_window, True, True, 0)  # Use pack_start with expand and fill

        self.show_all()

## ---------------------------------------------------------------------

def _dialog_add_image_formats(dialog):
    formats = {
        "All files": { "pattern": "*",      "mime_type": "application/octet-stream", "name": "all" },
        "PNG files":  { "pattern": "*.png",  "mime_type": "image/png",       "name": "png" },
        "JPEG files": { "pattern": "*.jpeg", "mime_type": "image/jpeg",      "name": "jpeg" },
        "PDF files":  { "pattern": "*.pdf",  "mime_type": "application/pdf", "name": "pdf" }
    }

    for name, data in formats.items():
        filter = Gtk.FileFilter()
        filter.set_name(name)
        filter.add_pattern(data["pattern"])
        filter.add_mime_type(data["mime_type"])
        dialog.add_filter(filter)

## ---------------------------------------------------------------------

def export_dialog(widget, parent = None):
    """Show a file chooser dialog to select a file to save the drawing."""
    file_name, selected_filter = None, None

    dialog = Gtk.FileChooserDialog(
        title="Save As", parent=parent, action=Gtk.FileChooserAction.SAVE)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    _dialog_add_image_formats(dialog)

#   # Add file format filters
#   filter_sdrw = Gtk.FileFilter()
#   filter_sdrw.set_name("Screendrawer files")
#   filter_sdrw.add_pattern("*.sdrw")
#   dialog.add_filter(filter_sdrw)

    # Show the dialog
    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
        selected_filter = dialog.get_filter().get_name()
        selected_filter = formats[selected_filter]["name"]
        print(f"Save file as: {file_name}, Format: {selected_filter}")

    dialog.destroy()
    return file_name, selected_filter

def import_image_dialog(widget, parent = None):
    """Show a file chooser dialog to select an image file."""
    dialog = Gtk.FileChooserDialog(
        title="Select an Image",
        parent=parent,
        action=Gtk.FileChooserAction.OPEN,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    # Filter to only show image files
    file_filter = Gtk.FileFilter()
    file_filter.set_name("Image files")
    file_filter.add_mime_type("image/jpeg")
    file_filter.add_mime_type("image/png")
    file_filter.add_mime_type("image/tiff")
    dialog.add_filter(file_filter)

    # Show the dialog and wait for the user response
    response = dialog.run()

    pixbuf = None
    image_path = None
    if response == Gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    elif response == Gtk.ResponseType.CANCEL:
        print("No image selected")

    # Clean up and destroy the dialog
    dialog.destroy()
    return image_path

def open_drawing_dialog(parent):
    # Create a file chooser dialog
    dialog = Gtk.FileChooserDialog(
        title="Select an .sdrw file",
        action=Gtk.FileChooserAction.OPEN,
        parent=parent,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    # Show the dialog and wait for the user response
    response = dialog.run()

    image_path = None
    if response == Gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    elif response == Gtk.ResponseType.CANCEL:
        print("No image selected")

    # Clean up and destroy the dialog
    dialog.destroy()
    return image_path
