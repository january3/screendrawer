"""
Dialogs for the ScreenDrawer application.
"""

import os # <remove>
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk # <remove>
from .pen import Pen # <remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

## ---------------------------------------------------------------------
FORMATS = {
    "All files": { "pattern": "*",      "mime_type": "application/octet-stream", "name": "any" },
    "PNG files":  { "pattern": "*.png",  "mime_type": "image/png",       "name": "png" },
    "JPEG files": { "pattern": "*.jpeg", "mime_type": "image/jpeg",      "name": "jpeg" },
    "PDF files":  { "pattern": "*.pdf",  "mime_type": "application/pdf", "name": "pdf" }
}


## ---------------------------------------------------------------------
class help_dialog(Gtk.Dialog):
    """A dialog to show help information."""
    def __init__(self, parent):
        log.debug(f"parent: {parent}")
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
<b>General / Interface:</b>
w: Toggle UI (hide / show widgets)         F1, h, ?: Show this help dialog
Ctrl-q, x, q: Quit                         

<b>UI:</b> (toggle with 'w')
Color selector: click to select pen color, shift-click to select background color
Tool selector: click to select tool
Page selector: click to select page, click on '+' to add a new page

<b>Mouse:</b>

<b>All modes:</b>                                 <b>Move mode:</b>
shift-click:  Enter text mode              click: Select object             Resizing: click in corner
                                           click and drag: Move object      Rotating: ctrl-shift-click in corner
ctrl-click:   Change line width            ctrl-a: Select all
ctrl-shift-click: Change transparency

Moving object to left lower screen corner deletes it.

<b>Shortcut keys:</b>

<b>Drawing modes:</b> (simple key, when not editing a text)

<b>d:</b> Draw mode (pencil)                 <b>m, |SPACE|:</b> Move mode (move objects around, copy and paste)
<b>t:</b> Text mode (text entry)             <b>r:</b> rectangle mode  (draw a rectangle)
<b>c:</b> Circle mode (draw an ellipse)      <b>e:</b> Eraser mode (delete objects with a click)
<b>s:</b> Shape mode (draw a filled shape)   <b>i:</b> Color p<b>i</b>cker mode (pick a color from the screen)
<b>Shift-s:</b> Segmented path mode (double-click to finish a segmented path)

<b>Works always:</b>                                                             <b>Move mode only:</b>
<b>With Ctrl:</b>              <b>Simple key (not when entering text)</b>               <b>With Ctrl:</b>             <b>Simple key (not when entering text)</b>
Ctrl-e: Export drawing                                                    Ctrl-c: Copy content   Tab: Next object
Ctrl-l: Clear drawing   l: Clear drawing                                  Ctrl-v: Paste content  Shift-Tab: Previous object
                                                                          Ctrl-x: Cut content    Shift-letter: quick color selection e.g.
                                                                                                 Shift-r for red
Ctrl-i: insert image                                                                             |Del|: Delete selected object(s)
Ctrl-z: undo            |Esc|: Finish text input                                                 
Ctrl-y: redo            |Enter|: New line (when typing)                   Alt-Up, Alt-Down: Move object up, down in stack
                                                                          Alt-PgUp, Alt-PgDown: Move object to front, back
Ctrl-k: Select color                     f: fill with current color       Alt-s: convert drawing(s) to shape(s)
Ctrl-Shift-k: Select bg color
Ctrl-plus, Ctrl-minus: Change text size  o: toggle outline                Alt-d: convert shape(s) to drawing(s)
Ctrl-b: Cycle background transparency                                     Alt-p: apply pen to selection
Ctrl-p: toggle between two pens                                           Alt-Shift-p: apply pen color to background
Ctrl-g: toggle grid                      1-5: select brush

Ctrl-Shift-g: toggle "group while drawing" mode                           

<b>Brushes:</b> (select with 1-5)

1: general rounded brush
2: marker
3: calligraphy
4: pencil (pressure changes transparency)
5: tapered brush

<b>Group operations:</b>
g, u: group, ungroup (move mode only)
Shift-c, Shift-u: clip / unclip group (need a rectangle, circle or shape as the last selected object)

Alt-Shift-Arrow: flush objects in a group or selection:
    Alt-Shift-Left:      to the left
    Alt-Shift-Right:     to the right
    Alt-Shift-Up:        to the top
    Alt-Shift-Down:      to the bottom

<b>Pages and layers:</b>

Pages (= slides) can hold multiple layers. When you select and move
objects, you are always acting on the current layer.

Shift-n: Next / new page                  Ctrl-Shift-n: Next / new layer
Shift-p: Previous page                    Ctrl-Shift-p: Previous layer
Shift-d: Delete current page              Ctrl-Shift-d: Delete layer
Shift-i: Insert new page after current

If you have more than one page, exporting to PDF will create a multipage PDF
if you check the "multipage" checkbox.

<b>Taking screenshots:</b>
Ctrl-Shift-f: screenshot: for a screenshot, if you have at least one rectangle                    This is likely to change in the future.
object (r mode) selected, then it will serve as the selection area. The                        
screenshot will be pasted into the drawing. If no rectangle is selected, then
the mode will change to "rectangle" and the next rectangle you draw will be
used as the capture area.

<b>Saving / importing:</b>
Ctrl-i: Import image from a file (jpeg, png)
Ctrl-o: Open a drawing from a file (.sdrw, that is the "native format") -
        note that the subsequent modifications will be saved to that file only
Ctrl-e: Export selection or whole drawing to a file (png, jpeg, pdf)
Ctrl-Shift-s: "Save as" - save drawing to a file (.sdrw, that is the "native format") - note
        that the subsequent modifications will be saved to that file only
Ctrl-c, Ctrl-v: copy and paste objects

When you copy a selection or individual objects, you can paste them into
other programs as a PNG image.

If you have more than one page, exporting to PDF will create a multipage PDF
if you check the "multipage" checkbox.

<b>Group while drawing mode:</b>
To make drawing complex objects (or handwriting, for that matter) easier,
you can press Ctrl-Shift-g to toggle "group while drawing" mode. In this mode,
all objects you draw are automatically grouped together. Stop by pressing
either the escape key or Ctrl-Shift-g again.

</span>

The state is saved in / loaded from `{parent.state.config().savefile()}` so you can continue drawing later.
An autosave happens every minute or so. Press ESC to exit this help screen.
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
    for name, data in FORMATS.items():
        file_filter = Gtk.FileFilter()
        file_filter.set_name(name)
        file_filter.add_pattern(data["pattern"])
        file_filter.add_mime_type(data["mime_type"])
        dialog.add_filter(file_filter)

## ---------------------------------------------------------------------

def test_func(button):

    log.debug("itsa me, test_func")
    log.debug(f"button value: {button.get_active()}")

def export_dialog_extra_widgets(ui_opts = None):
    # Create a ComboBoxText for file format selection
    format_selector = Gtk.ComboBoxText()
    
    # Add file format options
    formats = [ "By extension", "PDF", "SVG", "PNG"]
    for fmt in formats:
        format_selector.append_text(fmt)
    
    # Set the default selection to either
    # ui_opts["format"] or if that is None, to 0
    if ui_opts and ui_opts.get("format"):
        format_selector.set_active(formats.index(ui_opts["format"]))
    else:
        format_selector.set_active(0)

    # Create a checkbox for "export all pages as PDF"
    export_all_checkbox = Gtk.CheckButton(label="Export all pages as PDF")
    if ui_opts and ui_opts.get("all_pages_pdf"):
        export_all_checkbox.set_active(ui_opts["all_pages_pdf"])

    # Checkbox for "export screen view"
    export_screen_checkbox = Gtk.CheckButton(label="Export screen view")
    if ui_opts and ui_opts.get("export_screen"):
        export_screen_checkbox.set_active(ui_opts["export_screen"])
 
    # Function to update checkbox sensitivity
    def update_checkbox_sensitivity(combo):
        selected_format = combo.get_active_text()
        export_all_checkbox.set_sensitive(selected_format in [ "PDF", "By extension"])
    
    # Connect the combo box's changed signal to update the checkbox
    format_selector.connect("changed", update_checkbox_sensitivity)
    
    # Initial update of checkbox sensitivity
    update_checkbox_sensitivity(format_selector)
  
    # Create a label for the selector
    label = Gtk.Label(label="File Format:")
    
    # Create a horizontal box to hold the label and selector
    hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
    hbox.pack_start(label, False, False, 0)
    hbox.pack_start(format_selector, False, False, 0)
    hbox.pack_start(export_all_checkbox, False, False, 0)
    hbox.pack_start(export_screen_checkbox, False, False, 0)
    
    return hbox, format_selector, export_all_checkbox, export_screen_checkbox


def export_dialog(parent, export_dir = None, filename=None, ui_opts = None):
    """Show a file chooser dialog to select a file to save the drawing as
    an image / pdf / svg."""
    log.debug("export_dialog")
    file_name, selected_filter = None, None
    selected_format = None
    export_screen = False

    ## doesn't really work because we don't have a standalone window with
    ## its own title bar
    selected = ui_opts.get("selected", False)
    title = "Export selected objects As" if selected else "Export all objects As"

    dialog = Gtk.FileChooserDialog(
        title=title, parent=parent, action=Gtk.FileChooserAction.SAVE)

    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    ret =  export_dialog_extra_widgets(ui_opts = ui_opts)
    hbox, fmt_selector, exp_all_cb, exp_screen_cb = ret

    dialog.set_extra_widget(hbox)
    hbox.show_all()

    current_directory = export_dir or os.getcwd()
    log.debug(f"current_directory: {current_directory}, filename: {filename}")

    if filename: 
        dialog.set_filename(filename)

    dialog.set_current_folder(current_directory)

    _dialog_add_image_formats(dialog)

    log.debug(f"filter: {dialog.get_filter()}")
    all_pages_pdf = False

    # Show the dialog
    response = dialog.run()

    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
        selected_filter = dialog.get_filter().get_name()
        selected_filter = FORMATS[selected_filter]["name"]
        selected_format = fmt_selector.get_active_text()
        all_pages_pdf = exp_all_cb.get_active()
        export_screen = exp_screen_cb.get_active()

        if selected_format == "by extension":
            selected_format = "any"

        log.debug("Save file as: %s; Format: %s/%s",
                  file_name, selected_filter, selected_format)
        log.debug("all pages as PDF: %s; export_screen: %s",
                  all_pages_pdf, export_screen)


    dialog.destroy()
    return file_name, selected_format, all_pages_pdf, export_screen

def save_dialog(parent):
    """Show a file chooser dialog to set the savefile."""
    log.debug("save_dialog")
    #file_name, selected_filter = None, None
    file_name = None

    dialog = Gtk.FileChooserDialog(
        title="Save As", parent=parent, action=Gtk.FileChooserAction.SAVE)

    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    #_dialog_add_image_formats(dialog)

    # Show the dialog
    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
       #selected_filter = dialog.get_filter().get_name()
       #selected_filter = formats[selected_filter]["name"]
       #print(f"Save file as: {file_name}, Format: {selected_filter}")

    dialog.destroy()
    return file_name


def import_image_dialog(parent, import_dir = None):
    """Show a file chooser dialog to select an image file."""
    dialog = Gtk.FileChooserDialog(
        title="Select an Image",
        parent=parent,
        action=Gtk.FileChooserAction.OPEN,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = import_dir or os.getcwd()
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
    """Show a file chooser dialog to select a .sdrw file."""
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

def FontChooser(pen, parent):

    # check that pen is an instance of Pen
    if not isinstance(pen, Pen):
        raise ValueError("Pen is not defined or not of class Pen")

    font_dialog = Gtk.FontChooserDialog(title="Select a Font", parent=parent)
    #font_dialog.set_preview_text("Zażółć gęślą jaźń")
    font_dialog.set_preview_text("Sphinx of black quartz, judge my vow.")

    # You can set the initial font for the dialog
    font_dialog.set_font(pen.font_family + " " +
                         pen.font_style + " " +
                         str(pen.font_weight) + " " +
                         str(pen.font_size))

    response = font_dialog.run()

    font_description = None

    if response == Gtk.ResponseType.OK:
        font_description = font_dialog.get_font_desc()

    font_dialog.destroy()
    return font_description


def ColorChooser(parent, title = "Select Pen Color"):
    """Select a color for drawing."""
    # Create a new color chooser dialog
    color_chooser = Gtk.ColorChooserDialog(title, parent = parent)

    # Show the dialog
    response = color_chooser.run()
    color = None

    # Check if the user clicked the OK button
    if response == Gtk.ResponseType.OK:
        color = color_chooser.get_rgba()
        #self.set_color((color.red, color.green, color.blue))

    # Don't forget to destroy the dialog
    color_chooser.destroy()
    return color



