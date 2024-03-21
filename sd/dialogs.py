import gi
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib

## ---------------------------------------------------------------------
class HelpDialog(Gtk.Dialog):
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


