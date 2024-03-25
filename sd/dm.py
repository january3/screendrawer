import cairo # <remove>
from sd.drawable import DrawableFactory, SelectionTool             # <remove>
from sd.commands import MoveCommand, ResizeCommand, RotateCommand  # <remove>
from sd.events   import MouseEvent                                 # <remove>
from sd.utils    import get_color_under_cursor, rgb_to_hex         # <remove>
from sd.wiglets  import *                                          # <remove>
from sd.pen      import Pen                                        # <remove>



class DrawManager:
    """
    DrawManager is a class that manages the drawing canvas.
    It holds information about the state, the mouse events, the position of
    the cursor, whether there is a current object being generated, whether
    a resize operation is in progress, whether the view is paned or zoomed etc.

    DrawManager must be aware of GOM, because GOM holds all the objects
    """
    def __init__(self, gom, app, cursor, bg_color = (.8, .75, .65), transparent = 0.0):
        self.__current_object = None
        self.__pos = None
        self.__gom = gom
        self.__app = app
        self.__cursor = cursor
        self.__mode = "draw"

        # objects that indicate the state of the drawing area
        self.__hover = None
        self.__wiglet_active = None
        self.__resizeobj = None
        self.__selection_tool = None
        self.__current_object = None

        # drawing parameters
        self.__hidden = False
        self.__bg_color = bg_color
        self.__transparent = transparent
        self.__outline = False
        self.__modified = False

        # defaults for drawing
        self.__pen  = Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1)
        self.__pen2 = Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2)

    def pen_set(self, pen, alternate = False):
        """Set the pen."""
        if alternate:
            self.__pen2 = pen
        else:
            self.__pen = pen

    def pen(self, alternate = False):
        """Get the pen."""
        return self.__pen2 if alternate else self.__pen

    def hide(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__hidden = value
        return self.__hidden

    def hide_toggle(self):
        """Toggle the visibility of the drawing."""
        self.__hidden = not self.__hidden
        ##self.app.queue_draw()

    def current_object(self):
        """Get the current object."""
        return self.__current_object

    def mode(self, mode = None):
        """Get or set the mode."""
        if mode:
            self.__mode = mode
            self.__cursor.default(self.__mode)
        return self.__mode

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__modified = value
        return self.__modified

    def bg_color(self, color=None):
        if color:
            self.__bg_color = color
        return self.__bg_color

    def transparent(self, value=None):
        if value:
            self.__transparent = value
        return self.__transparent

    def on_draw(self, widget, cr):
        """Handle draw events."""
        if self.__hidden:
            print("I am hidden!")
            return True

        cr.set_source_rgba(*self.__bg_color, self.__transparent)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)

        if self.__current_object:
            self.__current_object.draw(cr)

        if self.__wiglet_active:
            self.__wiglet_active.draw(cr)
        return True

    def draw(self, cr):
        """Draw the objects in the given context. Used also by export functions."""

        for obj in self.__gom.objects():
            hover    = obj == self.__hover and self.__mode == "move"
            selected = self.__gom.selection.contains(obj) and self.__mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = self.__outline)
    #   if self.current_object:
    #       print("drawing current object:", self.current_object, "mode:", self.mode)
    #       self.current_object.draw(cr)

        # If changing line width, draw a preview of the new line width
      
    def clear(self):
        """Clear the drawing."""
        self.__gom.selection.clear()
        self.__resizeobj      = None
        self.__current_object = None
        self.__gom.remove_all()
        self.__app.queue_draw()

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_right_click(self, event, hover_obj):
        """Handle right click events - context menus."""
        if hover_obj:
            self.__mode = "move"
            self.__cursor.default(self.__mode)

            if not self.__gom.selection.contains(hover_obj):
                self.__gom.selection.set([ hover_obj ])

            # XXX - this should happen directly?
            self.__app.mm.object_menu(self.__gom.selected_objects()).popup(None, None, None, None, event.button, event.time)
        else:
            self.__app.mm.context_menu().popup(None, None, None, None, event.button, event.time)
        self.__app.queue_draw()

    # ---------------------------------------------------------------------

    def move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""
        corner_obj = ev.corner()
        hover_obj  = ev.hover()
        pos = ev.pos()
        shift, ctrl = ev.shift(), ev.ctrl()

        if corner_obj[0] and corner_obj[0].bbox():
            print("starting resize")
            obj    = corner_obj[0]
            corner = corner_obj[1]
            print("ctrl:", ctrl, "shift:", shift)
            # XXX this code here is one of the reasons why rotating or resizing the
            # whole selection does not work. The other reason is that the
            # selection object itself is not considered when detecting
            # hover or corner objects.
            if ctrl and shift:
                self.__resizeobj = RotateCommand(obj, origin = pos, corner = corner)
            else:
                self.__resizeobj = ResizeCommand(obj, origin = pos, corner = corner, proportional = ctrl)
            self.__gom.selection.set([ obj ])
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.__resizeobj)
            self.__cursor.set(corner)
        elif hover_obj:
            if ev.shift():
                # add if not present, remove if present
                print("adding object", hover_obj)
                self.__gom.selection.add(hover_obj)
            if not self.__gom.selection.contains(hover_obj):
                print("object not in selection, setting it", hover_obj)
                self.__gom.selection.set([ hover_obj ])
            self.__resizeobj = MoveCommand(self.__gom.selection, pos)
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.__resizeobj)
            self.__cursor.set("grabbing")
        else:
            self.__gom.selection.clear()
            self.__resizeobj   = None
            print("starting selection tool")
            self.__selection_tool = SelectionTool([ pos, (pos[0] + 1, pos[1] + 1) ])
            self.__current_object = self.__selection_tool # XXX -> this is
                                                          # a hack to force the draw function draw the selection tool
            self.__app.queue_draw()
        return True

    def create_object(self, ev):
        """Create an object based on the current mode."""
        # not managed by GOM: first create, then decide whether to add to GOM
        obj = DrawableFactory.create_drawable(self.__mode, pen = self.__pen, ev=ev)
        if obj:
            self.__current_object = obj

    def get_mode(self):
        """Get the current mode."""
        return self.__mode

    # XXX this code should be completely rewritten, cleaned up, refactored
    # and god knows what else. It's a mess.
    def on_button_press(self, widget, event):
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.__modified = True # better safe than sorry

        ev = MouseEvent(event, self.__gom.objects())
        shift, ctrl, pressure = ev.shift(), ev.ctrl(), ev.pressure()
        hover_obj = ev.hover()

        # double click on a text object: start editing
        if event.button == 1 and ev.double() and hover_obj and hover_obj.type == "text" and self.__mode in ["draw", "text", "move"]:
            # put the cursor in the last line, end of the text
            # this should be a Command event
            hover_obj.move_caret("End")
            self.__current_object = hover_obj
            self.__app.queue_draw()
            self.__cursor.set("none")
            return True

        # Ignore clicks when text input is active
        if self.__current_object:
            if  self.__current_object.type == "text":
                print("click, but text input active - finishing it first")
                self.finish_text_input()
            else:
                print("click, but text input active - ignoring it; object=", self.__current_object)
            return True

        # right click: open context menu
        if event.button == 3 and not shift:
            self.on_right_click(event, hover_obj)
            return True

        if event.button != 1:
            return True

        # Start changing line width: single click with ctrl pressed
        if ctrl and event.button == 1 and self.__mode == "draw": 
            if not shift: 
                self.__wiglet_active = WigletLineWidth((event.x, event.y), self.__pen)
            else:
                self.__wiglet_active = WigletTransparency((event.x, event.y), self.__pen)
            return True

        if self.__mode == "colorpicker":
            #print("picker mode")
            color = get_color_under_cursor()
            self.set_color(color) 
            color_hex = rgb_to_hex(color)
            self.__app.clipboard.set_text(color_hex)
            return True

        elif self.__mode == "move":
            return self.move_resize_rotate(ev)

        # erasing an object, if an object is underneath the cursor
        elif self.__mode == "eraser" and hover_obj: 
                ## XXX -> GOM 
                # self.history.append(RemoveCommand([ hover_obj ], self.objects))
                self.__gom.remove_objects([ hover_obj ], clear_selection = True)
                self.__resizeobj   = None
                self.__cursor.revert()

        # simple click: create modus
        else:
            self.create_object(ev)

        self.__app.queue_draw()

        return True

    # Event handlers
    # XXX same comment as above
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        obj = self.__current_object

        if obj and obj.type in [ "shape", "path" ]:
            print("finishing path / shape")
            obj.path_append(event.x, event.y, 0)
            obj.finish()
            if len(obj.coords) < 3:
                obj = None
            self.__app.queue_draw()

        if obj:
            # remove objects that are too small
            bb = obj.bbox()
            if bb and obj.type in [ "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                obj = None

        if obj:
            if obj != self.__selection_tool:
                self.__gom.add_object(obj)
            else:
                self.__current_object = None

        if self.__wiglet_active:
            self.__wiglet_active.event_finish()
            self.__wiglet_active = None
            self.__app.queue_draw()
            return True

        # if selection tool is active, finish it
        if self.__selection_tool:
            print("finishing selection tool")
            #self.objects.remove(self.selection_tool)
            #bb = self.selection_tool.bbox()
            objects = self.__selection_tool.objects_in_selection(self.__gom.objects())
            if len(objects) > 0:
                self.__gom.selection.set(objects)
            else:
                self.__gom.selection.clear()
            self.__selection_tool = None
            self.__app.queue_draw()
            return True

        # if the user clicked to create a text, we are not really done yet
        if self.__current_object and self.__current_object.type != "text":
            print("there is a current object: ", self.__current_object)
            self.__gom.selection.clear()
            self.__current_object = None
            self.__app.queue_draw()
            return True

        if self.__resizeobj:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            self.__resizeobj.event_finish()
            obj = self.__resizeobj.obj
            if self.__resizeobj.command_type == "move" and  event.x < 10 and event.y > self.get_size()[1] - 10:
                # command group because these are two commands: first move,
                # then remove
                self.__gom.command_append([ self.__resizeobj, RemoveCommand([ obj ], self.__gom.objects()) ])
                self.__selection.clear()
            else:
                self.__gom.command_append([ self.__resizeobj ])
            self.__resizeobj    = None
            self.__cursor.revert()
            self.__app.queue_draw()
        return True


    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.__gom.objects())
        x, y = ev.pos()
        self.__cursor.update_pos(x, y)

        if self.__wiglet_active:
            self.__wiglet_active.event_update(x, y)
            self.__app.queue_draw()
            return True

        obj = self.__current_object or self.__selection_tool

        if obj:
            obj.update(x, y, ev.pressure())
            self.__app.queue_draw()
        elif self.__resizeobj:
            self.__resizeobj.event_update(x, y)
            self.__app.queue_draw()
        elif self.__mode == "move":
            object_underneath = ev.hover()
            prev_hover = self.__hover

            if object_underneath:
                if object_underneath.type == "text":
                    self.__cursor.set("text")
                else:
                    self.__cursor.set("moving")
                self.__hover = object_underneath
            else:
                self.__cursor.revert()
                self.__hover = None

            corner_obj = ev.corner()

            if corner_obj[0] and corner_obj[0].bbox():
                self.__cursor.set(corner_obj[1])
                self.__hover = corner_obj[0]
                self.__app.queue_draw()

            if prev_hover != self.__hover:
                self.__app.queue_draw()

        # stop event propagation
        return True

    # ---------------------------------------------------------------------
    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        if self.__current_object and self.__current_object.type == "text":
            self.__current_object.caret_pos = None
            if self.__current_object.strlen() == 0:
                self.__gom.kill_object(self.__current_object)
            self.__current_object = None
        self.__cursor.revert()
    # ---------------------------------------------------------------------

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__transparent = {1: 0, 0: 0.5, 0.5: 1}[self.__transparent]

    def outline_toggle(self):
        """Toggle outline mode."""
        self.__outline = not self.__outline

    # ---------------------------------------------------------------------

    def stroke_increase(self):
        """Increase whatever is selected."""
        self.stroke_change(1)

    def stroke_decrease(self):
        """Decrease whatever is selected."""
        self.stroke_change(-1)

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        cobj = self.__current_object()
        if cobj and cobj.type == "text":
            print("Changing text size")
            cobj.stroke_change(direction)
            self.__pen.font_size = cobj.pen.font_size
        else: 
            for obj in self.__gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.__mode == "draw":
            self.__pen.line_width = max(1, self.__pen.line_width + direction)
        elif self.__mode == "text":
            self.__pen.font_size = max(1, self.__pen.font_size + direction)

    def set_color(self, color):
        self.__pen.color_set(color)
        self.__gom.selection_color_set(color)

    def set_font(self, font_description):
        """Set the font."""
        self.__pen.font_set_from_description(font_description)
        self.__gom.selection_font_set(font_description)
        if self.__current_object and self.__current_object.type == "text":
            self.__current_object.pen.font_set_from_description(font_description)

#   def smoothen(self):
#       """Smoothen the selected object."""
#       if self.selection.n() > 0:
#           for obj in self.selection.objects:
#               obj.smoothen()

    def switch_pens(self):
        """Switch between pens."""
        self.__pen, self.__pen2 = self.__pen2, self.__pen

    def apply_pen_to_bg(self):
        """Apply the pen to the background."""
        self.__bg_color = self.__pen.color


