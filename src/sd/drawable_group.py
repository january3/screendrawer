"""
Classes that represent groups of drawable objects.
"""

from .utils import flatten_and_unique         # <remove>
from .drawable import Drawable                # <remove>

class DrawableGroup(Drawable):
    """
    Class for creating groups of drawable objects or other groups.
    Most of the time it just passes events around.

    Attributes:
        objects (list): The list of objects in the group.
    """
    def __init__(self, objects = None, objects_dict = None, mytype = "group"):

        if objects is None:
            objects = [ ]
        if objects_dict:
            objects = [ Drawable.from_dict(d) for d in objects_dict ]

        #print("Creating DrawableGroup with ", len(objects), "objects")
        super().__init__(mytype, [ (None, None) ], None)
        self.objects = objects

    def contains(self, obj):
        """Check if the group contains the object."""
        return obj in self.objects

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to one of the objects."""
        for obj in self.objects:
            if obj.is_close_to_click(click_x, click_y, threshold):
                return True
        return False

    def fill_toggle(self):
        """Toggle the fill of the objects"""
        for obj in self.objects:
            obj.fill_toggle()
        self.mod += 1

    def stroke_change(self, direction):
        """Change the stroke size of the objects in the group."""
        for obj in self.objects:
            obj.stroke_change(direction)
        self.mod += 1

    def transmute_to(self, mode):
        """Transmute all objects within the group to a new type."""
        print("transmuting group to", mode)
       #for i in range(len(self.objects)):
       #    self.objects[i] = DrawableFactory.transmute(self.objects[i], mode)
        self.mod += 1

    def to_dict(self):
        """Convert the group to a dictionary."""
        return {
            "type": self.type,
            "objects_dict": [ obj.to_dict() for obj in self.objects ],
        }

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(),
            "orig_bbox": self.bbox(),
            "objects": { obj: obj.bbox() for obj in self.objects }
            }

        for obj in self.objects:
            obj.resize_start(corner, origin)
        self.mod += 1

    def get_primitive(self):
        """Return the primitives of the objects in the group."""
        primitives = [ obj.get_primitive() for obj in self.objects ]
        return flatten_and_unique(primitives)

    def rotate_start(self, origin):
        """Start the rotation operation."""
        self.rot_origin = origin
        for obj in self.objects:
            obj.rotate_start(origin)
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the objects in the group."""
        if set_angle:
            self.rotation = angle
        else:
            self.rotation += angle
        for obj in self.objects:
            obj.rotate(angle, set_angle)
        self.mod += 1

    def rotate_end(self):
        """Finish the rotation operation."""
        for obj in self.objects:
            obj.rotate_end()
        self.rot_origin = None
        self.rotation = 0
        self.mod += 1

    def resize_update(self, bbox):
        """Resize the group of objects. we need to calculate the new
           bounding box for each object within the group"""
        orig_bbox = self.resizing["orig_bbox"]

        scale_x, scale_y = bbox[2] / orig_bbox[2], bbox[3] / orig_bbox[3]

        for obj in self.objects:
            obj_bb = self.resizing["objects"][obj]

            x, y, w, h = obj_bb
            w2, h2 = w * scale_x, h * scale_y

            x2 = bbox[0] + (x - orig_bbox[0]) * scale_x
            y2 = bbox[1] + (y - orig_bbox[1]) * scale_y

            ## recalculate the new bbox of the object within our new bb
            obj.resize_update((x2, y2, w2, h2))

        self.resizing["bbox"] = bbox
        self.mod += 1

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        for obj in self.objects:
            obj.resize_end()
        self.mod += 1

    def length(self):
        """Return the number of objects in the group."""
        return len(self.objects)

    def bbox(self, actual = False):
        """Return the bounding box of the group."""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.objects:
            return None

        left, top, width, height = self.objects[0].bbox(actual = actual)
        bottom, right = top + height, left + width

        for obj in self.objects[1:]:
            x, y, w, h = obj.bbox(actual = actual)
            left, top = min(left, x, x + w), min(top, y, y + h)
            bottom, right = max(bottom, y, y + h), max(right, x, x + w)

        width, height = right - left, bottom - top
        return (left, top, width, height)

    def add(self, obj):
        """Add an object to the group."""
        if obj not in self.objects:
            self.objects.append(obj)
        self.mod += 1

    def remove(self, obj):
        """Remove an object from the group."""
        self.objects.remove(obj)
        self.mod += 1

    def move(self, dx, dy):
        """Move the group by dx, dy."""
        for obj in self.objects:
            obj.move(dx, dy)
        self.mod += 1

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the group of objects on the Cairo context."""
        for obj in self.objects:
            obj.draw(cr, hover=False, selected=selected)

        cr.set_source_rgb(0, 0, 0)

        if self.rotation:
            cr.save()
            x, y = self.rot_origin[0], self.rot_origin[1]
            cr.translate(x, y)
            cr.rotate(self.rotation)
            cr.translate(-x, -y)

        if selected:
            cr.set_source_rgb(1, 0, 0)
            self.bbox_draw(cr, lw=.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

        if self.rotation:
            cr.restore()

class SelectionObject(DrawableGroup):
    """
    Class for handling the selection of objects.

    It is an extension of the DrawableGroup class, with additional methods for
    selecting and manipulating objects. Note that more often than not, the
    methods in this class need to have access to the global list of all
    object (e.g. to inverse a selection).

    Attributes:
        objects (list): The list of selected objects.
        _all_objects (list): The list of all objects in the canvas.
    """

    def __init__(self, all_objects):
        super().__init__([ ], None, mytype = "selection_object")

        print("Selection Object with ", len(all_objects), "objects")
        self._all_objects = all_objects

    def copy(self):
        """Return a copy of the selection object."""
        # the copy can be used for undo operations
        print("copying selection to a new selection object")
        return DrawableGroup(self.objects[:])

    def n(self):
        """Return the number of objects in the selection."""
        return len(self.objects)

    def is_empty(self):
        """Check if the selection is empty."""
        return not self.objects

    def clear(self):
        """Clear the selection."""
        self.objects = [ ]

    def set(self, objects):
        """Set the selection to a list of objects."""
        print("setting selection to", objects)
        self.objects = objects

    def add(self, obj):
        """Add an object to the selection."""
        print("adding object to selection:", obj, "selection is", self.objects)
        if not obj in self.objects:
            self.objects.append(obj)

    def all(self):
        """Select all objects."""
        print("selecting everything")
        self.objects = self._all_objects[:]
        print("selection has now", len(self.objects), "objects")
        print("all objects have", len(self._all_objects), "objects")

    def next(self):
        """
        Return a selection object with the next object in the list,
        relative to the current selection.
        """

        all_objects = self._all_objects

        if not all_objects:
            return

        if not self.objects:
            self.objects = [ all_objects[0] ]
            return

        idx = all_objects.index(self.objects[-1])
        idx += 1
        if idx >= len(all_objects):
            idx = 0

        self.objects = [ all_objects[idx] ]


    def prev(self):
        """
        Return a selection object with the previous object in the list,
        relative to the current selection.
        """

        all_objects = self._all_objects

        if not all_objects:
            return

        if not self.objects:
            self.objects = [ all_objects[-1] ]
            return

        idx = all_objects.index(self.objects[-1])
        idx -= 1
        if idx < 0:
            idx = len(all_objects) - 1
        self.objects = [ all_objects[idx] ]


    def reverse(self):
        """
        Return a selection object with the objects in reverse order.
        """
        if not self.objects:
            print("no selection yet, selecting everything")
            self.objects = self._all_objects[:]
            return

        new_sel = [ ]
        for obj in self._all_objects:
            if not self.contains(obj):
                new_sel.append(obj)

        self.objects = new_sel

Drawable.register_type("group", DrawableGroup)
