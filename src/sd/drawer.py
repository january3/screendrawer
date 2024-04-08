"""Class which draws the actual objects and caches them."""
import cairo
from .drawable import DrawableGroup


class Drawer:
    """Class which draws the actual objects and caches them."""
    def __init__(self):
        self.__cache        = None
        self.__prev_outline = False
        self.__obj_mod_hash = { }

    def cache(self, objects):
        """
        Cache the objects.

        :param objects: The objects to cache.
        """

        if not objects:
            print("empty objects")
            return None

        grp = DrawableGroup(objects)
        bb  = grp.bbox()
        if not bb:
            print("empty bb")
            return None
        x, y, width, height = bb
        self.__cache = {
                "surface": cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1),
                "objects": objects,
                "x": x,
                "y": y,
                }
        cr_tmp = cairo.Context(self.__cache["surface"])
        cr_tmp.translate(-x, -y)
        return cr_tmp
        #grp.draw(cr_tmp)

    def paint_cache(self, cr):
        """
        Paint the cache.

        :param cr: The Cairo context to paint to.
        """

        cr.set_source_surface(self.__cache["surface"], self.__cache["x"], self.__cache["y"])
        cr.paint()

    def draw(self, cr, objects, selection, hover_obj, outline, mode):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param selection: The selection.
        :param hover_obj: The object the mouse is hovering over.
        :param outline: Whether to draw the outline.
        :param mode: The drawing mode.
        """

        modhash = self.__obj_mod_hash
        active  = [ ]
        same    = [ ]
        changed = [ ]
        for obj in objects:
            hover    = obj == hover_obj and mode == "move"
            selected = selection.contains(obj) and mode == "move"

            if not obj in modhash or modhash[obj] != [ obj.mod, hover, selected ]:
                changed.append(obj)
            else:
                same.append(obj)

            modhash[obj] = [ obj.mod, hover, selected ]

        print("changed", len(changed), "same", len(same))
        if not self.__cache or self.__cache["objects"] != same:
            print("caching", len(same), "objects")
            self.__cache = None
            cr_tmp = self.cache(same)
            if cr_tmp:
                self.draw_surface(cr_tmp, same, selection, hover_obj, outline, mode)

        if not self.__cache:
            active = objects
        else:
            self.paint_cache(cr)
            print("drawing cache")
            active = changed
            print("drawing", len(changed), "changed objects")

        print("drawing", len(active), "objects")
        self.draw_surface(cr, active, selection, hover_obj, outline, mode)

    def draw_surface(self, cr, objects, selection, hover_obj, outline, mode):
        """
        Draw the objects on the page.
        """
        for obj in objects:
            hover    = obj == hover_obj and mode == "move"
            selected = selection.contains(obj) and mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = outline)
