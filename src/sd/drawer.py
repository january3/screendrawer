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
            return

        grp = DrawableGroup(objects)
        bb  = grp.bbox()
        if not bb:
            print("empty bb")
            return
        x, y, width, height = bb
        self.__cache = {
                "surface": cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1),
                "x": x,
                "y": y,
                }
        cr_tmp = cairo.Context(self.__cache["surface"])
        cr_tmp.translate(-x, -y)
        grp.draw(cr_tmp)

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

        newhash = { }
        active = [ ]
        for obj in objects:
            hover    = obj == hover_obj and mode == "move"
            selected = selection.contains(obj) and mode == "move"
            if not hover and not selected:
                newhash[obj] = obj.mod
            else:
                active.append(obj)

        if newhash != self.__obj_mod_hash:
            self.__obj_mod_hash = newhash
            self.__cache = None
        else:
            if not self.__cache:
                self.cache(list(newhash.keys()))

        if self.__cache:
            self.paint_cache(cr)
        else:
            active = objects

        for obj in active:
            hover    = obj == hover_obj and mode == "move"
            selected = selection.contains(obj) and mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = outline)
