"""Class which draws the actual objects and caches them."""
import logging                                # <remove>
import gi                                     # <remove>
gi.require_version('Gtk', '3.0')              # <remove> pylint: disable=wrong-import-position
import cairo                                  # <remove>
from .drawable_group import DrawableGroup     # <remove>
from .drawable_group import ClippingGroup     # <remove>
from .drawable_primitives import Rectangle    # <remove>
from .pen import Pen                          # <remove>
from .utils import bbox_is_overlap            # <remove>
log = logging.getLogger(__name__)             # <remove>
log.setLevel(logging.INFO)                    # <remove>

def draw_on_surface(cr, objects, selection, state):
    """
    Draw the objects on the given graphical context.
    """

    for obj in objects:

        hover    = obj == state.hover_obj() and state.mode() == "move"
        selected = selection.contains(obj) and state.mode() == "move"

        obj.draw(cr, hover=hover,
                 selected=selected,
                 outline = state.graphics().outline())

def obj_status(obj, selection, state):
    """Calculate the status of an object."""

    hover_obj = state.hover_obj()
    hover    = obj == hover_obj and state.mode() == "move"
    selected = selection.contains(obj) and state.mode() == "move"
    is_cur_obj = obj == state.current_obj()

    return (obj.mod, hover, selected, is_cur_obj)

def create_cache_surface(objects, mask = None, trafo = None):
    """
    Create a cache surface.

    :param objects: The objects to cache.
    """

    if not objects:
        return None

    if mask:
        grp = ClippingGroup(mask, objects)
    else:
        grp = DrawableGroup(objects)

    bb  = grp.bbox(actual = True)

    if not bb:
        return None

    x, y, width, height = bb

    if width <= 0 or height <= 0:
        #log.debug("no bb overlap with mask, skipping")
        #log.debug("clipgroup bbox: %s", grp.bbox(actual = True))
        return None

    if trafo:
        bb = [ (x, y), (x + width, y + height) ]
        bb = trafo.apply(bb)
        x, y, width, height = bb[0][0], bb[0][1], bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]
        #log.debug("surface size: %d x %d", int(width), int(height))

    # create a surface that fits the bounding box of the objects
    surface = cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1)
    cr = cairo.Context(surface)
    cr.translate(-x, -y)
    ret = {
            "surface": surface,
            "cr": cr,
            "x": x,
            "y": y,
            }
    return ret



class Drawer:
    """Singleton Class which draws the actual objects and caches them."""
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__new_instance is None:
            cls.__new_instance = super(Drawer, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self):
        self.__cache        = None
        self.__obj_mod_hash = { }
        self.__trafo        = None
        self.__win_size     = None
        self.__win_pos_rel  = None
        self.__mask         = None
        self.__mask_bbox    = None

    def new_cache(self, groups, selection, state):
        """
        Generate the new cache when the objects have changed.

        :param groups: The groups of objects to cache.
        """

        #log.debug("generating new cache")
        self.__cache = { "groups": groups,
                         "surfaces": [ ],}

        cur = groups["first_is_same"]

        # for each non-empty group of objects that changed
        # generate a cache surface and draw the group on it
        for obj_grp in groups["groups"]:
            if not cur or not obj_grp:
                cur = not cur
                continue

            surface = create_cache_surface(obj_grp, mask = self.__mask, trafo = self.__trafo)
            self.__cache["surfaces"].append(surface)
            if surface:
                cr = surface["cr"]
                self.__trafo.transform_context(cr)
                draw_on_surface(cr, obj_grp, selection, state)
            cur = not cur

    def update_cache(self, objects, selection, state, force_redraw):
        """
        Update the cache.

        :param objects: The objects to update.
        :param selection: The selection.
        :param state: The state.
        """

        if force_redraw:
            self.__obj_mod_hash = { }

        groups = self.__find_groups(objects, selection, state)

       #if self.__cache:
       #    if self.__cache["groups"] != groups:
       #        log.debug("groups have changed!")
       #        log.debug("cached: %s", self.__cache["groups"])
       #        log.debug("new: %s", groups)

        if not self.__cache or self.__cache["groups"] != groups:
            self.new_cache(groups, selection, state)

    def __find_groups(self, objects, selection, state):
        """
        Method to detect which objects changed from the previous time.
        These objects are then split into groups separated by objects that
        did change, so when drawing, the stacking order is maintained
        despite cacheing.

        :param objects: The objects to find groups for.
        :param selection: The selection object to determine whether an
                           object is selected (selected state is drawn
                           differently, so that counts as a change).
        :param state: The state object, holding information about the
                      drawing mode and hover object.
        
        Two values are returned. First, a list of groups, alternating
        between groups that have changed and groups that haven't. Second,
        a boolean indicating whether the first group contains objects
        that have changed.
        """
        modhash = self.__obj_mod_hash

        cur_grp       = [ ]
        groups        = [ cur_grp ]
        first_is_same = None

        is_same = True
        prev    = None

        # The goal of this method is to ensure correct stacking order
        # of the drawn active objects and cached groups.
        for obj in objects:
            bb = obj.bbox(actual = True)

            if not bbox_is_overlap(bb, self.__mask_bbox):
                continue

            status = obj_status(obj, selection, state)

            is_same = obj in modhash and modhash[obj] == status and not status[3]
            is_same = is_same and not obj.modified()

            #log.debug("object of type %s is same: %s (status=%s)", obj.type, is_same, status)

            if first_is_same is None:
                first_is_same = is_same

            # previous group type was different
            if prev is not None and prev != is_same:
                cur_grp = [ ]
                groups.append(cur_grp)

            cur_grp.append(obj)

            modhash[obj] = status
            prev = is_same

        ret = { "groups": groups, "first_is_same": first_is_same }
        return ret

    def draw_cache(self, cr, selection, state):
        """
        Process the cache. Draw the cached objects as surfaces and the rest
        normally.
        """

        is_same = self.__cache["groups"]["first_is_same"]
        i = 0
        n_cached = 0
        n_groups = 0

        for obj_grp in self.__cache["groups"]["groups"]:
            n_groups += 1

            # ignore empty groups (that might happen in edge cases)
            if not obj_grp:
                is_same = not is_same
                continue

            # objects in this group have changed: draw it normally on the surface
            if not is_same:
                cr.save()
                self.__trafo.transform_context(cr)
                draw_on_surface(cr, obj_grp, selection, state)
                cr.restore()
                is_same = not is_same
                continue

            # objects in this group remained the same: draw the cached surface
            if is_same:
                #print("Drawing cached surface")
                surface = self.__cache["surfaces"][i]
                if surface:
                    cr.set_source_surface(surface["surface"], surface["x"], surface["y"])
                    cr.paint()
                i += 1
                is_same = not is_same
                n_cached += 1
        cr.save()
       #self.__trafo.transform_context(cr)
       #self.__mask.draw(cr)
       #cr.restore()
       #log.debug("Cached %d groups out of %d", n_cached, n_groups)

    def draw(self, cr, page, state, force_redraw=False):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param page: The page object.
        :param state: The state object.
        """

        if force_redraw:
            log.debug("Forced redraw")

        self.__win_size = state.graphics().win_size()
        self.__trafo = page.trafo()
        wrelpos = self.__trafo.apply_reverse([(0, 0),
                                              self.__win_size])
        x0, y0 = wrelpos[0]
        x1, y1 = wrelpos[1]
        self.__mask_bbox = [ x0, y0, x1, y1 ]
        self.__mask = Rectangle([ (x0, y0), (x1, y0), 
                                 (x1, y1), (x0, y1),
                                 (x0, y0) ],
                                pen = Pen(color = (0, 1, 1)))
        #log.debug("wsize: %s wrelpos: %s",
                  #self.__win_size,
                  #wrelpos)


        #log.debug("Drawing objects on the page, force_redraw=%s", force_redraw)
        # extract objects from the page provided
        objects = page.objects_all_layers()

        # extract the selection object that we need
        # to determine whether an object in selected state
        selection = page.layer().selection()

        # check if the cache needs to be updated
        self.update_cache(objects, selection, state, force_redraw)


        # draw the cache
        cr.save()
        #page.trafo().transform_context(cr)
        self.draw_cache(cr, selection, state)
        cr.restore()

        return True
