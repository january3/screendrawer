"""Class which draws the actual objects and caches them."""
import cairo
from .drawable import DrawableGroup

def draw_on_surface(cr, objects, selection, state):
    """
    Draw the objects on the given graphical context.
    """

    for obj in objects:

        hover    = obj == state.hover_obj() and state.mode() == "move"
        selected = selection.contains(obj) and state.mode() == "move"

        obj.draw(cr, hover=hover,
                 selected=selected,
                 outline = state.outline())

def obj_status(obj, selection, state):
    """Calculate the status of an object."""

    hover_obj = state.hover_obj()
    hover    = obj == hover_obj and state.mode() == "move"
    selected = selection.contains(obj) and state.mode() == "move"

    return (obj.mod, hover, selected)

def create_cache_surface(objects):
    """
    Create a cache surface.

    :param objects: The objects to cache.
    """

    if not objects:
        return None

    grp = DrawableGroup(objects)
    bb  = grp.bbox(actual = True)

    if not bb:
        return None

    # create a surface that fits the bounding box of the objects
    x, y, width, height = bb
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

    def new_cache(self, groups, selection, state):
        """
        Generate the new cache when the objects have changed.

        :param groups: The groups of objects to cache.
        """

        self.__cache = { "groups": groups,
                         "surfaces": [ ],}

        cur = groups["first_is_same"]

        # for each non-empty group of objects that remained the same,
        # generate a cache surface and draw the group on it
        for obj_grp in groups["groups"]:
            if not cur or not obj_grp:
                cur = not cur
                continue

            surface = create_cache_surface(obj_grp)
            self.__cache["surfaces"].append(surface)
            cr = surface["cr"]
            draw_on_surface(cr, obj_grp, selection, state)
            cur = not cur

    def update_cache(self, objects, selection, state):
        """
        Update the cache.

        :param objects: The objects to update.
        :param selection: The selection.
        :param state: The state.
        """

        groups = self.__find_groups(objects, selection, state)

        if not self.__cache or self.__cache["groups"] != groups:
            print("regenerating cache")
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
            status = obj_status(obj, selection, state)

            is_same = obj in modhash and modhash[obj] == status

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

        for obj_grp in self.__cache["groups"]["groups"]:
            #print("i=", i, "is_same=", is_same, "objects=", obj_grp)

            # ignore empty groups (that might happen in edge cases)
            if not obj_grp:
                is_same = not is_same
                continue

            # objects in this group have changed: draw it normally on the surface
            if not is_same:
                draw_on_surface(cr, obj_grp, selection, state)
                is_same = not is_same
                continue

            # objects in this group remained the same: draw the cached surface
            if is_same:
                surface = self.__cache["surfaces"][i]
                cr.set_source_surface(surface["surface"], surface["x"], surface["y"])
                cr.paint()
                i += 1
                is_same = not is_same

    def draw(self, cr, page, state):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param page: The page object.
        :param state: The state object.
        """

        # extract objects from the page provided
        objects = page.objects_all_layers()

        # extract the selection object that we need
        # to determine whether an object in selected state
        selection = page.selection()

        # check if the cache needs to be updated
        self.update_cache(objects, selection, state)

        # draw the cache
        self.draw_cache(cr, selection, state)

        return True
