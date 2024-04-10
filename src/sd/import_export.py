"""
This module provides functions to import and export drawings in various formats.
"""

from os import path # <remove>
import pickle       # <remove>

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib # <remove> pylint: disable=wrong-import-position

import cairo                                # <remove>
from sd.drawable import Drawable            # <remove>
from sd.drawable_group import DrawableGroup # <remove>
from sd.page import Page                    # <remove>

def guess_file_format(filename):
    """Guess the file format from the file extension."""
    _, file_format = path.splitext(filename)
    file_format = file_format[1:]
    # lower case
    file_format = file_format.lower()
    # jpg -> jpeg
    if file_format == "jpg":
        file_format = "jpeg"
    # check
    if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
        raise ValueError("Unrecognized file extension")
    return file_format

def convert_file(input_file, output_file, file_format = "all", border = None, page = 0):
    """Convert a drawing from the internal format to another."""
    config, objects, pages = read_file_as_sdrw(input_file)
    print("page = ", page)

    if pages:
        if len(pages) <= page:
            raise ValueError(f"Page number out of range (max. {len(pages) - 1})")
        print("read drawing from", input_file, "with", len(pages), "pages")
        p = Page()
        p.import_page(pages[page])
        objects = p.objects_all_layers()

    print("read drawing from", input_file, "with", len(objects), "objects")
    objects = DrawableGroup(objects)

    bbox = config.get("bbox", None) or objects.bbox()
    if border:
        bbox = objects.bbox()
        bbox = (bbox[0] - border, bbox[1] - border, bbox[2] + 2 * border, bbox[3] + 2 * border)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    if file_format == "all":
        if output_file is None:
            raise ValueError("No output file format provided")
        file_format = guess_file_format(output_file)
    else:
        if output_file is None:
            # create a file name with the same name but different extension
            output_file = path.splitext(input_file)[0] + "." + file_format

    export_image(objects,
                 output_file, file_format,
                 bg = bg, bbox = bbox,
                 transparency = transparency)


def export_image(objects, filename,
                 file_format = "all",
                 bg = (1, 1, 1),
                 bbox = None,
                 transparency = 1.0):
    """Export the drawing to a file."""

    # if filename is None, we send the output to stdout
    if filename is None:
        print("export_image: no filename provided")
        return

    if file_format == "all":
        # get the format from the file name
        _, file_format = path.splitext(filename)
        file_format = file_format[1:]
        # lower case
        file_format = file_format.lower()
        # jpg -> jpeg
        if file_format == "jpg":
            file_format = "jpeg"
        # check
        if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
            raise ValueError("Unrecognized file extension")
        print("export_image: guessing format from file name:", file_format)

    if bbox is None:
        bbox = objects.bbox()
    print("Bounding box:", bbox)
    # to integers
    width, height = int(bbox[2]), int(bbox[3])


    # Create a Cairo surface of the same size as the bbox
    if file_format == "png":
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    elif file_format == "svg":
        surface = cairo.SVGSurface(filename, width, height)
    elif file_format == "pdf":
        surface = cairo.PDFSurface(filename, width, height)
    else:
        raise ValueError("Invalid file format: " + file_format)

    cr = cairo.Context(surface)
    # translate to the top left corner of the bounding box
    cr.translate(-bbox[0], -bbox[1])

    cr.set_source_rgba(*bg, transparency)
    cr.paint()
    objects.draw(cr)

    # Save the surface to the file
    if file_format == "png":
        surface.write_to_png(filename)
    elif file_format in [ "svg", "pdf" ]:
        surface.finish()

def save_file_as_sdrw(filename, config, objects = None, pages = None):
    """Save the objects to a file in native format."""
    # objects are here for backwards compatibility only
    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(filename, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        print("Saved drawing to", filename)
        return True
    except OSError as e:
        print(f"Error saving file due to a file I/O error: {e}")
        return False
    except pickle.PicklingError as e:
        print(f"Error saving file because an object could not be pickled: {e}")
        return False

def read_file_as_sdrw(filename):
    """Read the objects from a file in native format."""
    if not path.exists(filename):
        print("No saved drawing found at", filename)
        return None, None, None

    print("READING file as sdrw:", filename)

    config, objects, pages = None, None, None

    try:
        with open(filename, "rb") as file:
            state = pickle.load(file)
            if "objects" in state:
                print("found objects in savefile")
                objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            if "pages" in state:
                print("found pages in savefile")
                pages = state['pages']
                for p in pages:
                    # this is for compatibility; newer drawings are saved
                    # with a "layers" key which is then processed by the
                    # page import function - best if page takes care of it
                    if "objects" in p:
                        p['objects'] = [ Drawable.from_dict(d) for d in p['objects'] ] or [ ]
                            
            config = state['config']
    except OSError as e:
        print(f"Error saving file due to a file I/O error: {e}")
        return None, None, None
    except pickle.PicklingError as e:
        print(f"Error saving file because an object could not be pickled: {e}")
        return None, None, None
    return config, objects, pages
