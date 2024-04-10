"""
This module provides functions to import and export drawings in various formats.
"""

from os import path # <remove>
import pickle       # <remove>
import yaml         # <remove>

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position

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
    if file_format not in [ "png", "jpeg", "pdf", "svg", "yaml" ]:
        raise ValueError("Unrecognized file extension")
    return file_format

def convert_file(input_file, output_file, file_format = "any", border = None, page_no = None):
    """Convert a drawing from the internal format to another."""
    print("page_no = ", page_no)

    if file_format == "any":
        if output_file is None:
            raise ValueError("No output file format provided")
        file_format = guess_file_format(output_file)
    else:
        if output_file is None:
            # create a file name with the same name but different extension
            output_file = path.splitext(input_file)[0] + "." + file_format

    config, objects, pages = read_file_as_sdrw(input_file)

    # if there is only one page, we can use the regular converter for pdfs
    if page_no is None and pages and len(pages) == 1:
        page_no = 0

    # also when we have objects, and not pages, we can use the regular
    # converter for pdfs
    if not pages:
        page_no = 0

    if file_format in [ "png", "jpeg", "svg" ] or (file_format == "pdf"
                                                   and page_no is not None):
        convert_file_to_image(input_file, output_file, file_format, border, page_no)
    elif file_format == "yaml":
        print("Exporting to yaml")
        export_file_as_yaml(output_file, config, objects=objects.to_dict())
    elif file_format == "pdf":
        convert_to_multipage_pdf(input_file, output_file, border)
    else:
        raise NotImplementedError("Conversion to " + file_format + " is not implemented")

def convert_file_to_image(input_file, output_file, file_format = "png", border = None, page_no = 0):
    """Convert a drawing to an image file: png, jpeg, svg."""
    if page_no is None:
        page_no = 0

    # yeah so we read the file twice, shoot me
    config, objects, pages = read_file_as_sdrw(input_file)

    if pages:
        if len(pages) <= page_no:
            raise ValueError(f"Page number out of range (max. {len(pages) - 1})")
        print("read drawing from", input_file, "with", len(pages), "pages")
        p = Page()
        p.import_page(pages[page_no])
        objects = p.objects_all_layers()

    print("read drawing from", input_file, "with", len(objects), "objects")
    objects = DrawableGroup(objects)

    bbox = objects.bbox()
    if border:
        bbox = objects.bbox()
        bbox = (bbox[0] - border, bbox[1] - border, bbox[2] + 2 * border, bbox[3] + 2 * border)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    export_image(objects,
                 output_file, file_format,
                 bg = bg, bbox = bbox,
                 transparency = transparency)

def convert_to_multipage_pdf(input_file, output_file, border = None):
    """Convert a drawing to a multipage PDF file."""
    print("Converting to multipage PDF")
    config, _, pages = read_file_as_sdrw(input_file)
    if not pages:
        raise ValueError("No multiple pages found in the input file")

    page_obj = []

    for p in pages:
        page = Page()
        page.import_page(p)
        obj_grp = DrawableGroup(page.objects_all_layers())
        page_obj.append(obj_grp)

    export_objects_to_multipage_pdf(page_obj, output_file, config, border)

def find_max_width_height(obj_list):
    """Find the maximum width and height of a list of objects."""
    width, height = None, None

    for o in obj_list:
        bb = o.bbox()
        if not width or not height:
            width, height = bb[2], bb[3]
            continue
        width = max(width, bb[2])
        height = max(height, bb[3])

    return width, height

def export_objects_to_multipage_pdf(obj_list, output_file, config, border = None):
    """
    Export a list of objects to a multipage PDF file.

    Each object in the list will be drawn on a separate page.
    """
    if not border:
        border = 0

    width, height = find_max_width_height(obj_list)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    width, height = int(width + 2 * border), int(height + 2 * border)

    surface = cairo.PDFSurface(output_file, width, height)
    cr = cairo.Context(surface)

    cr.set_source_rgba(*bg, transparency)
    cr.paint()

    nobj = len(obj_list)

    for i, o in enumerate(obj_list):
        bb = o.bbox()

        cr.save()
        cr.translate(border - bb[0], border - bb[1])
        o.draw(cr)
        cr.restore()
        if i < nobj - 1:
            surface.show_page()
    surface.finish()

def __draw_object(cr, obj, bg, bbox, transparency):
    """Draw an object on a Cairo context."""

    cr.save()
    cr.translate(-bbox[0], -bbox[1])
    cr.set_source_rgba(*bg, transparency)
    cr.paint()
    obj.draw(cr)
    cr.restore()

def export_image_jpg(obj, filename, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a JPEG file."""
    raise NotImplementedError("JPEG export is not implemented")

def export_image_pdf(obj, filename, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a SVG file."""

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.PDFSurface(filename, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)
    surface.finish()

def export_image_svg(obj, filename, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a SVG file."""

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.SVGSurface(filename, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.finish()

def export_image_png(obj, filename, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a PNG file."""

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.write_to_png(filename)

def export_image(objects, filename,
                 file_format = "any",
                 bg = (1, 1, 1),
                 bbox = None,
                 transparency = 1.0):
    """Export the drawing to a file."""

    # if filename is None, we send the output to stdout
    if filename is None:
        print("export_image: no filename provided")
        return

    if file_format == "any":
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

    # Create a Cairo surface of the same size as the bbox
    if file_format == "png":
        export_image_png(objects, filename, bg, bbox, transparency)
    elif file_format == "svg":
        export_image_svg(objects, filename, bg, bbox, transparency)
    elif file_format == "pdf":
        export_image_pdf(objects, filename, bg, bbox, transparency)
    else:
        raise NotImplementedError("Export to " + file_format + " is not implemented")

def export_file_as_yaml(filename, config, objects = None, pages = None):
    """Save the objects to a YAML file."""

    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(filename, 'w') as f:
            yaml.dump(state, f)
        print("Saved drawing to", filename)
        return True
    except OSError as e:
        print(f"Error saving file due to a file I/O error: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"Error saving file because: {e}")
        return False


# ------------------- handling of the native format -------------------

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
