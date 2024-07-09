"""
This module provides functions to import and export drawings in various formats.
"""

import logging                                                   # <remove>
from os import path # <remove>
import pickle       # <remove>
import yaml         # <remove>

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position

import cairo                                # <remove>
from sd.drawable import Drawable            # <remove>
from sd.drawable_group import DrawableGroup # <remove>
from sd.page import Page                    # <remove>
log = logging.getLogger(__name__)                                # <remove>

def __draw_object(cr, obj, bg, bbox, transparency):
    """Draw an object on a Cairo context."""

    cr.save()

    # we translate the object to the origin of the context
    cr.translate(-bbox[0], -bbox[1])

    # paint the background
    cr.set_source_rgba(*bg, transparency)
    cr.paint()

    obj.draw(cr)

    cr.restore()


def __find_max_width_height(obj_list):
    """Find the maximum width and height of a list of objects."""
    width, height = None, None

    for o in obj_list:
        bb = o.bbox()

        if bb is None:
            continue

        if not width or not height:
            width, height = bb[2], bb[3]
            continue

        width  = max(width, bb[2])
        height = max(height, bb[3])

    return width, height

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
    """
    Convert a drawing from the internal format to another format.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to. If "any", the
    format will be guessed from the file extension.
    :param border: The border around the objects.
    :param page_no: The page number to export (if the drawing has multiple pages).
    """
    log.debug("Converting file %s to %s as %s page_no=%s",
              input_file, output_file, file_format, page_no)
    if file_format == "any":
        if output_file is None:
            raise ValueError("No output file format provided")
        file_format = guess_file_format(output_file)
    else:
        if output_file is None:
            # create a file name with the same name but different extension
            output_file = path.splitext(input_file)[0] + "." + file_format

    # if we have a page specified, we need to convert to pdf using the
    # one-page converter
    if file_format in [ "png", "jpeg", "svg" ] or (file_format == "pdf"
                                                   and page_no is not None):
        if not page_no:
            page_no = 0
        convert_file_to_image(input_file, output_file, file_format, border, page_no)

   #elif file_format == "yaml":
   #    print("Exporting to yaml")
   #    export_file_as_yaml(output_file, config, objects=objects.to_dict())

    elif file_format == "pdf":
        log.debug("Converting to multipage PDF")
        convert_to_multipage_pdf(input_file, output_file, border)
    else:
        raise NotImplementedError("Conversion to " + file_format + " is not implemented")

def convert_file_to_image(input_file, output_file, file_format = "png", border = None, page_no = 0):
    """
    Convert a drawing to an image file: png, jpeg, svg.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to.
    :param border: The border around the objects.
    :param page_no: The page number to export (if the drawing has multiple pages).
    """
    if page_no is None:
        page_no = 0

    # yeah so we read the file twice, shoot me
    config, objects, pages = read_file_as_sdrw(input_file)

    if pages:
        if len(pages) <= page_no:
            raise ValueError(f"Page number out of range (max. {len(pages) - 1})")
        log.debug("read drawing from %s with %d pages",
                  input_file, len(pages))
        p = Page()
        p.import_page(pages[page_no])
        objects = p.objects_all_layers()

    log.debug("read drawing from %s with %d objects",
              input_file, len(objects))

    if not objects:
        log.warning("No objects found in the input file on page %s", page_no)
        return

    objects = DrawableGroup(objects)


    bbox = objects.bbox()
    if border:
        bbox = objects.bbox()
        bbox = (bbox[0] - border, bbox[1] - border, bbox[2] + 2 * border, bbox[3] + 2 * border)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    cfg = { "bg": bg, "bbox": bbox, "transparency": transparency, "border": border }

    export_image(objects,
                 output_file, file_format, cfg)

def convert_to_multipage_pdf(input_file, output_file, border = None):
    """
    Convert a native drawing to a multipage PDF file.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param border: The border around the objects.
    """
    log.debug("Converting to multipage PDF")
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

def export_objects_to_multipage_pdf(obj_list, output_file, config, border = 10):
    """
    Export a list of objects to a multipage PDF file.

    :param obj_list: A list of objects to export.
    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param border: The border around the objects.

    Each object in the list will be drawn on a separate page.
    """
    if not border:
        border = 0

    log.debug("Exporting %s objects to multipage PDF with border %s",
              len(obj_list), border)

    width, height = __find_max_width_height(obj_list)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    width, height = int(width + 2 * border), int(height + 2 * border)

    surface = cairo.PDFSurface(output_file, width, height)
    cr = cairo.Context(surface)

    cr.set_source_rgba(*bg, transparency)
    cr.paint()

    nobj = len(obj_list)

    # each object is a DrawableGroup for a single page
    for i, o in enumerate(obj_list):
        bb = o.bbox()

        # some pages might be empty
        if bb:
            cr.save()
            cr.translate(border - bb[0], border - bb[1])
            o.draw(cr)
            cr.restore()

        # do not show_page on the last page.
        if i < nobj - 1:
            surface.show_page()
    surface.finish()

def export_image_jpg(obj, output_file, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a JPEG file."""
    raise NotImplementedError("JPEG export is not implemented")

def export_image_pdf(obj, output_file, cfg):
    """
    Export the drawing to a single-page PDF file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    bbox = cfg.get("bbox", None)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.PDFSurface(output_file, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)
    surface.finish()

def export_image_svg(obj, output_file, cfg):
    """
    Export the drawing to a SVG file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    bbox = cfg.get("bbox", None)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.SVGSurface(output_file, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.finish()

def export_image_png(obj, output_file, cfg):
    """
    Export the drawing to a PNG file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    bbox = cfg.get("bbox", None)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.write_to_png(output_file)

def export_image(obj, output_file, file_format = "any", config = None, all_pages_pdf = False):
    """
    Export the drawing to a file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to. If "any", the
    format will be guessed from the file extension.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    :param all_pages_pdf: If True, all pages will be exported to a single PDF file.
    """
    if not config:
        config = { "bg": (1, 1, 1), "bbox": None, "transparency": 1.0, "border": None }

    log.debug("exporting to %s file %s all_pages_pdf: %s",
        file_format, output_file, all_pages_pdf)

    # if output_file is None, we send the output to stdout
    if output_file is None:
        log.debug("export_image: no output_file provided")
        return

    if file_format == "any":
        # get the format from the file name
        _, file_format = path.splitext(output_file)
        file_format = file_format[1:]
        # lower case
        file_format = file_format.lower()
        # jpg -> jpeg
        if file_format == "jpg":
            file_format = "jpeg"
        # check
        if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
            raise ValueError("Unrecognized file extension")
        log.debug("guessing format from file name: %s",
            file_format)

    # Create a Cairo surface of the same size as the bbox
    if file_format == "png":
        export_image_png(obj, output_file, config)
    elif file_format == "svg":
        export_image_svg(obj, output_file, config)
    elif file_format == "pdf":
        if all_pages_pdf:
            export_objects_to_multipage_pdf(obj, output_file, config, border=10)
        else:
            export_image_pdf(obj, output_file, config)
    else:
        raise NotImplementedError("Export to " + file_format + " is not implemented")

def export_file_as_yaml(output_file, config, objects = None, pages = None):
    """
    Save the objects to a YAML file.

    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param objects: The objects to save (dict).
    :param pages: The pages to save (dict).

    Pages and Drawable objects need to be converted to dictionaries before
    saving them to a file using their to_dict() method.
    """

    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(output_file, 'w', encoding = 'utf-8') as f:
            yaml.dump(state, f)
        log.debug("Saved drawing to %s", output_file)
        return True
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return False
    except yaml.YAMLError as e:
        log.warning("Error saving file because: %s", e)
        return False


# ------------------- handling of the native format -------------------

def save_file_as_sdrw(output_file, config, objects = None, pages = None):
    """
    Save the objects to a file in native format.

    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param objects: The objects to save (dict).
    :param pages: The pages to save (dict).

    Pages and Drawable objects need to be converted to dictionaries before
    saving them to a file using their to_dict() method.
    """
    # objects are here for backwards compatibility only
    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(output_file, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        log.debug("Saved drawing to %s", output_file)
        return True
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return False
    except pickle.PicklingError as e:
        log.warning("Error saving file, an object could not be pickled: %s", e)
        return False

def read_file_as_sdrw(input_file):
    """
    Read the objects from a file in native format.

    :param input_file: The name of the file to read from.
    """
    if not path.exists(input_file):
        log.warning("No saved drawing found at %s", input_file)
        return None, None, None

    log.debug("READING file as sdrw: %s", input_file)

    config, objects, pages = None, None, None

    try:
        with open(input_file, "rb") as file:
            state = pickle.load(file)
            if "objects" in state:
                log.debug("found objects in savefile")
                objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            if "pages" in state:
                log.debug("found pages in savefile")
                pages = state['pages']
                for p in pages:
                    # this is for compatibility; newer drawings are saved
                    # with a "layers" key which is then processed by the
                    # page import function - best if page takes care of it
                    if "objects" in p:
                        p['objects'] = [ Drawable.from_dict(d) for d in p['objects'] ] or [ ]

            config = state['config']
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return None, None, None
    except pickle.PicklingError as e:
        log.warning("Error saving file because an object could not be pickled: %s", e)
        return None, None, None
    return config, objects, pages
