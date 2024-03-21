import cairo # <remove>
from os import path # <remove>
import pickle # <remove>
from sd.drawable import Drawable # <remove>

def export_image(width, height, filename, draw_func, file_format = "all"):
    """Export the drawing to a file."""
    # Create a Cairo surface of the same size as the window content

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

    if file_format == "png":
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    elif file_format == "svg":
        surface = cairo.SVGSurface(filename, width, height)
    elif file_format == "pdf":
        surface = cairo.PDFSurface(filename, width, height)
    else:
        raise ValueError("Invalid file format: " + file_format)

    cr = cairo.Context(surface)
    cr.set_source_rgba(1, 1, 1)
    cr.paint()
    draw_func(cr)

    # Save the surface to the file
    if file_format == "png":
        surface.write_to_png(filename)
    elif file_format in [ "svg", "pdf" ]:
        surface.finish()

def save_file_as_sdrw(filename, config, objects):
    """Save the objects to a file."""
    state = { 'config': config, 'objects': objects }
    try:
        with open(filename, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        print("Saved drawing to", filename)
        return True
    except Exception as e:
        print("Error saving file:", e)
        return False



def read_file_as_sdrw(filename):
    """Read the objects from a file."""
    if not path.exists(filename):
        print("No saved drawing found at", filename)
        return None, None

    config, objects = None, None

    try:
        with open(filename, "rb") as file:
            state = pickle.load(file)
            objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            config = state['config']
    except Exception as e:
        print("Error reading file:", e)
        return None, None
    return config, objects

