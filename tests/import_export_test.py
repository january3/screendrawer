import pytest
from pathlib import Path
from sd.import_export import read_file_as_sdrw, save_file_as_sdrw
from sd.page import Page

def _check_file(config, objects, pages):
    """Check against the predefined file structure"""
    if not config == {'bg_color': (0.8, 0.75, 0.65), 'transparent': 0, 'show_wiglets': True, 'bbox': (0, 0, 1920, 1053), 'pen': {'color': (0.1549999999999998, 1.0, 0.7585714285714287), 'line_width': 19.415960693359374, 'transparency': 1, 'fill_color': None, 'font_size': 24, 'font_family': 'Sans', 'font_weight': 'normal', 'brush': 'slanted', 'font_style': 'normal'}, 'pen2': {'color': (1, 1, 0), 'line_width': 40, 'transparency': 0.2, 'fill_color': None, 'font_size': 24, 'font_family': 'Sans', 'font_weight': 'normal', 'brush': 'rounded', 'font_style': 'normal'}, 'page': 0}:
        print("Actual config found:")
        print(config)
        assert False, "Config is incorrect"

    assert len(pages) == 2, "Number of pages is incorrect"
    assert "layers" in pages[0], "Layers not found in page"
    layers = pages[0]["layers"]
    assert len(layers) == 3, "Number of layers is incorrect"
    assert len(layers[0]) == 3, "Number of objects in layer 0 is incorrect"
    assert len(layers[1]) == 2, "Number of objects in layer 1 is incorrect"
    assert len(layers[2]) == 2, "Number of objects in layer 2 is incorrect"
    assert objects is None, "Objects should be None"


def test_read_file_as_sdrw(tmp_path):
    """Check whether a file can be read correctly"""

    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"

    config, objects, pages = read_file_as_sdrw(file_in)
    _check_file(config, objects, pages)

def test_write_file_as_sdrw(tmp_path):
    """
    This test checks that a file can be written to disk and then read back.

    This does not check whether the objects can be imported correctly.
    """

    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    file_out = tmp_path / "output"

    config, objects, pages = read_file_as_sdrw(file_in)
    save_file_as_sdrw(file_out, config, objects = None, pages = pages)

    # check that file exists
    assert file_out.exists(), "Output file was not created"

    # check that the file is not empty
    assert file_out.stat().st_size > 0, "File is empty"

    config, objects, pages = read_file_as_sdrw(file_out)
    _check_file(config, objects, pages)

def test_that_page_can_be_imported():
    """
    This test checks that objects can actually be imported from a file.

    We use Page().import_page to import the objects, and then check whether 
    the objects are correctly imported: whether there the number of objects
    is correct, whether there is a correct number of layers, and whether the
    current layer is correct. We also manipulate the layers and pages
    through the Page() interface.
    """

    start_page = Page()

    test_data_dir = Path(__file__).parent / "data"
    file_in = test_data_dir / "foobar.sdrw"
    config, objects, pages = read_file_as_sdrw(file_in)
    start_page.import_page(pages[0])

    # current layer number
    assert start_page.layer_no() == 2, "Current layer number is incorrect"
    assert start_page.number_of_layers() == 3, "Number of layers is incorrect"
    assert start_page.translate() == (73.44677734375, 22.890167236328125), "Translate is incorrect"
    obj = start_page.objects_all_layers()
    assert len(obj) == 7, "Number of objects is incorrect"

    # just the current layer objects
    obj = start_page.objects()
    assert len(obj) == 2, "Number of objects on start page is incorrect"
    start_page.kill_object(obj[0])
    assert len(obj) == 1, "Number of objects on start page is incorrect after killing one object"
    start_page.prev_layer()
    obj = start_page.objects()
    assert len(obj) == 2, "Number of objects is incorrect after switching to previous layer"
    start_page.delete_layer()
    assert start_page.number_of_layers() == 2, "Number of layers is incorrect"
    start_page.prev_layer()
    assert start_page.layer_no() == 0, "Current layer number is incorrect"

