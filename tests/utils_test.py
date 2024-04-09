# test_utils.py
import math
import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.utils import rgb_to_hex, distance, get_default_savefile
from sd.utils import flatten_and_unique, sort_by_stack
from sd.utils import get_screenshot, segment_intersection, pp

def test_rgb_to_hex():
    assert rgb_to_hex((0, 0, 0)) == "#000000", "Black"
    assert rgb_to_hex((1, 1, 1)) == "#ffffff", "White"
    assert rgb_to_hex((0.5, 0.5, 0.5)) == "#7f7f7f", "Grey"

def test_distance():
    assert distance((0, 0), (0, 0)) == 0, "Distance to self"
    assert distance((0, 0), (1, 1)) == math.sqrt(2), "Unit distance"
    assert distance((-1, -1), (1, 1)) == 2*math.sqrt(2), "Quadrant distance"

def test_distance_with_negative_coordinates():
    assert distance((-1, -2), (-4, -6)) == 5, "Distance with negative coordinates"

def test_get_default_savefile(tmpdir):
    app_name = "TestApp"
    app_author = "TestAuthor"
    expected_path = tmpdir.mkdir("test_data_dir").join("savefile")
    # Mock the appdirs.user_data_dir to return our tmpdir
    with mock.patch('sd.utils.appdirs.user_data_dir', return_value=str(expected_path.dirname)):
        savefile = get_default_savefile(app_name, app_author)
    assert savefile == str(expected_path), "Default savefile path is incorrect"

def test_flatten_and_unique():
    nested_list = [1, [2, 3, [2, 4, 5]], 6, [1, [7]], 7]
    expected_result = [1, 2, 3, 4, 5, 6, 7]
    assert sorted(flatten_and_unique(nested_list)) == expected_result, "Failed to correctly flatten and deduplicate list"

def test_sort_by_stack():
    obj   = [5, 2, 1, 3, 4]
    stack = [1, 2, 9, 3, 19, 4, 99, 5]
    assert sort_by_stack(obj, stack) == [1, 2, 3, 4, 5], "Failed to sort by stack"

@patch('sd.utils.ImageGrab.grab')
@patch('sd.utils.GdkPixbuf.Pixbuf.new_from_file')
def test_get_screenshot(mock_new_from_file, mock_grab, tmp_path):
    # Mock the screenshot saving
    mock_pixbuf = MagicMock()
    mock_new_from_file.return_value = mock_pixbuf

    # Mock `grab` to simulate capturing a screenshot without actually doing it
    mock_grab.return_value = MagicMock()
    
    # Simulate the window's position
    window = MagicMock()
    window.get_position.return_value = (100, 100)

    # Use the tmp_path fixture for a temporary file path
    x0, y0, x1, y1 = 10, 10, 110, 110  # Arbitrary coordinates

    pixbuf, temp_file_name = get_screenshot(window, x0, y0, x1, y1)

    # Verify the grab was called with the correct coordinates
    mock_grab.assert_called_once_with(bbox=(100 + x0, 100 + y0, 100 + x1, 100 + y1))

    # Verify the returned pixbuf is our mock_pixbuf
    assert pixbuf == mock_pixbuf, "Pixbuf does not match the mocked return value"

@pytest.mark.parametrize("p1,p2,p3,p4,expected", [
    # Intersecting in the middle
    ((0, 0), (10, 10), (0, 10), (10, 0), (True, (5, 5))),
    # Parallel, no intersection
    ((0, 0), (10, 0), (0, 1), (10, 1), (False, None)),
    # Same line, disjoint
    ((0, 0), (5, 5), (6, 6), (10, 10), (False, None)),
    # Sharing an endpoint
    ((0, 0), (5, 5), (5, 5), (10, 5), (True, (5, 5))),
    # Completely overlapping (special case, might need handling depending on implementation)
    ((0, 0), (10, 10), (0, 0), (10, 10), (False, None)),  # or whatever expected based on your logic
])
def test_segment_intersection(p1, p2, p3, p4, expected):
    assert segment_intersection(p1, p2, p3, p4) == expected

def test_pp():
    assert pp((1.9, 2.1)) == [1, 2]


