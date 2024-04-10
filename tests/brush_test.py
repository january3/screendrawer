import math
import pytest
from pytest import approx
from sd.brush import find_intervals, calculate_slant_outlines
from sd.brush import normal_vec_scaled, calc_segments
from sd.brush import point_mean, calc_normal_outline
from sd.brush import BrushFactory, Brush
from sd.utils import normal_vec


def test_find_intervals():
    """Test the find_intervals function"""

    values = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
    nbins  = 3

    bins, bin_size = find_intervals(values, nbins)

    assert bins == [ [0, 1, 2], [3, 4, 5], [6, 7, 8] ], "Failed to find intervals"
    assert bin_size == 1/3, "Failed to find the correct bin size"

def test_calculate_slant_outlines():
    """Test the calculate_slant_outline function"""

    coords = [ (0, 0), (1, 0), (1, 1), (2, 2) ]
    dx0, dy0, dx1, dy1 = -0.5, -0.5, 0.5, 0.5

    o_l, o_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
    assert o_l == [(0.5, 0.5), (1.5, 0.5), (0.5, 0.5), (1.5, 1.5)] 
    assert o_r == [(-0.5, -0.5), (0.5, -0.5), (1.5, 1.5), (2.5, 2.5)]

def test_normal_vec_scaled():
    """Test the normal_vec_scaled function"""

    width = 2
    p0 = (0, 0)
    p1 = (10, 10)

    nx, ny = normal_vec_scaled(p0, p1, width)
    assert ny - math.sqrt(2) < 0.00001
    assert nx - math.sqrt(2) < 0.00001

def test_calc_segments():
    """Test calculating normal segments"""

    p0 = (0, 0)
    p1 = (100, 100)

    width = 100 / math.sqrt(2)

    ls, rs, le, re = calc_segments(p0, p1, width)
    assert ls == approx((-50, 50))
    assert rs == approx((50, -50))
    assert le == approx((50, 150))
    assert re == approx((150, 50))

def test_point_mean():
    """Test the point mean function"""

    p0 = (0, 0)
    p1 = (10, 10)

    p = point_mean(p0, p1)
    assert p == (5, 5)

def test_calc_normal_outline():
    """Test calculating the normal outline"""

    coords = [ (0, 0), (1, 0.5), (2, .5), (2, 1) ]
    width = 1
    pressure = [ 0.5, 0.5, 0.5, 0.5 ]

    o_l, o_r = calc_normal_outline(coords, pressure, width, False)
    assert len(o_l) == 6
    assert len(o_r) == 6
    assert len(o_l[0]) == 2
    assert len(o_r[0]) == 2


@pytest.mark.parametrize("brush", ["rounded", "slanted", "marker", "pencil"])
def test_brush_factory(brush):
    """Test the brush factory"""

    b = BrushFactory.create_brush(brush)
    assert b is not None, "Failed to create brush"
    assert isinstance(b, Brush), "Failed to create the correct brush"
    assert b.brush_type() == brush, "Failed to create the correct brush type"

    outline = [ (0, 0), (1, 1), (2, 2), (3, 3) ]
    b.outline(outline)
    assert outline == b.outline(), "Failed to set the outline"
    bbox = b.bbox()

    assert bbox == (0, 0, 3, 3), "Failed to calculate the bounding box"

    b.move(1, 1)
    assert b.outline() == [ (1, 1), (2, 2), (3, 3), (4, 4) ], "Failed to move the brush"

    b.scale((0, 0, 1, 1), (0, 0, 2, 2))
    assert b.outline() == [ (2, 2), (4, 4), (6, 6), (8, 8) ], "Failed to scale the brush"

    d = b.to_dict()
    assert d is not None, "Failed to convert brush to dict"
    assert isinstance(d, dict), "Failed to convert brush to dict"
    assert d["brush_type"] == brush, "Failed to convert brush to dict"

    coords = [ (0, 0), (1, 1), (2, 2), (3, 3), (4, 4) ]
    b.calculate(0.75, coords)
    b.calculate(0.75, coords, (1, 1, 1, 1, .2))

    # the following should fail
    with pytest.raises(Exception):
        b.calculate(0.75, coords, (1, 1, 1, 1))
