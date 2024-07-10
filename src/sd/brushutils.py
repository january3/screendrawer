"""Utilities for calculation of brush strokes."""

import logging                                                   # <remove>
import cairo                                                     # <remove>
from .utils import calc_arc_coords2, normal_vec # <remove>
from .utils import calculate_angle2                              # <remove>
from .utils import distance                                      # <remove>
from .utils import calc_intersect                                # <remove>

log = logging.getLogger(__name__)                                # <remove>

def get_current_color_and_alpha(ctx):
    """Get the current color and alpha from the Cairo context."""
    pattern = ctx.get_source()
    if isinstance(pattern, cairo.SolidPattern):
        r, g, b, a = pattern.get_rgba()
        return r, g, b, a
    # For non-solid patterns, return a default or indicate it's not solid
    return None, None, None, None  # Or raise an exception or another appropriate response

def bin_values(values, n_bins):
    """Bin values into n_bin bins between 0 and 1"""

    bin_size = 1 / n_bins
    ret = [ ]

    for value in values:
        ret.append(int(value / bin_size) * bin_size)

    return ret

def min_pr(val):
    """Scale the pressure such that it does not go below .2"""
    return .2 + .8 * val

def find_intervals(values, n_bins):
    """
    Divide the range [0, 1] into n_bins and return a list of bins with 
    indices of values falling into each bin.
    """

    # Calculate the bin size
    bin_size = 1 / n_bins

    # Initialize bins as a list of empty lists
    bins = [[] for _ in range(n_bins)]

    # Assign each value to a bin
    for i, value in enumerate(values):
        if value == 1:  # Handle the edge case where value is exactly 1
            bins[-1].append(i)
        else:
            bin_index = int(value / bin_size)
            bins[bin_index].append(i)

    return bins, bin_size

def smooth_pressure(pressure, window_size = 3):
    """Smooth the pressure values."""
    n = len(pressure)
    smoothed = [ ]
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size)
        smoothed.append(sum(pressure[start:end]) / (end - start))
    return smoothed

def calculate_slant_outlines(coords, dx0, dy0, dx1, dy1):
    """Calculate the left and right outlines."""
    outline_l, outline_r = [ ], [ ]
    slant_vec   = (dx0 - dx1, dy0 - dy1)
    p_prev = coords[0]
    x, y = coords[0]
    outline_l.append((x + dx0, y + dy0))
    outline_r.append((x + dx1, y + dy1))
    prev_cs_angle = None

    for p in coords[1:]:
        x, y = p
        coord_slant_angle = calculate_angle2((x - p_prev[0], y - p_prev[1]), slant_vec)

        # avoid crossing of outlines
        if prev_cs_angle is not None:
            if prev_cs_angle * coord_slant_angle < 0:
                outline_l, outline_r = outline_r, outline_l

        prev_cs_angle = coord_slant_angle

        outline_l.append((x + dx0, y + dy0))
        outline_r.append((x + dx1, y + dy1))
        p_prev = p

    return outline_l, outline_r

def normal_vec_scaled(p0, p1, width):
    """Calculate the normal vector of a line segment."""
    nx, ny = normal_vec(p0, p1)
    nx, ny = nx * width, ny * width
    return nx, ny

def calc_segments_2(p0, p1, width):
    """Calculate the segments of an outline."""
    nx, ny = normal_vec_scaled(p0, p1, width)

    l_seg_s = (p0[0] + nx, p0[1] + ny)
    r_seg_s = (p0[0] - nx, p0[1] - ny)
    l_seg_e = (p1[0] + nx, p1[1] + ny)
    r_seg_e = (p1[0] - nx, p1[1] - ny)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)

def calc_segments_3(p0, p1, w1, w2):
    """Calculate the segments of an outline."""
    nx1, ny1 = normal_vec_scaled(p0, p1, w1)
    nx2, ny2 = normal_vec_scaled(p0, p1, w2)

    l_seg_s = (p0[0] + nx1, p0[1] + ny1)
    r_seg_s = (p0[0] - nx1, p0[1] - ny1)
    l_seg_e = (p1[0] + nx2, p1[1] + ny2)
    r_seg_e = (p1[0] - nx2, p1[1] - ny2)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)


def calc_outline_short_generic(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""

    outline_l = []
    outline_r = []
    line_width = line_width or 1

    p0, p1 = coords[0], coords[1]
    width  = line_width * pressure[0] / 2

    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r

def calc_normal_outline_short(coords, widths, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""

    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = widths[0] / 2

    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r


def calc_normal_segments(coords):
    """Calculate the normal segments of a path using numpy."""

    coords = np.array(coords)
    widths = np.array(widths)

    # Calculate differences between consecutive points
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])

    # Calculate lengths of the segments
    lengths = np.sqrt(dx**2 + dy**2)

    # Normalize the vectors
    dx_normalized = dx / lengths
    dy_normalized = dy / lengths

    # Calculate normal vectors
    nx = -dy_normalized
    ny = dx_normalized

    # Scale normal vectors by widths
    #nx_scaled = nx * widths[:-1]
    #ny_scaled = ny * widths[:-1]

    return np.column_stack((nx_scaled, ny_scaled))


def calc_normal_outline(coords, widths, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)
    #print("CALCULATING NORMAL OUTLINE")

    if n < 2:
        return [], []

    if n == 2:
        return calc_normal_outline_short(coords, widths, rounded)

    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    w1 = widths[0] / 2
    w2 = widths[1] / 2
    l_seg1, r_seg1 = calc_segments_3(p0, p1, w1, w2)

    ## append the points for the first coord
    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)
    outline_l.append(l_seg1[0])
    outline_r.append(r_seg1[0])

    for i in range(n - 2):
        p2 = coords[i + 2]
        w1 = widths[i + 1] / 2
        w2 = widths[i + 2] / 2

        l_seg2, r_seg2 = calc_segments_3(p1, p2, w1, w2)

        intersect_l = calc_intersect(l_seg1, l_seg2)
        intersect_r = calc_intersect(r_seg1, r_seg2)

        if intersect_l is not None:
            outline_l.append(intersect_l)
        else:
            outline_l.append(l_seg1[1])
            if rounded:
                arc_coords = calc_arc_coords2(l_seg1[1], l_seg2[0], p1, 10)
                outline_l.extend(arc_coords)
            outline_l.append(l_seg2[0])

        if intersect_r is not None:
            outline_r.append(intersect_r)
        else:
            outline_r.append(r_seg1[1])
            if rounded:
                arc_coords = calc_arc_coords2(r_seg1[1], r_seg2[0], p1, 10)
                outline_r.extend(arc_coords)
            outline_r.append(r_seg2[0])

        p0, p1 = p1, p2
        l_seg1, r_seg1 = l_seg2, r_seg2

    outline_l.append(l_seg2[1])
    outline_r.append(r_seg2[1])

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p0, 10)
        outline_l.extend(arc_coords)

    log.debug("outline lengths: %d, %d", len(outline_l), len(outline_r))
    log.debug("coords length: %d", len(coords))
    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calc_pencil_outline(coords, pressure, line_width):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush v3.

    The main difference from normal outlines is that we are creating two
    outlines and a pressure vector, all with exactly the same number of
    points. This allows to create segments with different transparency
    values.
    """

    n = len(coords)

    if n < 2:
        return [], [], []

    if n == 2:
        outline_l, outline_r = calc_outline_short_generic(coords, pressure, line_width, True)
        return outline_l, outline_r, pressure

    pressure_ret = [ ]
    line_width = line_width or 1
    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = line_width / 5 #* pressure[0] / 2
    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    ## append the points for the first coord
    arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
    outline_r.extend(arc_coords[:5][::-1])
    outline_l.extend(arc_coords[5:])
    pressure_ret = pressure_ret + [pressure[0]] * 5

    outline_l.append(l_seg1[0])
    outline_r.append(r_seg1[0])
    pressure_ret.append(pressure[0])

    np = 7

    for i in range(n - 2):
        p2 = coords[i + 2]
        width  = line_width / 3#* pressure[i] / 2

        l_seg2, r_seg2 = calc_segments_2(p1, p2, width)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_l = calc_intersect(l_seg1, l_seg2)

        if intersect_l is None:
            arc_coords = calc_arc_coords2(l_seg1[1], l_seg2[0], p1, np)
            outline_l.extend(arc_coords)
        else:
            outline_l.extend([intersect_l] * np)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_r = calc_intersect(r_seg1, r_seg2)

        if intersect_r is None:
            arc_coords = calc_arc_coords2(r_seg1[1], r_seg2[0], p1, np)
            outline_r.extend(arc_coords)
        else:
            outline_r.extend([intersect_r] * np)

        pressure_ret.extend([ pressure[i] ] * np)

        l_seg1, r_seg1 = l_seg2, r_seg2
        p0, p1 = p1, p2

    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[1])
    pressure_ret.append(pressure[-1])

    arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p0, 10)
    outline_r.extend(arc_coords[:5])
    outline_l.extend(arc_coords[5:][::-1])
    pressure_ret = pressure_ret + [pressure[-1]] * 5

    log.debug("outline lengths: %d, %d pres=%d", len(outline_l), len(outline_r), len(pressure))
    return outline_l, outline_r, pressure_ret

def calc_normal_outline_tapered(coords, pressure, line_width, taper_pos, taper_length):
    """Calculate the normal outline of a path for tapered brush."""
    n = len(coords)

    if n < 2:
        return [], []

    if n == 2:
        return calc_outline_short_generic(coords, pressure, line_width, False)

    line_width = line_width or 1

    outline_l = []
    outline_r = []

    taper_l_cur = 0
    p0, p1 = coords[0], coords[1]

    for i in range(n - 2):
        p2 = coords[i + 2]

        if i >= taper_pos:
            taper_l_cur += distance(p0, p1)
            taper_l_cur = min(taper_l_cur, taper_length)

            w0  = line_width * pressure[i] / 2 * (1 - taper_l_cur / taper_length)
            w1  = line_width * pressure[i + 1] / 2 * (1 - taper_l_cur / taper_length)
            w2  = line_width * pressure[i + 2] / 2 * (1 - taper_l_cur / taper_length)
        else:
            w0  = line_width * pressure[i] / 2
            w1  = line_width * pressure[i + 1] / 2
            w2  = line_width * pressure[i + 2] / 2

        l_seg1, r_seg1 = calc_segments_3(p0, p1, w0, w1)
        l_seg2, r_seg2 = calc_segments_3(p1, p2, w1, w2)

        if i == 0:
        ## append the points for the first coord
            arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
            outline_r.extend(arc_coords)
            outline_l.append(l_seg1[0])
            outline_r.append(r_seg1[0])

        outline_l.append(l_seg1[1])
        outline_l.append(l_seg2[0])
        outline_r.append(r_seg1[1])
        outline_r.append(r_seg2[0])

        if i == n - 3:
            outline_l.append(point_mean(l_seg2[1], r_seg2[1]))
        p0, p1 = p1, p2

    return outline_l, outline_r
