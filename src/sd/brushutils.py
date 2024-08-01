"""Utilities for calculation of brush strokes."""

import logging                                   # <remove>
import gi                                        # <remove>
gi.require_version("GLib", "2.0")                # <remove> pylint: disable=wrong-import-position
from gi.repository import GLib                   # <remove>
import cairo                                     # <remove>
import numpy as np                               # <remove>
from .utils import normal_vec                    # <remove>
from .utils import calculate_angle2              # <remove>
from .utils import distance                      # <remove>
from .utils import calc_intersect                # <remove>

log = logging.getLogger(__name__)                # <remove>
NP_VEC = { 7: np.linspace(0, 1, 7) }
TIME_OLD = 0
TIME_NEW = 0

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

    ret = np.column_stack((nx, ny))
    return ret

def calc_normal_segments_scaled(normal_segments, widths):
    """multiply the normal segments by weights"""

    widths = np.array(widths) / 2
    nw0 = normal_segments * widths[:-1, np.newaxis]
    nw1 = normal_segments * widths[1:, np.newaxis]

    return nw0, nw1

def determine_side_math(p1, p2, p3):
    """Determine the side of a point p3 relative to a line segment p1->p2."""
    det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 'left' if det > 0 else 'right'

def calc_arc_coords2(p1, p2, c, n = 20):
    """Calculate the coordinates of an arc between points p1 and p2.
       The arc is a fragment of a circle with centre in c."""
    # pylint: disable=too-many-locals
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    if n not in NP_VEC:
        NP_VEC[n] = np.linspace(0, 1, n)

    # calculate the radius of the circle
    radius = np.sqrt((x2 - xc)**2 + (y2 - yc)**2)
    side = determine_side_math(p1, p2, c)

    # calculate the angle between the line p1->c and p1->p2
    a1 = np.arctan2(y1 - c[1], x1 - c[0])
    a2 = np.arctan2(y2 - c[1], x2 - c[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * np.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * np.pi

    #angles = np.linspace(a1, a2, n)
    angles = a1 + (a2 - a1) * NP_VEC[n]

    # Calculate the arc points
    x_coords = xc + radius * np.cos(angles)
    y_coords = yc + radius * np.sin(angles)

    # Combine x and y coordinates
    #coords = np.column_stack((x_coords, y_coords))
    coords = list(zip(x_coords, y_coords))
    return coords

def calc_segments(coords, nw0, nw1):
    """calculate starting and ending points of the segments"""

    l_seg_s = coords[:-1] + nw0
    r_seg_s = coords[:-1] - nw0
    l_seg_e = coords[1:] + nw1
    r_seg_e = coords[1:] - nw1

    return [l_seg_s, l_seg_e], [r_seg_s, r_seg_e]

def calc_intersections(lseg_s, lseg_e):
    """
    Calculate intersection points between consecutive line segments.
    Each segment consists of two points, start and end.
    """
    # pylint: disable=too-many-locals

    x1, y1 = lseg_s[:-1].T  # Start points of segments
    x2, y2 = lseg_e[:-1].T  # End points of segments
    x3, y3 = lseg_s[1:].T   # Start points of the next segments
    x4, y4 = lseg_e[1:].T   # End points of the next segments

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Avoid division by zero
    denom_nonzero = denom != 0

    # Calculate t and u only where denom is non-zero
    t = np.zeros_like(denom)
    u = np.zeros_like(denom)

    t[denom_nonzero] = ((x1[denom_nonzero] - x3[denom_nonzero]) *
                        (y3[denom_nonzero] - y4[denom_nonzero]) -
                        (y1[denom_nonzero] - y3[denom_nonzero]) *
                        (x3[denom_nonzero] - x4[denom_nonzero])) / denom[denom_nonzero]

    u[denom_nonzero] = ((x1[denom_nonzero] - x3[denom_nonzero]) *
                        (y1[denom_nonzero] - y2[denom_nonzero]) -
                        (y1[denom_nonzero] - y3[denom_nonzero]) *
                        (x1[denom_nonzero] - x2[denom_nonzero])) / denom[denom_nonzero]
    # Conditions for intersection
    intersect_cond = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)

    # Calculate intersection points
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)

    # Mask out non-intersections
    intersect_x[~intersect_cond] = np.nan
    intersect_y[~intersect_cond] = np.nan

    # Stack results
    intersections = np.stack((intersect_x, intersect_y), axis=-1)

    return intersections

def construct_outline(segments, intersections):
    """Construct the left or right outline from segments and intersections."""
    starts, ends = segments
    is_intersect = ~np.isnan(intersections[:, 0]) & ~np.isnan(intersections[:, 1])

    n = len(intersections)
    is_s = np.sum(~is_intersect) # is segment

    # joints count double
    l_out = n + is_s

    result = np.zeros((l_out + 2, 2), dtype=segments[0].dtype)

    seg_int = np.where(~is_intersect)[0]
    selector_seg = seg_int + np.arange(is_s)

    result[selector_seg + 1] = ends[:-1][~is_intersect]
    result[selector_seg + 2] = starts[1:][~is_intersect]

    selector = np.zeros(l_out + 2, dtype=bool)
    selector[:] = True
    selector[[0, -1]] = False

    selector[selector_seg + 1] = False
    selector[selector_seg + 2] = False
    result[selector] = intersections[is_intersect]
    result[0] = starts[0]
    result[-1] = ends[-1]
    return result

def calc_arc_coords_np(p1, p2, c, n = 20):
    """Calculate the coordinates of an arc between points p1 and p2.
       The arc is a fragment of a circle with centre in c."""
    # pylint: disable=too-many-locals
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    if n not in NP_VEC:
        NP_VEC[n] = np.linspace(0, 1, n)

    # calculate the radius of the circle
    radius = np.sqrt((x2 - xc)**2 + (y2 - yc)**2)
    side = determine_side_math(p1, p2, c)

    # calculate the angle between the line p1->c and p1->p2
    a1 = np.arctan2(y1 - c[1], x1 - c[0])
    a2 = np.arctan2(y2 - c[1], x2 - c[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * np.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * np.pi

    angles = a1 + (a2 - a1) * NP_VEC[n]

    # Calculate the arc points
    x_coords = xc + radius * np.cos(angles)
    y_coords = yc + radius * np.sin(angles)

    return x_coords, y_coords

def construct_outline_round(segments, intersections, coords, n = 3,
                            ret_segments = False):
    """Construct the left or right outline from segments and intersections."""
    # pylint: disable=too-many-locals

    starts, ends = segments
    n_segments = len(starts)
    #print("starts", starts)
    #print("ends", ends)
    is_intersect = ~np.isnan(intersections[:, 0]) & ~np.isnan(intersections[:, 1])

    #print(is_intersect)
    n_int = len(intersections)

    # number of rounded joints
    is_rj = np.sum(~is_intersect)

    # round joints count n + 1 times
    l_out = n_int + is_rj * (n - 1)
    #print("n_segments = ", n_segments, "n =", n, "n_int = ", n_int, "is_rj = ", is_rj, "l_out = ", l_out)

    result = np.zeros((l_out + 2, 2), dtype=segments[0].dtype)
    
    # first and last point
    result[0] = starts[0]
    result[-1] = ends[-1]

    seg_int = np.where(~is_intersect)[0]
    #print("seg_int", seg_int)
    #print("intersections at:", np.where(is_intersect)[0])

    # selector_seg selects the segment joints that need to be an arc
    # it contains the positions in the resulting array of the segment joints
    selector_seg = seg_int + np.arange(is_rj) * (n - 1)
    #print("selector_seg", selector_seg)

    # only these which are not intersections
    starts = starts[1:][~is_intersect]
    ends = ends[:-1][~is_intersect]

    point_selector = np.arange(len(coords))[1:-1]
    #print("point_selector", point_selector)

    # fill in the arcs
    # i is the index of the segment joint
    # ind is the position of the joint in the result array
    # idx is the index of the point in the coords array
    for i, ind in enumerate(selector_seg):
        idx = point_selector[~is_intersect][i]
        ac_x, ac_y = calc_arc_coords_np(ends[i], starts[i], coords[idx], n)
        result[ind + 1:ind + n + 1, 0] = ac_x
        result[ind + 1:ind + n + 1, 1] = ac_y
        #print("i =", i, "idx=", idx, "ind=", ind, "p=", coords[idx], "arc_coords =", arc_coords)
    index_column = np.arange(result.shape[0]).reshape(-1, 1)
    print("after adding arcs\n", np.hstack((index_column, result)).astype(int))

    # selector indicates where in the array are the intersections
    selector = np.zeros(l_out + 2, dtype=bool)
    selector[:] = True
    selector[[0, -1]] = False

    for i in range(n):
        selector[selector_seg + i + 1] = False

    #result[selector_seg + 1] = ends
    #print("after adding ends\n", result[:10])
    #result[selector_seg + n] = starts
    #print("after adding starts\n", result[:10])
    result[selector] = intersections[is_intersect]
    #print("after adding intersections\n", result[:10])
    if not ret_segments:
        return result, None

    seg_positions = np.zeros((n_segments, 2), dtype=int)
    seg_positions[np.where(is_intersect)[0], 0] = np.where(selector)[0] - 1

    # number of points corresponding to the segment including start and end
    seg_positions[np.where(is_intersect)[0], 1] = 2

    seg_positions[np.where(~is_intersect)[0], 0] = selector_seg
    # number of points corresponding to the segment including start and arc
    seg_positions[np.where(~is_intersect)[0], 1] = n + 1

    seg_positions[-1, 0] = l_out
    seg_positions[-1, 1] = 2
    print("seg_positions\n", seg_positions)

    return result, seg_positions

def round_tip(rseg, lseg, coords, n = 3):
    """Round tip of the brush."""

    p_s = coords[0]
    r_s = rseg[0][0]
    l_s = lseg[0][0]
    ac_x_s, ac_y_s = calc_arc_coords_np(r_s, l_s, p_s, n)
    arc_coords_s = np.column_stack((ac_x_s, ac_y_s))

    p_e = coords[-1]
    r_e = rseg[1][-1]
    l_e = lseg[1][-1]
    ac_x_e, ac_y_e = calc_arc_coords_np(l_e, r_e, p_e, n)
    arc_coords_e = np.column_stack((ac_x_e, ac_y_e))

    return arc_coords_s, arc_coords_e

def calc_normal_outline(coords, widths, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)

    if n < 2:
        return [], []

    if n == 2:
        return calc_normal_outline_short(coords, widths, rounded)

    t1 = GLib.get_monotonic_time()
    coords_np = np.array(coords)

    # normal vectors
    n_segm = calc_normal_segments(coords_np)

    # normal vectors scaled by the widths
    nw0, nw1 = calc_normal_segments_scaled(n_segm, widths)

    # calculate the outline segments
    lseg, rseg = calc_segments(coords_np, nw0, nw1)

    # figure whether and if yes, where the segments intersect
    l_intersect = calc_intersections(*lseg)
    r_intersect = calc_intersections(*rseg)

    if rounded:
        outline_l, _ = construct_outline_round(lseg, l_intersect,
                                            coords, n = 3)
        outline_r, _ = construct_outline_round(rseg, r_intersect,
                                            coords, n = 3)
        tip_s, tip_e = round_tip(rseg, lseg, coords, n = 10)
        outline_l = np.concatenate((tip_s, outline_l, tip_e))
    else:
        outline_l = construct_outline(lseg, l_intersect)
        outline_r = construct_outline(rseg, r_intersect)

    t2 = GLib.get_monotonic_time()

    log.debug("outline lengths: %d, %d", len(outline_l), len(outline_r))
    log.debug("coords length: %d", len(coords))
    global TIME_NEW
    TIME_NEW += t2 - t1
    log.debug("time, new: %s", TIME_NEW)
    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def __consecutive_runs(values):
    """Find the lengths of consecutive identical numbers."""

    # Convert the list to a NumPy array
    array = np.array(values)

    # Find the indices where the value changes
    change_indices = np.where(np.diff(array) != 0)[0] + 1

    # Add the start and end indices
    change_indices = np.concatenate(([0], change_indices, [len(array)]))

    # Calculate the lengths of consecutive identical numbers
    lengths = np.diff(change_indices)

    # Get the unique values corresponding to each segment
    unique_values = array[change_indices[:-1]]

    return lengths, unique_values

def __consecutive_runs_2(values):
    """
    Find the lengths of consecutive identical pairs of numbers.
    """

    a0 = values[:,0]
    a1 = values[:,1]

    a0_diff = np.diff(a0) != 0
    a1_diff = np.diff(a1) != 0

    print("values:", values)
    print("a0_diff:", a0_diff)
    print("a1_diff:", a1_diff)

    # Find the indices where the value changes
    change_indices = np.where(a0_diff | a1_diff)[0] + 1

    # Add the start and end indices
    change_indices = np.concatenate(([0], change_indices, [len(values)]))

    # Calculate the lengths of consecutive identical numbers
    lengths = np.diff(change_indices)

    return np.column_stack((change_indices[:-1], lengths))

def __pressure_to_bins(pressure, n_bins = 5):
    """Convert pressure values to bins."""

    print("pressure:", pressure)
    pressure = np.array(pressure)
    pressure = (pressure[:-1] + pressure[1:]) / 2
    print("pressure:", pressure)

    bin_edges = np.linspace(0, 1, n_bins + 1, endpoint=False)[1:]
    print("bin_edges:", bin_edges)
    bin_edges = np.append(bin_edges, 1.0)
    print("bin_edges:", bin_edges)

    # Find the bin index for each number
    pressure = np.digitize(pressure, bins=bin_edges, right=True)
    print("pressure:", pressure)

    return pressure

def __make_segments(coords, ledge, redge, l_intersect, r_intersect,
                    join_lengths):
    """
    Make segments from the left and right outlines.

    Returns two arrays. First array contains the coordinates of all
    segments in a continuous manner (so that scaling or moving would become
    easy). This is the same as the outline, but the coordinates are
    rearranged to form distinct segments.

    The second array contains information about the segments: starting
    coordinate, length, coordinate at which the segment ends are, and the
    transparency value.
    """

    l_is_intersect = ~np.isnan(l_intersect[:, 0]) & ~np.isnan(l_intersect[:, 1])
    r_is_intersect = ~np.isnan(r_intersect[:, 0]) & ~np.isnan(r_intersect[:, 1])
    b_is_intersect = l_is_intersect & r_is_intersect

    ## we will return the outline as a single numpy object
    # this will have n - 1 segments
    # each segment will have 4 coordinates with no intersections,
    # 2 + arc coordinates with 1 intersection and 
    # 2 * arc coordinates with 2 intersections

    #join_lengths, join_vals = __consecutive_runs(pressure) 

    # create np array with n_coord rows and three columns

    outline_l, l_seg_pos = construct_outline_round(ledge, l_intersect,
                                        coords, n = 3, ret_segments = True)
    outline_r, r_seg_pos = construct_outline_round(redge, r_intersect,
                                        coords, n = 3, ret_segments = True)

    print("outline_l: \n", outline_l[:], "\nl_seg_pos: \n", l_seg_pos)
    print("outline_r: \n", outline_r[:], "\nr_seg_pos: \n", r_seg_pos)

    print("join_lengths: ", join_lengths, "sum:", np.sum(join_lengths))
    print("l_seg_pos length: ", len(l_seg_pos), "r_seg_pos length: ", len(r_seg_pos))

    tot_seg_length = len(outline_l) + len(outline_r) + len(join_lengths) * 2 - 2
    print("outline_l length: ", len(outline_l), "outline_r length: ", len(outline_r))
    print("tot_seg_length = ", tot_seg_length)

    # first column: the coordinate at which the segment starts in the
    # resulting array
    # second column: number of coordinates in the segment
    # third column: the coordinate at which the actual segment ends are
    # found (note: the outline continues beyond this point, since there is
    # the right outline after the left. the second column shows where the
    # left outline ends and the right outline starts)
    # fourth column: the binned pressure value for this segment
    # XXX: each segment should have a starting and ending pressure value
    # or actually how are the pressure values used?
    segments = np.zeros((tot_seg_length, 2))

    seg_info = np.zeros((len(join_lengths), 5), dtype = int)

    left  = None
    right = None

    pos = 0
    seg = 0
    n_r = 0
    n_l = 0

    for jseg in range(len(join_lengths)):
        left  = [ ]
        right = [ ]
        j_len = join_lengths[jseg]
        n_coords_l = np.sum(l_seg_pos[seg:seg + j_len, 1]) - j_len + 1
        n_coords_r = np.sum(r_seg_pos[seg:seg + j_len, 1]) - j_len + 1
        n_coords = n_coords_l + n_coords_r
        n_l += n_coords_l
        n_r += n_coords_r
        seg_info[jseg, 0] = pos
        seg_info[jseg, 1] = n_coords
        seg_info[jseg, 3] = seg
        seg_info[jseg, 4] = seg + j_len - 1

        print("building joined segment jseg = ", jseg, 
              "join_lengths[jseg] = ", j_len,
              "n_coords_l = ", n_coords_l, "n_coords_r = ", n_coords_r,
              "n_coords = ", n_coords, "n_l = ", n_l, "n_r = ", n_r)

        # first the left segment: from start to end
        pos0 = pos
        for j in range(j_len):
            if j == 0:
                coord_start = l_seg_pos[seg + j, 0]
                coord_n     = l_seg_pos[seg + j, 1]
            else:
                coord_start = l_seg_pos[seg + j, 0] + 1
                coord_n     = l_seg_pos[seg + j, 1] - 1

            segments[pos0:pos0 + coord_n, 0:2] = outline_l[coord_start:coord_start + coord_n, 0:2]
            print("j = ", j, "coord_start = ", coord_start, "coord_n = ", coord_n, "seg + j = ", seg + j, "pos0 = ", pos0)
            print("outline_l:\n", outline_l[coord_start:coord_start + coord_n, 0:2])
            #segments[pos0:pos0 + coord_n,2] = jseg
            #segments[pos0:pos0 + coord_n,3] = join_vals[jseg]

            #seg += 1
            pos0 += coord_n

        seg_info[jseg, 2] = pos0 - 1
        print("value at pos0: ", segments[pos0 - 1])
        #print("segments:\n", segments[:20].astype(int))
        # then the right segment: from end to start
        pos0 = 0
        for j in range(j_len):
            if j == 0:
                coord_start = r_seg_pos[seg + j, 0]
                coord_n     = r_seg_pos[seg + j, 1]
            else:
                coord_start = r_seg_pos[seg + j, 0] + 1
                coord_n     = r_seg_pos[seg + j, 1] - 1

            s_to   = pos + n_coords - pos0
            s_from = s_to - coord_n
            print("j = ", j, "coord_start = ", coord_start, "coord_n = ", coord_n, "seg + j = ", seg + j, "pos = ", pos)
            print("outline_r:\n", outline_r[coord_start:coord_start + coord_n, 0:2])
            print("outline_r:\n", outline_r[coord_start:coord_start + coord_n, 0:2][::-1])
            segments[s_from:s_to, 0:2] = outline_r[coord_start:coord_start + coord_n, 0:2][::-1]
            #segments[s_from:s_to,2] = jseg
            #segments[s_from:s_to,3] = join_vals[jseg]

            pos0 += coord_n

        pos += n_coords
        seg += j_len

    print("n_l = ", n_l, "n_r = ", n_r, "pos = ", pos)
    print("outline_l length: ", len(outline_l), "outline_r length: ", len(outline_r))
    print("segments:\n", segments.astype(int))
    print("len segments: ", len(segments))
    print("seg_info:\n", seg_info.astype(int))
    return segments, seg_info, None

def __calc_midpoints(seg_info, lseg, rseg, pressure):
    """
    Calculate the segment midpoints used to generate the pencil gradient
    """

    midpoints = np.zeros((len(seg_info), 6), dtype = lseg[0].dtype)
    midpoints[:,4:6] = pressure
    sel = seg_info[:,3]
    midpoints[:,0:2] = (lseg[0][sel] + rseg[0][sel])/2
    midpoints[:,2:4] = (lseg[1][sel] + rseg[1][sel])/2
    print("midpoints:\n", midpoints)
    
    return midpoints

def calc_pencil_segments(coords, widths, pressure):
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
        return [], []

    # normal vectors
    print("len coords: ", len(coords), "len pressure: ", len(pressure))
    coords_np = np.array(coords)
    n_segm = calc_normal_segments(coords_np)

    # normal vectors scaled by the widths
    nw0, nw1 = calc_normal_segments_scaled(n_segm, widths)

    # calculate the outline segments
    lseg, rseg = calc_segments(coords_np, nw0, nw1)

    # figure whether and if yes, where the segments intersect
    l_intersect = calc_intersections(*lseg)
    r_intersect = calc_intersections(*rseg)

    pressure = __pressure_to_bins(pressure, n_bins = 5)
    pressure = pressure / 5 * 0.8 + 0.2
    #print("len pressure now: ", len(pressure))

    pp = np.column_stack((pressure[:-1], pressure[1:]))
    #print("pp: \n", pp)
    runs = __consecutive_runs_2(pp)
    pp = pp[runs[:,0]]
    #print("runs:\n", runs)

    segments, seg_info, pressure = __make_segments(coords, lseg, rseg, 
                               l_intersect, r_intersect,
                                                   runs[:,1])
    #print("len coords:", len(coords), "len lseg:", len(lseg[0]))

    midpoints = __calc_midpoints(seg_info, lseg, rseg, pp)

    return segments, seg_info, midpoints


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

    npt = 7

    for i in range(n - 2):
        p2 = coords[i + 2]
        width  = line_width / 3#* pressure[i] / 2

        l_seg2, r_seg2 = calc_segments_2(p1, p2, width)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_l = calc_intersect(l_seg1, l_seg2)

        if intersect_l is None:
            arc_coords = calc_arc_coords2(l_seg1[1], l_seg2[0], p1, npt)
            outline_l.extend(arc_coords)
        else:
            outline_l.extend([intersect_l] * npt)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_r = calc_intersect(r_seg1, r_seg2)

        if intersect_r is None:
            arc_coords = calc_arc_coords2(r_seg1[1], r_seg2[0], p1, npt)
            outline_r.extend(arc_coords)
        else:
            outline_r.extend([intersect_r] * npt)

        pressure_ret.extend([ pressure[i] ] * npt)

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
