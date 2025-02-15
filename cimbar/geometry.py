from collections import namedtuple


Midpoints = namedtuple('Midpoints', 'top right bottom left')


def line_intersection(lineA, lineB):
    # takes a tuple of tuples
    def compute(p, q):
        xdiff = q[0] - p[0]
        ydiff = p[1] - q[1]
        determinant = q[0] * p[1] - p[0] * q[1]
        return xdiff, ydiff, determinant

    ax, ay, adet = compute(*lineA)
    bx, by, bdet = compute(*lineB)

    D = ay * bx - ax * by
    if abs(D) < 1e-8:
        return None

    Dx = adet * bx - ax * bdet
    Dy = ay * bdet - adet * by
    x = Dx / D
    y = Dy / D
    return x, y


def calculate_midpoints(sq):
    cross1 = (sq.top_left, sq.bottom_right)
    cross2 = (sq.top_right, sq.bottom_left)
    center = line_intersection(cross1, cross2)

    right = (sq.top_right, sq.bottom_right)
    left = (sq.top_left, sq.bottom_left)
    vertical = (center, line_intersection(right, left))

    top = (sq.top_left, sq.top_right)
    bottom = (sq.bottom_left, sq.bottom_right)
    horizontal = (center, line_intersection(top, bottom))

    tmid = line_intersection(top, vertical) if vertical[1] else (center[0], sq.top_left[1])
    bmid = line_intersection(bottom, vertical) if vertical[1] else (center[0], sq.bottom_left[1])
    lmid = line_intersection(left, horizontal) if horizontal[1] else (sq.top_left[0], center[1])
    rmid = line_intersection(right, horizontal) if horizontal[1] else (sq.top_right[0], center[1])
    return Midpoints(tmid, rmid, bmid, lmid)
