class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Given three colinear points a, b, c, the function checks if


def is_on_segment(a, b, c):
    if ((b.x <= max(a.x, c.x)) and (b.x >= min(a.x, c.x)) and
            (b.y <= max(a.y, c.y)) and (b.y >= min(a.y, c.y))):
        return True
    return False

def get_orientation(a, b, c):
    val = (float(b.y - a.y) * (c.x - b.x)) - (float(b.x - a.x) * (c.y - b.y))
    # Clockwise
    if (val > 0):
        return 1
    # Counterclockwise
    elif (val < 0):
        return 2
    # Colinear
    else:
        return 0

def is_intersecting(p1, q1, p2, q2):
    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0) and is_on_segment(p1, p2, q1):
        return True
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0) and is_on_segment(p1, q2, q1):
        return True
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0) and is_on_segment(p2, p1, q2):
        return True
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0) and is_on_segment(p2, q1, q2):
        return True
    # If none of the cases
    return False
