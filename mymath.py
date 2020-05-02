from vector import Vector
import matplotlib as plt
import numpy as np
import math


class mymath():

    def get_circle_center_from_points(self, p1, p2, p3):
        temp = p2.sqr_magnitude()
        bc = (p1.sqr_magnitude() - temp) / 2.0
        cd = (temp - p3.sqr_magnitude()) / 2.0
        det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)
        if abs(det) < 0.000001:
            return Vector(None, None)
        det = 1 / det
        # print("hh")
        return Vector((bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * det, ((p1.x - p2.x) * cd - (p2.x - p3.x) * bc) * det)

    def get_curvature_from_points(self, p1, p2, p3):
        cc = self.get_circle_center_from_points(p1, p2, p3)
        if cc.x == None:
            return 0
        radius = (cc - p2).magnitude()
        radius = max(radius, 0.0000001)
        return 1 / radius

    def is_collinear(self,p1, p2, p3):
        a = p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)
        if (a == 0):
            return True
        else:
            return False

    def get_angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p2[0] - p3[0], p2[1] - p3[1]])
        return math.degrees(abs(math.acos(v1.dot(v2) / ((math.hypot(v1[0], v1[1])) * (math.hypot(v2[0], v2[1]))))))

    def get_menger_curvature(self, p1, p2, p3):
        if self.is_collinear(p1,p2,p3):
            print("here")
            return 0
        p12 = p2 - p1
        p23 = p3 - p2
        p13 = p3 - p1
        temp = (p12.dot(p23))/(p12.magnitude()*p23.magnitude())
        if temp > 1:
            temp = 1
        angle_at_p2 = math.pi - math.acos(temp)
        return 2*math.sin(angle_at_p2)/p13.magnitude()

# m = mymath()
# print(m.get_menger_curvature(Vector(0,0), Vector(4,1), Vector(0,2)))
