import numpy as np
import math

stubx = [0, 1, 2, 3, 4]
stuby = [5, 4, 3, 2, 1]


class my_math():

    def __init__(self, x, y):
        assert (len(x) == len(y))
        self.x = x
        self.y = y

    # points in (x,y) format
    def get_angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p2[0] - p3[0], p2[1] - p3[1]])
        if v1.all() == v2.all():
            return 0
        else:
            return abs(math.acos(v1.dot(v2) / ((math.hypot(v1[0], v1[1])) * (math.hypot(v2[0], v2[1])))))

    def get_sum_of_angles(self):
        total = 0
        for i in range(len(self.x) - 2):
            total += self.get_angle_between_points(
                np.array([self.x[i], self.y[i]]), np.array([self.x[i + 1], self.y[i + 1]]),
                np.array([self.x[i + 2], self.y[i + 2]]))
        return total

    # points in (x,y) format
    def get_distance_between_points(self,p1,p2):
        return math.hypot(p1[0] -p2[0], p1[1]-p2[1])

    def get_total_distance(self):
        total = 0
        for i in range(len(self.x) - 1):
            total += self.get_distance_between_points(np.array([self.x[i],self.y[i]]), np.array([self.x[i+1],self.y[i+1]]))
        return total
# mm = my_math(stubx, stuby)
