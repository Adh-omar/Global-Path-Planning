import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import sympy as sym
from vector import Vector

# stuby = np.array([1, 5, 3, 8, 12, 7, 19])
# stubx = np.array([0, 2, 6, 9, 12, 18, 24])
# v = np.array([[stubx[i] ** j for j in range(len(stubx))] for i in range(len(stubx))])
# print(v)


class calculus:
    def __init__(self, x, y):
        assert (len(x) == len(y))
        # if len(x) < len(y):
        #     a = x
        #     x = y
        #     y = a
        self.x = x
        self.y = y
        self.bool = False
        seen = set()
        for i in range(len(self.x)):
            while (self.x[i] in seen):
                self.x[i] += 0.01
            seen.add(self.x[i])
        self.curve_coeff = self.interpolatex()

    def interpolatex(self):
        v = np.array([[self.x[i] ** j for j in range(len(self.x))] for i in range(len(self.x))])
        vinv = np.linalg.inv(v)
        coeff = np.matmul(vinv, self.y)
        return coeff

    def get_coefficients(self):
        return self.curve_coeff

    def diff(self, coeff):
        ret = np.array([coeff[i] * i for i in range(1, len(coeff))])
        return ret

    def calculate_curve_value_for_point(self, coeff, x):
        ret = 0
        for i in range(len(coeff)):
            ret += coeff[i] * (x ** i)
        return ret

    def calculate_curvature_at_point(self, x, coeff):
        dydx = self.diff(coeff)
        dydxval = self.calculate_curve_value_for_point(dydx, x)
        dy2dx = self.diff(dydx)
        d2ydxval = self.calculate_curve_value_for_point(dy2dx, x)
        curvature = d2ydxval / ((1 + (dydxval ** 2))) ** 3 / 2
        return curvature

    def calculate_curvature_for_curve(self):
        total = 0
        coeff = self.interpolatex()
        for i in range(len(self.x)):
            total += abs(self.calculate_curvature_at_point(self.x[i], coeff))
        return total

    # def circle_center_from_3_points(self, ):

    def wrap_index(self, index, array):
        len = len(array)
        if index < 0:
            return index + len
        if index >= len:
            return index - len

    def plot(self):
        xnew = np.arange(min(self.x), max(self.x) + 0.1, 0.1)
        ynew = 0
        for i in range(len(self.x)):
            ynew += self.curve_coeff[i] * (xnew ** i)
        plt.plot(self.x, self.y, 'o', xnew, ynew, '-')
        plt.show()

#
# c = calculus(stubx, stuby)
# c.interpolatex()
# # print(c.calculate_curvature_for_curve())
# c.plot()
