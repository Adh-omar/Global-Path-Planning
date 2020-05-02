import numpy as np
import math
import matplotlib.pyplot as plt
from vector import Vector
from scipy.special import comb
import tests


class My_bezier():
    # points are vectors
    def __init__(self, points):
        self.points = points

    def two_points(self, t, p0, p1):
        return p0 * (1 - t) + p1 * t

    def quad_test(self, t, p0, p1, p2):
        return self.two_points(t, self.two_points(t, p0, p1), self.two_points(t, p1, p2))

    def recursive_bez(self, t, points):
        if len(points) == 1:
            return points[0]
        return self.recursive_bez(t, points[:-1]) * (1 - t) + self.recursive_bez(t, points[1:]) * t

    def get_binomial_coeff(self, n, i):
        if i > n:
            return 0
        else:
            return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))

    def get_bezier_for_track(self, t):
        q = Vector(0, 0)
        n = len(self.points)
        for i in range(n):
            coeff = self.get_binomial_coeff(n, i) * (t ** i) * ((1 - t) ** (n - i))
            # coeff = comb(n,i)
            q = q + self.points[i] * coeff
        return q

    def plot(self):
        track = tests.generate_track()
        xt = track[:, 0]
        yt = track[:, 1]
        i2 = 2
        i2 += 0
        i3 = 14
        i2 += 0
        midi = 8
        midi += 0
        pointarr = [Vector((xt[i2] + xt[i2 + 1]) / 2, (yt[i2] + yt[i2 + 1]) / 2), Vector(xt[midi], yt[midi]), Vector(xt[
                                                                                                                         midi + 1],
                                                                                                                     yt[
                                                                                                                         midi + 1]),
                    Vector((xt[i3] + xt[i3 + 1]) / 2, (yt[i3] + yt[i3 + 1]) / 2)]
        xs = [(xt[i2] + xt[i2 + 1]) / 2, xt[midi], xt[midi + 1], (xt[i3] + xt[i3 + 1]) / 2]
        ys = [(yt[i2] + yt[i2 + 1]) / 2, yt[midi], yt[midi + 1], (yt[i3] + yt[i3 + 1]) / 2]
        plt.scatter(xt, yt, color = 'orange')
        tarr = np.arange(0, 1.01, 0.01)
        xplot = np.zeros(tarr.shape)
        yplot = np.zeros(tarr.shape)
        for i in range(len(tarr)):
            # points = self.points
            v = self.recursive_bez(tarr[i], pointarr)
            yplot[i] = v.y
            xplot[i] = v.x
        print(xplot[-1], yplot[-1])
        plt.plot(xplot, yplot, '-', color='blue')
        # xplot = [self.points[i].x for i in range(len(self.points))]
        # yplot = [self.points[i].y for i in range(len(self.points))]
        plt.plot(xs, ys, '--', color='red')
        pointarr2 = [Vector(xt[midi], yt[midi]), Vector(xt[midi + 1], yt[midi + 1])]
        for i in range(len(tarr)):
            # points = self.points
            v = self.recursive_bez(tarr[i], pointarr2)
            yplot[i] = v.y
            xplot[i] = v.x
        print(xplot[-1], yplot[-1])
        plt.plot(xplot, yplot, '--', color='green')
        # plt.scatter(xplot, yplot, color='red')
        plt.show()


# points = [Vector(0, 0), Vector(-1, 8), Vector(7, -4), Vector(7, -4), Vector(12, 9)]
# mb = My_bezier(points)
# print(mb.get_binomial_coeff(5, 5))
# mb.plot()
