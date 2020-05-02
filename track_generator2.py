import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Track_Generator2():
    def __init__(self, resolution, radius):
        self.resolution = math.pi / resolution
        self.radius = radius
        self.track = self.generate_circle(radius)

    def generate_circle(self, radius):
        circle = np.array(
            [[math.cos(x) * radius, math.sin(x) * radius] for x in np.arange(0.1, math.pi * 2, self.resolution)])
        print(circle, "circle")
        return circle

    def get_random_array(self):
        randarray = np.zeros(len(self.track))
        for i in range(len(self.track)):
            r = random.random()
            if r > 0.7:
                randarray[i] = 1
        return randarray

    def deform(self):
        randarray = self.get_random_array()
        for i in range(len(randarray)):
            if (randarray[i] == 1):
                r = random.uniform(self.radius * 0.3, self.radius * 1.7)
                hypotenuse = math.hypot(self.track[i][0], self.track[i][1])
                # hypotenuse = abs(math.atan(self.track[i][1]/self.track[i][0]))
                print(hypotenuse, "hypotenuse")
                newhyp = r
                print(newhyp, "newhyp")
                self.track[i][0] = (newhyp * self.track[i][0]) / hypotenuse
                self.track[i][1] = (newhyp * self.track[i][1]) / hypotenuse

    def smooth_out(self):
        x = 1

    def plot(self):
        xplot = self.track[:, 0]
        yplot = self.track[:, 1]
        print(self.track, "jj")
        plt.scatter(xplot, yplot)
        plt.plot(xplot, yplot)
        xmin = np.amin(xplot)
        xmax = np.amax(xplot)
        ymin = np.amin(yplot)
        ymax = np.amax(yplot)
        if xmin < ymin:
            overallmin = xmin
        else:
            overallmin = ymin
        if xmax > ymax:
            overallmax = xmax
        else:
            overallmax = ymax
        plt.xlim(overallmin + overallmin * 0.3, overallmax + overallmax * 0.3)
        plt.ylim(overallmin + overallmin * 0.3, overallmax + overallmax * 0.3)
        plt.show()


t = Track_Generator2(30, 4)
t.deform()
t.plot()
