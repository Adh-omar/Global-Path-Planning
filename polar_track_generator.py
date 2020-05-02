import math
import numpy as np
import matplotlib.pyplot as plt
import random
import noise
from vector import Vector
from perlin import PerlinNoiseFactory
import time


# TODO: add chicanes
class polar_track_generator():
    def __init__(self, resolution, radius, track_width):
        self.radius = radius
        self.random_array = 0
        self.resolution = math.pi / resolution
        self.width = 0
        self.height = 0
        self.track = self.generate_ellipse()
        self.track_width = track_width
        self.cones = np.zeros((2 * len(self.track) - 2, 2))
        self.deform()

        self.place_cones()

    def get_cartesian_coordinates(self, radius, angle):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return x, y

    def get_polar_coordinates(self, x, y):
        r = math.hypot(x, y)
        theta = abs(math.atan(y / x))
        if x < 0 and y > 0:
            theta = math.pi - theta
        elif x < 0 and y < 0:
            theta += math.pi
        elif x > 0 and y < 0:
            theta = -theta + 2 * math.pi
        return r, theta

    def generate_circle(self):
        circle = np.array(
            [[math.cos(x) * self.radius, math.sin(x) * self.radius] for x in
             np.arange(0.1, math.pi * 2, self.resolution)])
        return circle

    def generate_ellipse(self):
        a = random.uniform(1, 9)
        b = random.uniform(1, 9)
        ellipse = np.array([[a * math.cos(x) * self.radius, b * math.sin(x) * self.radius] for x in
                            np.arange(0.0, math.pi * 2, self.resolution)])
        self.width = a
        self.height = b
        return ellipse

    # prob is probability you get a 1
    def get_random_array(self, prob):
        randarray = np.zeros(len(self.track))
        for i in range(len(self.track)):
            r = random.random()
            if r > (1 - prob):
                randarray[i] = 1
        return randarray

    def wrap_index(self, index, array):
        l = len(array)
        if index < 0:
            return index + l
        if index >= l:
            return index - l
        else:
            return index

    def get_slop_of_line(self, x1, y1, x2, y2):
        if x1 == x2:
            return math.inf
        return (y2 - y1) / (x2 - x1)

    def place_cones(self):
        for i in range(len(self.track) - 1):
            B = Vector(self.track[i][0], self.track[i][1])
            A = Vector(self.track[i + 1][0], self.track[i + 1][1])
            AB = B - A
            AB_perp_normed = AB.perp().normalized()
            p1 = B + AB_perp_normed * (self.track_width / 2)
            p2 = B - AB_perp_normed * (self.track_width / 2)
            p1 = p1 - AB.scalar_div(2)  # TODO: decide wether these 2 lines are staying
            p2 = p2 - AB.scalar_div(2)
            self.cones[i * 2][0] = p1.x
            self.cones[i * 2][1] = p1.y
            self.cones[i * 2 + 1][0] = p2.x
            self.cones[i * 2 + 1][1] = p2.y

    def deform(self):
        octaves = random.randint(2, 5)
        # the lower the more smooth
        smoothness = 1.1
        ##https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1
        p = PerlinNoiseFactory(1, octaves=octaves, unbias=True)
        for i in range(len(self.track)):
            r, theta = self.get_polar_coordinates(self.track[i][0], self.track[i][1])
            r += self.radius * smoothness * p.__call__(theta)
            self.track[i][0], self.track[i][1] = self.get_cartesian_coordinates(r, theta)

    def deform2(self):
        offset = random.uniform(-5, 5)
        octaves = random.randint(2, 9)
        repeat_offset = random.uniform(-self.resolution / 4, self.resolution / 4)
        for i in range(len(self.track)):
            r, theta = self.get_polar_coordinates(self.track[i][0], self.track[i][1])

            r += self.radius * noise.pnoise1(theta + offset, octaves=octaves, persistence=0.2, lacunarity=1.5,
                                             repeat=math.floor(self.resolution * math.pi))
            self.track[i][0], self.track[i][1] = self.get_cartesian_coordinates(r, theta)

    # place is where the feature will be added
    def add_feature(self, place, arr):
        # assert (place < len(self.track))
        # startx = self.track[place][0]
        # starty = self.track[place][1]
        # for i in range(len(arr)):
        #     self.track[self.wrap_index(i + place, self.track)][0] = startx + arr[i][0]
        #     self.track[self.wrap_index(i + place, self.track)][1] = starty + arr[i][1]
        for i in range(len(arr)):
            r, theta = self.get_polar_coordinates(self.track[self.wrap_index(i, self.track)][0],
                                                  self.track[self.wrap_index(i, self.track)][1])
            r += r*arr[i]
            self.track[self.wrap_index(i, self.track)][0], self.track[self.wrap_index(i, self.track)][
                1] = self.get_cartesian_coordinates(r, theta)

    def add_chicane(self):
        chicane = np.array(
            [[0, 1], [0, 1.5], [0, 1.8], [0, 1.5], [0, 1], [0, 0], [0, -0.5], [0, -1], [0, -0.5], [0, 0], [0, 1],
             [0, 1.5], [0, 1.8],
             [0, 1.5], [0, 1]])
        chicane = [1, 1.5, 1.8, 1.6, 1, 0, -0.5, -1, -0.5, 0, 1, 1.5, 1.8]
        # chicane = [chic * 10 for chic in chicane]
        print(chicane)
        self.add_feature(32, chicane)

    def get_track(self):
        return self.cones

    def plot(self):
        xplot = self.track[:, 0]
        yplot = self.track[:, 1]
        # plt.scatter(xplot, yplot, color='green')
        # plt.plot(xplot, yplot)
        xplotcone = self.cones[:, 0]
        yplotcone = self.cones[:, 1]
        # print(self.track, "strack")
        plt.scatter(xplotcone, yplotcone, color="blue")
        # plt.scatter(0, 0, color="black")
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

    def test_plot(self):
        ellipse = self.generate_ellipse()
        xplot = ellipse[:, 0]
        yplot = ellipse[:, 1]
        plt.scatter(xplot, yplot, color="orange")
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
        plt.xlim(overallmin + overallmin * 0.1, overallmax + overallmax * 0.1)
        plt.ylim(overallmin + overallmin * 0.1, overallmax + overallmax * 0.1)
        plt.show()




# Performance testing code
# y = np.zeros(81)
# for i in range(20, 101):
#     sum = 0
#     for j in range(20):
#         start_time = time.time()
#         t = polar_track_generator(i, 8, 8)
#         t.place_cones()
#         sum += time.time() - start_time
#     y[i - 20] = sum / 20
#     print(i)
# print(y)
# plt.plot(np.arange(20, 101, 1), y, '-', color='green')
# plt.xlabel("Number of cone pairs in track")
# plt.ylabel("Time taken to produce track (s)")
# plt.show()
# # start_time = time.time()
# # t = polar_track_generator(48, 8, 8)
# # # t.add_chicane()
# # t.place_cones()
# # # print(t.track.shape)
# # print("took " + str(time.time()-start_time) + " seconds to run")
# # t.plot()