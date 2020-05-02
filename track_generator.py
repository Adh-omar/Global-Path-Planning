import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Track_Generator:
    def __init__(self, track_width, angle_step):
        self.track_width = track_width
        self.track = np.array([[0., 0.], [self.track_width, 0]])
        self.angle_step = math.radians(angle_step)
        self.current_angle = 0

    def correct_track_angle(self, angle):
        self.current_angle += angle
        if self.current_angle < 0:
            self.current_angle += 2 * math.pi
        if self.current_angle >= 2 * math.pi:
            self.current_angle -= 2 * math.pi

    def generate_right_bend(self, radius, angle):
        angle = math.radians(angle)
        polarpoint = [self.track[len(self.track) - 1][0] + radius * math.cos(self.current_angle),
                      self.track[len(self.track) - 1][1] - radius * math.sin(self.current_angle)]
        i = math.floor(self.current_angle / self.angle_step)
        while i * self.angle_step <= angle + self.current_angle:
            x = polarpoint[0] - radius * math.cos(i * self.angle_step)
            y = polarpoint[1] + radius * math.sin(i * self.angle_step)
            x1 = polarpoint[0] - (radius + self.track_width) * math.cos(i * self.angle_step)
            y1 = polarpoint[1] + (radius + self.track_width) * math.sin(i * self.angle_step)
            i += 1
            self.track = np.append(self.track, [[x1, y1], [x, y]], axis=0)
        self.correct_track_angle(angle)

    def generate_left_bend(self, radius, angle):
        angle = math.radians(angle)
        polarpoint = [self.track[len(self.track) - 2][0] - radius * math.cos(self.current_angle),
                      self.track[len(self.track) - 2][1] + radius * math.sin(self.current_angle)]
        i = math.floor(self.current_angle / self.angle_step)
        while i * self.angle_step >= self.current_angle - angle:
            x = polarpoint[0] + radius * math.cos(-i * self.angle_step)
            y = polarpoint[1] + radius * math.sin(-i * self.angle_step)
            x1 = polarpoint[0] + (radius + self.track_width) * math.cos(-i * self.angle_step)
            y1 = polarpoint[1] + (radius + self.track_width) * math.sin(-i * self.angle_step)
            i -= 1
            self.track = np.append(self.track, [[x, y], [x1, y1]], axis=0)
        self.correct_track_angle(-angle)

    def generate_straight_line(self, length):
        for i in range(int(length)):
            origin_point = self.track[len(self.track) - 1]
            x = self.track_width * math.sin(self.current_angle) + origin_point[0]
            y = self.track_width * math.cos(self.current_angle) + origin_point[1]
            origin_point = self.track[len(self.track) - 2]
            x2 = self.track_width * math.sin(self.current_angle) + origin_point[0]
            y2 = self.track_width * math.cos(self.current_angle) + origin_point[1]
            self.track = np.append(self.track, [[x2, y2], [x, y]], axis=0)

    def shift_track_to_positive_quadrant(self):
        x = self.track[:, 0]
        minx = min(x)
        y = self.track[:, 1]
        miny = min(y)
        if miny < 0:
            y = np.array([u - miny for u in y])
        if minx < 0:
            x = np.array([u - minx for u in x])
        self.track = np.vstack((x, y)).T


    def plot(self):
        xplot = self.track[:, 0]
        yplot = self.track[:, 1]
        plt.scatter(xplot, yplot)
        # xmin = np.amin(xplot)
        # xmax = np.amax(xplot)
        # ymin = np.amin(yplot)
        # ymax = np.amax(yplot)
        # if xmin < ymin:
        #     overallmin = xmin
        # else:
        #     overallmin = ymin
        # if xmax > ymax:
        #     overallmax = xmax
        # else:
        #     overallmax = ymax
        # plt.xlim(overallmin + overallmin * 0.3, overallmax + overallmax * 0.3)
        # plt.ylim(overallmin + overallmin * 0.3, overallmax + overallmax * 0.3)
        plt.show()

    def get_track(self):
        self.generate_straight_line(6)
        self.generate_left_bend(4, 180)
        self.generate_straight_line(5)
        self.generate_right_bend(6, 90)
        self.generate_straight_line(4)
        self.generate_left_bend(6, 90)
        self.generate_straight_line(4)
        self.generate_left_bend(6, 110)
        self.generate_straight_line(7)
        self.generate_left_bend(3, 70)
        self.generate_straight_line(3)
        self.shift_track_to_positive_quadrant()
        return self.track
    def generate_track(self,features):
        for i in range(features):
            direction = random.randint(0,3)
            length = random.uniform(0,12)
            radius = random.uniform(0,9)
            angle = random.uniform(0,180)
            if direction == 0 or direction == 1:
                self.generate_straight_line(length)
            elif direction == 2:
                self.generate_left_bend(radius,angle)
            elif direction == 3:
                self.generate_right_bend(radius,angle)
        print(self.track)




# t = Track_Generator(6, 30)
# print(t.get_track())
# t.plot()
# t.generate_left_bend(4, 180)
# t.generate_straight_line(5)
# t.generate_right_bend(6, 90)
# t.generate_straight_line(4)
# t.generate_left_bend(6, 90)
# t.generate_straight_line(4)
# t.generate_left_bend(6,110)
# t.generate_straight_line(7)
# t.generate_left_bend(3,70)
# t.generate_straight_line(3)
#
# t.generate_track(19)
# t.plot()

# TODO: have some representation of the current direction of the track.
