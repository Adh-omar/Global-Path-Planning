import numpy as np
import random
import matplotlib.pyplot as plt
import math
from vector import Vector
import Point
from Point import Point
from Point import is_intersecting
import time

class Track_gen():
    def __init__(self,track_width,angle_step,inter_dist,features):
        self.track_width = track_width
        self.angle_step = math.radians(angle_step)
        self.track = [(0,0)]
        self.cones = None #placeholder
        self.current_angle = 0
        self.inter_dist = inter_dist
        self.generate_track(features)
        self.place_cones()
    def generate_straight_line(self,length):
        ret = [tuple([self.track[-1][0], self.track[-1][1]])]
        for i in range(int(length)):
            x = ret[-1][0] + self.inter_dist * math.sin(self.current_angle)
            y = ret[-1][1] + self.inter_dist * math.cos(self.current_angle)
            ret.append(tuple([x,y]))
        return ret[1:]
    def generate_right_bend(self,radius,angle):
        ret = []
        angle = math.radians(angle)
        n = math.floor(angle / self.angle_step)
        ogx = self.track[-1][0] + radius * math.cos(self.current_angle)
        ogy = self.track[-1][1] - radius * math.sin(self.current_angle)
        for i in range(1,n+1):
            x = ogx - radius * math.cos(i*self.angle_step + self.current_angle)
            y = ogy + radius * math.sin(i*self.angle_step + self.current_angle)
            ret.append(tuple([x,y]))
        # try:
        #     ret.append(tuple([ret[-1][0] + self.inter_dist * math.sin(self.current_angle+angle),ret[-1][1] + self.inter_dist * math.cos(self.current_angle+angle)]))
        # except IndexError:
        #     print("here Index")
        return ret
    def generate_left_bend(self,radius,angle):
        ret = []
        angle = math.radians(angle)
        n = math.floor(angle / self.angle_step)
        ogx = self.track[-1][0] - radius * math.cos(self.current_angle)
        ogy = self.track[-1][1] - radius * math.sin(self.current_angle)
        # print(ogx,ogy,"Sd")
        for i in range(1,n+1):
            x = ogx + radius * math.cos(i*self.angle_step - self.current_angle)
            y = ogy + radius * math.sin(i*self.angle_step - self.current_angle)
            ret.append(tuple([x,y]))
        # try:
        #     ret.append(tuple([ret[-1][0] + self.inter_dist * math.sin(self.current_angle-angle),ret[-1][1] + self.inter_dist * math.cos(self.current_angle-angle)]))
        # except IndexError:
        #     print("here index")
        return ret

    def correct_track_angle(self, angle):
        self.current_angle += angle
        if self.current_angle < 0:
            self.current_angle += 2 * math.pi
        if self.current_angle >= 2 * math.pi:
            self.current_angle -= 2 * math.pi

    def place_cones(self):
        self.cones = np.zeros((2*len(self.track)-2,2))
        for i in range(len(self.track)-1):
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

    def is_in_rectangle(self,x1, y1, x2,
                  y2, x, y):
        if (x > x1 and x < x2 and
                y > y1 and y < y2):
            return True
        else:
            return False

    def my_append(self,arr):
        for i in arr:
            self.track.append(i)

    def generate_track(self,features):
        self.my_append(self.generate_straight_line(3))
        for i in range(features):
            done = False
            arr = None
            while not done:
                direction = random.randint(0,3)
                length = random.randint(1,12)
                radius = random.uniform(3,9)
                angle = random.uniform(0,180)
                if direction == 0 or direction == 1:
                    arr = self.generate_straight_line(length)
                    # print("str" + str(length))
                elif direction == 2:
                    arr = self.generate_left_bend(radius,angle)
                    # print("left" + str(radius) + " " + str(angle))
                elif direction == 3:
                    arr = self.generate_right_bend(radius,angle)
                    # print("right" + str(radius) + " " + str(angle))
                if len(arr) > 0:
                    intersection = False
                    for i in range(len(self.track) - 1):
                        # print(self.track[i][0],"SDF")
                        p1 = Point(self.track[i][0], self.track[i][1])
                        p2 = Point(self.track[i + 1][0], self.track[i + 1][1])
                        for j in range(len(arr) - 1):
                            q1 = Point(arr[j][0], arr[j][1])
                            q2 = Point(arr[j + 1][0], arr[j + 1][1])
                            if is_intersecting(p1, p2, q1, q2):
                                # print("here")
                                intersection = True
                                break
                    if not intersection:
                        self.my_append(arr)
                        #correct angles if turns
                        if direction == 3:
                            self.correct_track_angle(math.radians(angle))
                        elif direction == 2:
                            self.correct_track_angle(math.radians(-angle))
                        # self.correct_track_angle(angle)
                        # print("dir =" + str(direction) + " radius " + str(radius) + " angle " + str(angle) + " length " + str(length))
                        done = True
                        break


    def plot(self):
        xplot = [self.track[i][0] for i in range(len(self.track))]
        yplot = [self.track[i][1] for i in range(len(self.track))]
        xplotcone = self.cones[:,0]
        yplotcone = self.cones[:,1]
        # plt.scatter(xplot,yplot)
        plt.scatter(xplotcone,yplotcone)
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

    def get_track(self):
        return self.cones

# Performance testing code
# y = np.zeros(51)
# for i in range(20, 71):
#     sum = 0
#     for j in range(20):
#         start_time = time.time()
#         t = Track_gen(4,30,3,i)
#         t.place_cones()
#         sum += time.time() - start_time
#     y[i - 20] = sum / 20
#     print(i)
# print(y)
# plt.plot(np.arange(20, 71, 1), y, '-', color='green')
# plt.xlabel("Number of features in track")
# plt.ylabel("Time taken to produce track (s)")
# plt.show()


# Example of custom track creation
# t = Track_gen(4,30,4,9)
# t.plot()
# t.my_append(t.generate_straight_line(3))
# t.my_append(t.generate_right_bend(4,40))
# t.correct_track_angle(-40)
# t.my_append(t.generate_straight_line(4))
# t.my_append(t.generate_right_bend(6,70))
# t.correct_track_angle(-70)
# t.my_append(t.generate_straight_line(5))
# t.generate_left_bend(4,120)
# t.generate_straight_line(12)
# t.generate_left_bend(3,180)
# t.generate_straight_line(12)
# t.generate_left_bend(1,60)
# t.generate_straight_line(12)
# t.place_cones()
# t.plot()
#




