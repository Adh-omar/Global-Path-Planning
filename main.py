import numpy as np
import pandas as pd
import tests
import random
import matplotlib.pyplot as plt
from calculus import calculus
from polar_track_generator import polar_track_generator
import math
import time
from progress.bar import Bar
from track_generator import Track_Generator
from my_math import my_math
from autograd import grad
from autograd import elementwise_grad as egrad
# from sympy import *
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds
import scipy.optimize as optimize
from mymath import mymath
from vector import Vector
from my_bezier import My_bezier
import bezier
from track_gen import Track_gen

# every even index cone is paired with its +1 odd index cone
stub = np.array([[0.0, 0.0], [4, 0], [0, 4], [4, 4], [0, 8], [4, 8]])
# print(len(stub)/2,"k")
stub2 = np.zeros((len(stub), 2))
n = 4
k_sp_stub = [0.5, 0.4, 0.4, 0.4, 0.3, 0.5, 0.6, 0.5, 0.4, 0.4, 0.4, 0.3, 0.5, 0.6, 0.5, 0.4, 0.4, 0.4, 0.3, 0.5, 0.6,
             0.5, 0.4, 0.4, 0.4, 0.3, 0.5, 0.6]


# print(stub2)


class Track:
    def __init__(self, resolution):
        # self.track_generator = Track_Generator(8, 30)
        # self.track_generator = Track_gen(10,30,4,10)
        self.track_generator = polar_track_generator(resolution, 8, 8)
        # self.cone_positions = tests.generate_track()
        self.cone_positions = self.track_generator.get_track()
        # self.cone_positions = tests.generate_track()
        self.shift_track_to_positive_quadrant()
        self.k_positions_MCP = np.zeros((int(len(self.cone_positions) / 2), 2))
        self.k_arr_MCP = np.array([self.initialize_MCP(x) for x in range(math.ceil(len(self.cone_positions) / 2))])
        self.k_positions_SP = np.zeros((int(len(self.cone_positions) / 2), 2))
        self.k_arr_SP = np.array([0.5 for x in range(math.ceil(len(self.cone_positions) / 2))])
        self.k_positions_FINAL = None  # just a plave holder
        self.k_arr_ECHO = np.array([0.5 for x in range(math.ceil(len(self.cone_positions) / 2))])
        self.ECHO_positions = None

    def initialize_MCP(self, x):
        if x % 2 == 0:
            return 0.4
        else:
            return 0.3

    # k*2 and k*2+1 give you the cone pair that k is associated with
    # redundant method
    def initialize_k(self):
        for i in range(0, len(self.k_positions_MCP)):
            self.k_positions_MCP[i][0] = (self.cone_positions[i * 2][0] + self.cone_positions[i * 2 + 1][0]) / 2
            self.k_positions_MCP[i][1] = (self.cone_positions[i * 2][1] + self.cone_positions[i * 2 + 1][1]) / 2

    def shift_track_to_positive_quadrant(self):
        x = self.cone_positions[:, 0]
        minx = min(x)
        y = self.cone_positions[:, 1]
        miny = min(y)
        if miny < 0:
            y = np.array([u - miny for u in y])
        if minx < 0:
            x = np.array([u - minx for u in x])
        self.cone_positions = np.vstack((x, y)).T

    # array is jus k values between 0 and 1
    def objective_function_SP(self, array):
        ret = 0
        for i in range(len(array) - 1):
            x1 = self.cone_positions[i * 2][0] + (
                    self.cone_positions[i * 2 + 1][0] - self.cone_positions[i * 2][0]) * array[i]
            y1 = self.cone_positions[i * 2][1] + (
                    self.cone_positions[i * 2 + 1][1] - self.cone_positions[i * 2][1]) * array[i]
            x2 = self.cone_positions[(i + 1) * 2][0] + (
                    self.cone_positions[(i + 1) * 2 + 1][0] - self.cone_positions[(i + 1) * 2][0]) * array[i + 1]
            y2 = self.cone_positions[(i + 1) * 2][1] + (
                    self.cone_positions[(i + 1) * 2 + 1][1] - self.cone_positions[(i + 1) * 2][1]) * array[i + 1]
            ret += math.hypot(x2 - x1, y2 - y1)
            # print(ret)
        return ret

    def get_angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p2[0] - p3[0], p2[1] - p3[1]])
        return abs(math.acos(v1.dot(v2) / ((math.hypot(v1[0], v1[1])) * (math.hypot(v2[0], v2[1])))))

    def objective_function_MCP2(self, array):
        kpos = self.get_k_set_in_cartesian(array)
        total = 0
        for i in range(len(kpos) - 4, 3):
            cl = calculus(kpos[i:i + 4, 0], kpos[i:i + 4, 1])
            total += cl.calculate_curvature_for_curve()
        return total

    def objective_function_MCP3(self, array):
        kpos = self.get_k_set_in_cartesian(array)
        mm = my_math(kpos[:, 0], kpos[:, 1])
        return mm.get_sum_of_angles()

    def objective_function_MCP(self, array):
        kpos = self.get_k_set_in_cartesian(array)
        total = 0
        mm = mymath()
        for i in range(1, len(kpos) - 2, 2):
            p1 = Vector(kpos[i - 1][0], kpos[i - 1][1])
            p2 = Vector(kpos[i][0], kpos[i][1])
            p3 = Vector(kpos[i + 1][0], kpos[i + 1][1])
            total += mm.get_menger_curvature(p1, p2, p3)

        return -total

    def objective_function_MCP4(self, array):
        kpos = self.get_k_set_in_cartesian(array)
        total = 0
        for i in range(1, len(kpos) - 1, 1):
            cl = calculus(kpos[i - 1:i + 2, 0], kpos[i - 1:i + 2, 1])
            total += cl.calculate_curvature_for_curve()
        return total

    def limit_array(self, array, ub, lb):
        for i in range(len(array)):
            if array[i] > ub:
                array[i] = 0.8
            elif array[i] < lb:
                array[i] = lb
        return array

    def minimize_objective_functions(self):
        bounds = Bounds(lb=0.2, ub=0.8)
        res_SP = minimize(self.objective_function_SP, self.k_arr_SP, method='SLSQP', bounds=bounds)
        self.k_arr_SP = res_SP.x
        # res_MCP = minimize(self.objective_function_MCP4, self.k_arr_MCP, method='L-BFGS-B',
        #                    bounds=bounds)  # L-BFGS-B TNC
        # self.k_arr_MCP = res_MCP.x
        self.get_ECHO_positions()

    # num is the number of ksets
    def mutate_k_MCP(self, num):
        l_threshold = 0.2
        u_threshold = 0.8
        mutation_range = 0.05
        k_sets = np.zeros((num, len(self.k_arr_MCP)))
        for j in range(num):
            curr_k_set = self.k_arr_MCP
            for i in range(1, len(self.k_arr_MCP)):
                rand = random.uniform(-mutation_range, mutation_range)
                while curr_k_set[i] + rand < l_threshold or curr_k_set[i] + rand > u_threshold:
                    rand = random.uniform(-mutation_range, mutation_range)
                curr_k_set[i] += rand
            k_sets[j] = curr_k_set
        return k_sets

    def mutate_k_SP(self, num):
        l_threshold = 0.2
        u_threshold = 0.8
        mutation_range = 0.05
        k_sets = np.zeros((num, len(self.k_arr_SP)))
        for j in range(num):
            curr_k_set = self.k_arr_SP
            for i in range(1, len(self.k_arr_SP)):
                rand = random.uniform(-mutation_range, mutation_range)
                while curr_k_set[i] + rand < l_threshold or curr_k_set[i] + rand > u_threshold:
                    rand = random.uniform(-mutation_range, mutation_range)
                curr_k_set[i] += rand
            k_sets[j] = curr_k_set
        return k_sets

    def get_k_set_in_cartesian(self, array):
        ret = np.zeros((len(array), 2))
        for i in range(len(array)):
            ret[i][0] = self.cone_positions[i * 2][0] + (
                    self.cone_positions[i * 2 + 1][0] - self.cone_positions[i * 2][0]) * array[i]
            ret[i][1] = self.cone_positions[i * 2][1] + (
                    self.cone_positions[i * 2 + 1][1] - self.cone_positions[i * 2][1]) * array[i]
        return ret

    def get_piecewise_curvature(self, n, kpos):
        total = 0
        for i in range(math.ceil(len(kpos) / n)):
            arr = kpos[n * i:n * i + n + 2]
            arr1 = arr[:, 0]
            arr2 = arr[:, 1]
            if max(arr2) - min(arr2) > max(arr1) - min(arr1):
                cl = calculus(arr2, arr1)
            else:
                cl = calculus(arr1, arr2)
            total += cl.calculate_curvature_for_curve()
        return total

    def get_best_k_set_MCP3(self, num):
        k_sets = self.mutate_k_MCP(num)
        curr = k_sets[0]
        curr_kpos = self.get_k_set_in_cartesian(curr)
        mm = mymath()
        curr_min_sum = mm.get_menger_curvature(Vector(curr_kpos[0][0], curr_kpos[0][1]),
                                               Vector(curr_kpos[1][0], curr_kpos[1][1]),
                                               Vector(curr_kpos[2][0], curr_kpos[2][1]))
        for i in range(1, len(k_sets[0]) - 1, 2):
            kpos = self.get_k_set_in_cartesian(k_sets[i])
            curr_sum = mm.get_menger_curvature(Vector(kpos[i - 1][0], kpos[i - 1][1]), Vector(kpos[i][0], kpos[i][1]),
                                               Vector(kpos[i + 1][0], kpos[i + 1][1]))
            if curr_sum < curr_min_sum:
                curr = k_sets[i]
                curr_min_sum = curr_sum
                curr_kpos = kpos
        return curr, curr_min_sum, curr_kpos

    def get_best_k_set_MCP(self, num):
        k_sets = self.mutate_k_MCP(num)
        curr = k_sets[0]
        curr_kpos = self.get_k_set_in_cartesian(curr)
        mm = my_math(curr_kpos[:, 0], curr_kpos[:, 1])
        curr_min_sum = mm.get_sum_of_angles()
        for i in range(len(k_sets)):
            kpos = self.get_k_set_in_cartesian(k_sets[i])
            mm = my_math(kpos[:, 0], kpos[:, 1])  # mm = my_math(curr_kpos[:, 0], curr_kpos[:, 1])
            sum_of_angles = mm.get_sum_of_angles()
            if sum_of_angles < curr_min_sum:
                curr = k_sets[i]
                curr_min_sum = sum_of_angles
                curr_kpos = kpos
        return curr, curr_min_sum, curr_kpos

    def get_best_k_set_MCP2(self, num):
        k_sets = self.mutate_k_MCP(num)
        curr = k_sets[0]
        curr_kpos = self.get_k_set_in_cartesian(curr)
        curr_min = 0
        for i in range(len(k_sets)):
            kpos = self.get_k_set_in_cartesian(k_sets[i])
            curr_sum = 0
            for i in range(len(kpos), 3):
                cl = calculus(kpos[i:i + 4, 0], kpos[i:i + 4, 1])
                curr_sum += cl.calculate_curvature_for_curve()
            if curr_min < curr_sum:
                curr = k_sets[i]
                curr_min = curr_sum
                curr_kpos = kpos
        return curr, curr_min, curr_kpos

    def get_best_k_set_SP(self, num):
        k_sets = self.mutate_k_SP(num)
        curr = k_sets[0]
        curr_kpos = self.get_k_set_in_cartesian(curr)
        mm = my_math(curr_kpos[:, 0], curr_kpos[:, 1])
        curr_min_dist = mm.get_total_distance()
        for i in range(len(k_sets)):
            kpos = self.get_k_set_in_cartesian(k_sets[i])
            mm = my_math(kpos[:, 0], kpos[:, 1])
            dist = mm.get_total_distance()
        if dist < curr_min_dist:
            curr = k_sets[i]
            curr_min_dist = dist
            curr_kpos = kpos
        return curr, curr_min_dist, curr_kpos

    def get_best_k_set2(self, num):
        k_sets = self.mutate_k_MCP(num)
        curr = k_sets[0]
        curr_kpos = self.get_k_set_in_cartesian(curr)
        mm = my_math(curr_kpos[:, 0], curr_kpos[:, 1])
        curr_min_curvature = self.get_piecewise_curvature(n, curr_kpos)
        for i in range(len(k_sets)):
            kpos = self.get_k_set_in_cartesian(k_sets[i])
            curvature = self.get_piecewise_curvature(n, kpos)
            if curvature < curr_min_curvature:
                curr = k_sets[i]
                curr_min_curvature = curvature
                curr_kpos = kpos
        return curr, curr_min_curvature, curr_kpos

    def plot_points(self):
        self.k_positions_SP = self.get_k_set_in_cartesian(self.k_arr_SP)
        # self.k_positions_MCP = self.get_k_set_in_cartesian(self.k_arr_MCP)
        self.ECHO_positions = self.get_k_set_in_cartesian(self.k_arr_ECHO)
        self.k_positions_FINAL = np.average([self.k_positions_SP, self.ECHO_positions], axis=0, weights=[0.4, 0.6])

    def genetic_algorithm(self, generations):
        for i in range(generations):
            self.k_arr_MCP, min_curvature, self.k_positions_MCP = self.get_best_k_set_MCP(500)
            # self.k_arr_SP, dist, self.k_positions_SP = self.get_best_k_set_SP(1000)

    def get_ECHO_positions(self):
        # for every cone pair
        for i in range(6, len(self.cone_positions) - 6, 2):
            nodes_curve = np.asfortranarray([[(self.cone_positions[i - 6][0] + self.cone_positions[i - 5][0]) / 2,
                                self.cone_positions[i][0], self.cone_positions[i + 1][0],
                                (self.cone_positions[i + 6][0] + self.cone_positions[i + 7][0]) / 2],
                                [(self.cone_positions[i - 6][1] + self.cone_positions[i - 5][1]) / 2,
                                self.cone_positions[i][1], self.cone_positions[i + 1][1],
                                (self.cone_positions[i + 6][1] + self.cone_positions[i + 7][
                                1]) / 2]])
            curve = bezier.Curve.from_nodes(nodes_curve)
            nodes_line = np.asfortranarray([[self.cone_positions[i][0], self.cone_positions[i + 1][0]],
                                            [self.cone_positions[i][1], self.cone_positions[i + 1][1]]])
            line = bezier.Curve.from_nodes(nodes_line)
            intersection = curve.intersect(line)
            try:
                self.k_arr_ECHO[int(i / 2)] = intersection[1][0]
            except IndexError:
                self.k_arr_ECHO[int(i / 2)] = 0.5
            # To respect the bounds to allow for the width of the car
            if self.k_arr_ECHO[int(i / 2)] < 0.1:
                self.k_arr_ECHO[int(i / 2)] = 0.1
            elif self.k_arr_ECHO[int(i / 2)] > 0.9:
                self.k_arr_ECHO[int(i / 2)] = 0.9
        # return intersections

    # returns an array with the curvature value at each point
    def get_curvature_of_final_path(self):
        kpos = self.k_positions_FINAL
        ret = np.zeros(len(kpos))
        mm = mymath()
        for i in range(1, len(kpos) - 1, 1):
            p1 = Vector(kpos[i - 1][0], kpos[i - 1][1])
            p2 = Vector(kpos[i][0], kpos[i][1])
            p3 = Vector(kpos[i + 1][0], kpos[i + 1][1])
            ret[i] = mm.get_curvature_from_points(p1, p2, p3)
        return ret

    def rearrange_array(self, array):
        ret = np.zeros(array.shape)
        l = len(array)
        for i in range(math.floor(l / 2)):
            ret[i] = array[i * 2]
        counter = math.floor(l / 2)
        for i in range(l):
            if not i % 2 == 0:
                ret[counter] = array[i]
                counter += 1
        return ret

    # p1 is vertex angle
    def get_angle_from_points(self, p1, p2, p3):
        return math.acos((p1.lengthto(p2) ** 2 + p1.lengthto(p3) ** 2 - p2.lengthto(p3) ** 2) / (
                2 * p1.lengthto(p2) * p1.lengthto(p3)))

    def get_direction(self, a, b, c):
        """Returns 1 if the point c lies on the right of the
         line segment ab when the line segment ab is horizontal,
          this function returns 1 when c is below the segment ab,
          returns -1 otherwise"""
        temp = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
        if temp < 0:
            return 1
        else:
            return -1

    def get_angle_from_points2(self, a, b, c):
        ab = b - a
        bc = c - b
        return math.acos((ab.dot(bc)) / (ab.magnitude() * bc.magnitude()))

    def get_steer_angles(self):
        ret = np.zeros((len(self.k_positions_FINAL)))
        for i in range(1, len(ret) - 1):
            pvertex = Vector(self.k_positions_FINAL[i][0], self.k_positions_FINAL[i][1])
            p1 = Vector(self.k_positions_FINAL[i - 1][0], self.k_positions_FINAL[i - 1][1])
            p2 = Vector(self.k_positions_FINAL[i + 1][0], self.k_positions_FINAL[i + 1][1])
            # first determines the sign of the angle and therefore the direction of turning
            ret[i] = self.get_direction(p1, pvertex, p2) * math.degrees(self.get_angle_from_points2(p1, pvertex, p2))
        return ret

    def plot(self):
        xplot = self.cone_positions[:, 0]
        yplot = self.cone_positions[:, 1]
        # kxplot_MCP = self.k_positions_MCP[:, 0]
        # kyplot_MCP = self.k_positions_MCP[:, 1]
        kxplot_SP = self.k_positions_SP[:, 0]
        kyplot_SP = self.k_positions_SP[:, 1]
        kxplot_FINAL = self.k_positions_FINAL[:, 0]
        kyplot_FINAL = self.k_positions_FINAL[:, 1]
        bezxplot = self.ECHO_positions[:, 0]
        bezyplot = self.ECHO_positions[:, 1]
        # plt.plot(xplot, yplot, '-', color='grey',)
        plt.scatter(xplot, yplot)
        # plt.plot(kxplot_MCP, kyplot_MCP, '-', color='blue', label='minimum curvature path')
        plt.plot(kxplot_SP, kyplot_SP, '-', color='green', label='shortest path')
        plt.plot(kxplot_FINAL, kyplot_FINAL, '-', color='magenta', label='Final')
        plt.plot(bezxplot, bezyplot, '-', color='red', label='ECHO')
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.)
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
        # speed = self.get_curvature_of_final_path()
        plt.show()


start_time = time.time()
t = Track(26)
t.minimize_objective_functions()
print("program took ", time.time() - start_time, " seconds to run.")
t.plot_points()
t.plot()
