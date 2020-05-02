import math

import numpy as np
import matplotlib.pyplot as plt

track_width = 5
stub = np.array([[0.0, 0.0], [track_width, 0], [0, track_width], [track_width, track_width], [0, track_width*2], [track_width, track_width*2]])



def generate_track():
    startt = stub
    r = 8
    originpoint = startt[5]
    polarpoint = [startt[5][0] + r, startt[5][1]]
    alpha = 22.5
    alpha = math.radians(alpha)
    for i in range(1, 5):
        x = polarpoint[0] - r * math.cos(i * alpha)
        y = polarpoint[1] + r * math.sin(i * alpha)
        x1 = polarpoint[0] - (r + track_width) * math.cos(i * alpha)
        y1 = polarpoint[1] + (r + track_width) * math.sin(i * alpha)
        startt = np.append(startt, [[x, y], [x1, y1]], axis=0)
    originpoint = startt[len(startt) - 2]
    for i in range(1, 6):
        startt = np.append(startt,
                           [[originpoint[0] + i * track_width, originpoint[1]], [originpoint[0] + i * track_width, originpoint[1] + track_width]],
                           axis=0)
    r = 3
    polarpoint = [startt[len(startt) - 1][0], startt[len(startt) - 1][1] + r]
    for i in range(1, 9):
        x = polarpoint[0] + r * math.sin(i * alpha)
        y = polarpoint[1] - r * math.cos(i * alpha)
        x1 = polarpoint[0] + (r + track_width) * math.sin(i * alpha)
        y1 = polarpoint[1] - (r + track_width) * math.cos(i * alpha)
        startt = np.append(startt, [[x, y], [x1, y1]], axis=0)
    originpoint = startt[len(startt) - 2]
    for i in range(1, 9):
        startt = np.append(startt,
                           [[originpoint[0] - i * track_width, originpoint[1]], [originpoint[0] - i * track_width, originpoint[1] + track_width]],
                           axis=0)


    xplot = startt[:, 0]
    yplot = startt[:, 1]
    # print(startt)
    # plt.scatter(xplot, yplot)
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
    plt.xlim(overallmin - overallmin * 0.1, overallmax + overallmax * 0.1)
    plt.ylim(overallmin - overallmin * 0.1, overallmax + overallmax * 0.1)
    # print(startt[len(startt)-1][1] - startt[len(startt)-2][1])
    #plt.show()
    return startt

# generate_track()
