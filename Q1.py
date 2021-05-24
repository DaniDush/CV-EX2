import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import fundamental_matrix
from tabulate import tabulate

ex_1_path = os.path.join('data', 'Q1')
# pts_left = []
# pts_right = []
POINTS_LEFT = [[207, 13], [279, 130], [309, 268], [408, 438], [377, 243], [419, 272], [483, 144], [587, 128],
               [610, 185], [727, 63]]

POINTS_RIGHT = [[265, 1], [313, 91], [389, 198], [536, 281], [385, 182], [446, 199], [458, 96], [541, 82], [562, 127],
                [655, 35]]

CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'


def click_event(event, x, y, flags, params):
    """ function to get coordinates by clicking """

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ',', y)
        pts_left.append([x, y])

    # checking for left mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ',', y)
        pts_right.append([x, y])


def draw_lines(img1, lines, pts1, colors=None):
    """ function to draw the epipolar lines """
    r, c, _ = img1.shape
    c_colors = []
    for idx, (r, pt1) in enumerate(zip(lines, pts1)):
        if colors is None:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            c_colors.append(color)
        else:
            color = colors[idx]

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)

    return img1, c_colors


def get_coordiantes(img):
    """ function to get image adjusted to mouse clicks """

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


def epipolar_distance(l_pts, r_pts, l_lines, r_lines):
    """ function to calculate the epipolar distance given pair of points and corresponding lines """
    r, c, _ = img_left.shape
    distance = 0

    for l_pt, r_pt, l_line, r_line in zip(l_pts, r_pts, l_lines, r_lines):
        left = np.dot((l_pt[0], l_pt[1], 1), l_line) ** 2 / np.sqrt(l_line[0] ** 2 + l_line[1] ** 2)
        right = np.dot((r_pt[0], r_pt[1], 1), r_line) ** 2 / np.sqrt(r_line[0] ** 2 + r_line[1] ** 2)
        distance += (left + right)

    distance /= len(l_pts)  # TODO maybe average by 2*len
    print(f"\n  Epipolar distance: {distance}")
    return distance


def algebraic_distance(r_pts, l_lines):
    """ function to calculate the algebraic distance given a point and corresponding line """
    distance = 0
    for lp, rl in zip(r_pts, l_lines):
        distance += abs(np.dot((lp[0], lp[1], 1), rl))

    distance /= len(r_pts)
    print(f"\n  Algebraic distance: {distance}\n")
    return distance


if __name__ == '__main__':
    to_normalize = [[True, f'{CGREEN}\nEstimate using normalized 8 point:{CEND}'],
                    [False, F'{CGREEN}\nEstimate using regular 8 point:{CEND}']]
    # SED-norm, Alg-norm, SED-regular, Alg-regular
    distances = []
    for n in to_normalize:

        img_left = cv2.imread(os.path.join(ex_1_path, 'im_courtroom_00086_left.jpg'))
        img_right = cv2.imread(os.path.join(ex_1_path, 'im_courtroom_00089_right.jpg'))

        if not POINTS_LEFT:
            get_coordiantes(img_left)
            get_coordiantes(img_right)

        pts_left = POINTS_LEFT
        pts_right = POINTS_RIGHT

        print("Points left:\n", pts_left)
        print("Points right:\n", pts_right)
        assert len(pts_left) == len(pts_right)

        print(n[1])
        pts_left = np.asarray(pts_left)
        pts_right = np.asarray(pts_right)

        # Compute fundamental matrix using 8 point (normalized or not)
        F = fundamental_matrix(pts_left, pts_right, normalize=n[0])
        print(F)
        # activate fundamental matrix
        # left lines will be calculated using right image points and plotted on left image, vice versa
        lines_left = []
        lines_right = []
        for pl, pr in zip(pts_left, pts_right):
            lines_right.append(np.dot(F, (pl[0], pl[1], 1)))
            lines_left.append(np.dot(np.transpose(F), (pr[0], pr[1], 1)))  # need transpose

        # draw lines
        img1, colors = draw_lines(img_left, lines_left, pts_left)
        img2, _ = draw_lines(img_right, lines_right, pts_right, colors)

        plt.subplot(121), plt.imshow(img1)
        plt.subplot(122), plt.imshow(img2)
        plt.show()

        # calc distances
        distances.append(epipolar_distance(pts_left, pts_right, lines_left, lines_right))
        distances.append(algebraic_distance(pts_right, lines_right))

    # print distances table
    print(tabulate([['Normalized 8-point', distances[0], distances[1]], ['Regular 8-point', distances[2], distances[3]]]
                   , headers=['Algorithm', 'Symmetric epipolar distance', 'Algebraic distance']))
