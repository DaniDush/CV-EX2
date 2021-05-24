import numpy as np
import cv2

kernel_size = [3, 9, 15]
path_1 = ['data/Q2/Art/im_left.png', 'data/Q2/Art/im_right.png']


def compare(image, temp):
    """ normalized cross-correlation """
    # Faster
    return cv2.matchTemplate(image, temp, cv2.TM_CCOEFF_NORMED).squeeze()


def SSD(a, b):
    """ Sum-Square-Differences """
    return ((a - b) ** 2).sum()


def get_optimal_line(disparity, cost_matrix, row):
    OCCLUSION = -0.1

    # Array to store Cost and Path for DP
    Cost = np.zeros(cost_matrix.shape)
    Path = np.zeros(cost_matrix.shape)

    # Set Boundary Condition
    for i in range(1, cost_matrix.shape[1]):
        Cost[i, 0] = OCCLUSION * i
        Cost[0, i] = OCCLUSION * i
        Path[i, 0] = 2
        Path[0, i] = 3

    # Calculate Optimal Path using DP [From Lecture 8's pseudo code]
    for i in range(1, cost_matrix.shape[1]):
        for j in range(1, cost_matrix.shape[1]):
            # Array that stores cost of 3 paths
            costs = np.array(
                [Cost[i - 1, j - 1] + cost_matrix[i, j], Cost[i - 1, j] + OCCLUSION, Cost[i, j - 1] + OCCLUSION])

            Cost[i, j] = np.max(costs)  # store optimum path's cost
            Path[i, j] = np.argmax(costs) + 1  # store optimum path

    # Set cursor to point current postion on DSI Path
    p, q = cost_matrix.shape[1] - 1, cost_matrix.shape[
        1] - 1  # starts from lower right corner, p: y-axis val., q: x-axis val.

    # Recover Optimal Path to get Results [From Lecture 8's pseudo code]
    while p * q:  # while p, q are both not '0'

        if Path[p, q] == 1:  # p matches q, go diagonally

            disparity[row, p] = abs(q - p)  # disparity score

            p = p - 1
            q = q - 1

        elif Path[p, q] == 2:  # unmatched, go up

            p = p - 1

        elif Path[p, q] == 3:  # unmatched, go left

            q = q - 1


if __name__ == '__main__':
    img_left = cv2.imread(path_1[0])
    img_right = cv2.imread(path_1[1])
    h, w, _ = img_left.shape
    k = kernel_size[1]
    half_k = int(np.floor(k / 2))
    disparity_matrix = np.zeros((h, w))  # where all disparities will be saved

    from time import time

    # find optimal disparity matrix
    for row in range(half_k, img_left.shape[0] - half_k - 1):
        start_time = time()
        cost_matrix = np.zeros((img_left.shape[1] - k + 1, img_left.shape[1] - k + 1))  # the dynamic programming matrix

        # # init first row and col
        strip = img_left[row - half_k: row + half_k + 1, :]
        template = img_right[row - half_k: row + half_k + 1, 0: half_k + half_k + 1]
        cost_matrix[:, 0] = compare(image=strip, temp=template)

        strip = img_right[row - half_k: row + half_k + 1, :]
        template = img_left[row - half_k: row + half_k + 1, 0: half_k + half_k + 1]
        cost_matrix[0, :] = compare(image=strip, temp=template)

        # # iterate trough matching rows and calculate their costs
        i = 0
        for left_col in range(half_k, img_left.shape[1] - half_k - 1):
            template = img_left[row - half_k: row + half_k + 1, left_col - half_k: left_col + half_k + 1]
            costs = compare(image=strip, temp=template)
            cost_matrix[i] = costs
            i += 1

        get_optimal_line(disparity_matrix, cost_matrix, row)

        print(f"Time = {time() - start_time}, Row={row}")

    # Improve Result by Occlusion Filling
    for i in range(0, h):
        for j in range(1, w):
            if disparity_matrix[i, j] == 0:  # if occluded pixel, copy from left pixel
                disparity_matrix[i, j] = disparity_matrix[i, j - 1]

    from PIL import Image

    img = Image.fromarray(disparity_matrix)
    img.show()
    print(disparity_matrix)
