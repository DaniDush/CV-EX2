import numpy as np
import cv2
from tabulate import tabulate
from utils import calc_errors
from PIL import Image

kernel_sizes = [3, 9, 15]
paths = [['data/Q2/Art/im_left.png', 'data/Q2/Art/im_right.png', 'data/Q2/Art/disp_left.png'],
         ['data/Q2/Dolls/im_left.png', 'data/Q2/Dolls/im_right.png', 'data/Q2/Dolls/disp_left.png'],
         ['data/Q2/Moebius/view1.png', 'data/Q2/Moebius/view5.png', 'data/Q2/Moebius/disp1.png']]

MODE = ['SSD', 'NCC']
SKIP_COST = 0


def NCC(image, temp):
    """ normalized cross-correlation """
    # Faster, will return ncc of template with each block
    return -cv2.matchTemplate(image, temp, cv2.TM_CCORR_NORMED).squeeze() + 1


def SSD(image, temp):
    """ Sum-Square-Differences """
    # # iterate over strip templates and compute ssd
    # ssd = [((temp - image[:, r-half_k:r+half_k+1]) ** 2).sum() for r in range(half_k, img_left.shape[1]-half_k)]
    # Faster,
    return cv2.matchTemplate(image, temp, cv2.TM_SQDIFF).squeeze()


def get_optimal_line(disparity, cost_matrix, row):
    occlusion = SKIP_COST

    # dynamic programming arrays
    full_cost = np.zeros(cost_matrix.shape)
    full_path = np.zeros(cost_matrix.shape)

    # set conditions (use to bound the occlusions)
    for i in range(1, cost_matrix.shape[1]):
        full_cost[i, 0] = occlusion * i
        full_cost[0, i] = occlusion * i
        full_path[i, 0] = 2
        full_path[0, i] = 3

    # Calculate Optimal Path using DP
    for i in range(1, cost_matrix.shape[1]):
        for j in range(1, cost_matrix.shape[1]):
            # Array that stores cost of 3 paths
            costs = np.array([full_cost[i - 1, j - 1] + cost_matrix[i, j], full_cost[i - 1, j] + occlusion,
                              full_cost[i, j - 1] + occlusion])

            full_cost[i, j] = np.min(costs)  # store optimum path's cost
            full_path[i, j] = np.argmin(costs) + 1  # store optimum path

    # back-tracking
    p, q = cost_matrix.shape[1] - 1, cost_matrix.shape[1] - 1

    # recover shortest path
    while p * q:  # while p, q are both not 0
        print(p, q)
        if full_path[p, q] == 1:  # match, go diagonally
            disparity[row - half_k, p] = abs(q - p)  # disparity score
            p -= 1
            q -= 1

        elif full_path[p, q] == 2:  # unmatched, go up
            p -= 1

        elif full_path[p, q] == 3:  # unmatched, go left
            q -= 1


if __name__ == '__main__':
    for path in paths:
        for m in MODE:
            for k in kernel_sizes:
                img_left = cv2.imread(path[0])
                img_right = cv2.imread(path[1])
                h, w, _ = img_left.shape
                half_k = int(np.floor(k / 2))
                padded_left = np.pad(array=img_left, pad_width=((half_k, half_k + 1), (half_k, half_k + 1), (0, 0)),
                                     mode='mean')
                padded_right = np.pad(array=img_right, pad_width=((half_k, half_k + 1), (half_k, half_k + 1), (0, 0)),
                                      mode='mean')
                disparity_matrix = np.zeros((h, w))  # where all disparities will be saved

                compare = NCC if MODE == 'NCC' else SSD
                SKIP_COST = 0.1 if MODE == 'NCC' else 30000
                from time import time

                cost_matrix = np.zeros((w, w))  # will be filled with costs

                # find optimal disparity matrix
                for row in range(half_k, h + half_k):
                    start_time = time()
                    cost_matrix.fill(0)
                    strip = padded_right[row - half_k: row + half_k + 1, :-1, :]

                    # # iterate trough matching rows and calculate their costs
                    for left_col in range(half_k, w + half_k):
                        template = padded_left[row - half_k: row + half_k + 1, left_col - half_k: left_col + half_k + 1]
                        # cost_matrix[left_col-half_k, :] = compare(image=strip, temp=template)
                        costs = compare(image=strip, temp=template)
                        disparity_matrix[row - half_k, left_col - half_k] = abs(left_col - half_k - np.argmin(costs))
                        cost_matrix[left_col - half_k, :] = costs

                    ### for dynamic programming get its optimal line
                    # get_optimal_line(disparity_matrix, cost_matrix, row)

                    # print(f"Time = {time() - start_time}, Row = {row - half_k}")

                # improve result by copy from left pixel
                # for i in range(0, h):
                #     for j in range(1, w):
                #         if disparity_matrix[i, j] == 0:  # if occluded pixel, copy from left pixel
                #             disparity_matrix[i, j] = disparity_matrix[i, j - 1]

                occluded_inds = np.where(disparity_matrix > 130)
                disparity_matrix[occluded_inds] = -1
                filter_s = 2
                # Improve Result by Occlusion Filling
                for i in range(filter_s, h - filter_s - 1):
                    for j in range(filter_s, w - filter_s - 1):
                        if disparity_matrix[i, j] == -1:  # if occulded pixel, copy from left pixel
                            good_vals = disparity_matrix[i - filter_s:i + filter_s + 1, j - filter_s:j + filter_s + 1]
                            good_inds = np.where(good_vals != -1)
                            disparity_matrix[i, j] = np.median(good_vals[good_inds])

                disparity_matrix[disparity_matrix == -1] = 0

                # max_val = np.max(disparity_matrix)
                # min_val = np.min(disparity_matrix)
                # for i in range(disparity_matrix.shape[0]):
                #     for j in range(disparity_matrix.shape[1]):
                #         disparity_matrix[i, j] = int(255 / (max_val - min_val) * (disparity_matrix[i, j] - max_val) + 255)

                img = Image.fromarray(disparity_matrix).convert('L')
                img.show()

                gt = Image.open(path[2]).convert('LA')
                gt_array = np.array(gt)
                file_name = path[0].split('/')[2]
                img_name = file_name + '_' + m + '_kernel' + str(k)
                img.save('Q2_results/' + img_name, "JPEG")
                print(disparity_matrix)

                avg_error, med_error, bad_05, bad_4 = calc_errors(disparity_matrix, gt_array)
                # print distances table
                print(f"Image = {file_name}, Kernel size = {k}, Cost function = {m}")
                print(tabulate([['AvgErr', avg_error], ['MedErr', med_error],
                                ['Bad05', bad_05], ['Bad4', bad_4]], headers=['Error']))
