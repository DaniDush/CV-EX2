import os

import numpy as np
import cv2
# import sintel_io as sint
from data.Q3.useful_python_code import sintel_io as sint
from PIL import Image as im, Image

cam1 = 'data/Q3/alley_2.cam'
depth1 = 'data/Q3/alley_2.dpt'
path1 = 'data/Q3/alley_2.png'


# reference:
#
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/
# box_dimensioner_multicam/helper_functions.py

# https://towardsdatascience.com/inverse-projection-transformation-c866ccedef1c


def reproject_to_3D(img, K, depth):
    result = np.zeros((img.shape[0], img.shape[1], 4))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            X, Y, Z = pixel_to_3D(K, [x, y], depth[x, y])
            result[x, y] = [X, Y, Z, 1]

    return result


def pixel_to_3D(K, pixel, depth):
    # X,Y,X - point in 3D
    # pixel - in 2D - float[2]
    # depth - depth value in coresponding pixel

    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
    x_3D = (pixel[0] - c_x) / f_x * depth
    y_3D = (pixel[1] - c_y) / f_y * depth
    return x_3D, y_3D, depth


def project_3D_to_2D(origin_img, img_3D, P):
    result = np.zeros((h, w, 4))
    result[:, :, 3] = 1000

    # too long!!!
    # TODO problem with 0,0
    for x in range(h):
        for y in range(w):
            new_x, new_y, new_z = P @ img1_3D[x, y]
            new_x = int(np.round(new_x / new_z))
            new_y = int(np.round(new_y / new_z))
            # print(new_x, new_y)
            if new_x >= h or new_y >= w or new_x <= -1 or new_y <= -1:
                continue

            if img1_3D[x, y, 2] < result[new_x, new_y, 3]:
                result[new_x, new_y, :3], result[new_x, new_y, 3] = origin_img[x, y, :], img1_3D[x, y, 2]

    print("finish projecting to 2D")
    return result[:, :, :3]


# TODO - convert theta to radians
def rotate_X_axis(theta):
    theta = np.deg2rad(theta)
    return np.array(([1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]))


def rotate_Y_axis(theta):
    theta = np.deg2rad(theta)
    return np.array(([np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]))


def rotate_Z_axis(theta):
    theta = np.deg2rad(theta)
    return np.array(([np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]))


if __name__ == '__main__':
    depth = sint.depth_read(depth1)
    M, _ = sint.cam_read(cam1)

    M_inv = np.linalg.inv(M)
    # reading image
    img_orig = cv2.imread(path1)

    h, w, _ = img_orig.shape

    img1_3D = reproject_to_3D(img_orig, M, depth)

    N = np.eye(3, 4)
    # N[0, 3] = 0.05
    for i in range(-10, 10):
        N[:3, :3] = rotate_Y_axis(i/5)
        P = M @ N

        new_image = np.uint8(project_3D_to_2D(img_orig, img1_3D, P))
        img = Image.fromarray(new_image).convert('RGB')
        img.save('Q3_results/' + str(i) + '.jpeg', "JPEG")

        cv2.imshow('BGR Image', new_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
