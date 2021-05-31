import os

import numpy as np
import cv2
# import sintel_io as sint
from data.Q3.useful_python_code import sintel_io as sint
from PIL import Image as im

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

    # return np.array([P@img1_3D[x,y] for x in range(img_3D.shape[0]) for y in range(img_3D.shape[1]) ])
    # result = P @ img1_3D
    result = np.zeros((origin_img.shape[0], origin_img.shape[1], 3))
    for x in range(img_3D.shape[0]):
        for y in range(img_3D.shape[1]):
            result[x, y] = P @ img1_3D[x, y]
            result[x, y, :2] = result[x, y,:2]/result[x, y, 2]
            # TODO: depth check 
    return result[:, :, :2]



if __name__ == '__main__':
    depth = sint.depth_read(depth1)
    # img = im.fromarray(depth)
    # print(depth.flatten())
    # M is the intrinsic matrix, N is the extrinsic matrix, so that
    M, _ = sint.cam_read(cam1)
    N = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))
    M_inv = np.linalg.inv(M)
    # reading image
    img_orig = cv2.imread(path1)

    h, w, _ = img_orig.shape

    img1_3D = reproject_to_3D(img_orig, M, depth)
    N = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))
    P = M@N

    project_3D_to_2D(img_orig, img1_3D, P)




    # print(img1)
    # cv2.imshow('BGR Image', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img.show()
