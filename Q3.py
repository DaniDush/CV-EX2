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

    # result = np.zeros((origin_img.shape[0], origin_img.shape[1], 3))
    # temp will hold rgb values of the original x,y on the mapped X,Y coordinates. in the 4 dimension will hold depth
    temp = np.zeros((origin_img.shape[0], origin_img.shape[1], 4)) - 1
    # TODO problem with 0,0
    for x in range(h):
        for y in range(w):
            new_x, new_y, new_z = P @ img1_3D[x, y]
            # result[x, y] = P @ img1_3D[x, y]
            # result[x, y, :2] = result[x, y, :2] / result[x, y, 2]
            new_x = int(np.round(new_x/new_z))
            new_y = int(np.round(new_y/new_z))
            # if we did not assign a value yet.
            if temp[new_x, new_y, 0] == -1:
                temp[new_x, new_y, :3], temp[new_x, new_y, 3] = origin_img[new_x, new_y, :], depth[new_x, new_y]
            else:
                # TODO is that that meaning of closer to the camera?
                if temp[new_x, new_y, 1] > depth[new_x, new_y]:
                    temp[new_x, new_y, :3], temp[new_x, new_y, 3] = origin_img[new_x, new_y, :], depth[new_x, new_y]
            pass

    print("Number of un-assigned indices: ", np.where(temp == -1)[0].shape)
    temp[temp == -1] = 0
    print("finish projecting to 2D")
    return temp[:, :, :3]


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

    new_image = project_3D_to_2D(img_orig, img1_3D, P)
    # print(img1)
    cv2.imshow('BGR Image', np.uint8(new_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img.show()
