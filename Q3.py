import os

import numpy as np
import cv2
#import sintel_io as sint
from data.Q3.useful_python_code import sintel_io as sint
from PIL import Image as im

cam1 = 'data/Q3/alley_2.cam'
depth1 = 'data/Q3/alley_2.dpt'
path1 = 'data/Q3/alley_2.png'

# reference:
#
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/
# box_dimensioner_multicam/helper_functions.py
def convert_img_to_3D(img,K,depth):

    result = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel = (x,y)
            X,Y,Z = img_plane_pixel_to_3D(K, pixel, depth[x][y] )
            print(x,y,depth[x][y])
            print(X,Y,Z)
            result.append([X, Y, Z])





def img_plane_pixel_to_3D(K, pixel, depth):
    # X,Y,X - point in 3D
    # pixel - in 2D - float[2]
    # depth - depth value in coresponding pixel

    f_x = K[0][0]
    f_y = K[1][1]
    c_x = K[0][2]
    c_y = K[1][2]
    x_3D = (pixel[0] - c_x) / f_x * depth
    y_3D = (pixel[1] - c_y) / f_y * depth
    z_3D = depth

    return x_3D,y_3D,z_3D




###########################################################################


def reproject_to_3D(img, M, depth):

    M_inv = np.linalg.inv(M)
    result = np.zeros((img.shape[0], img.shape[1], 3))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            X, Y, _ = M_inv@np.array([x,y,1])
            Z = depth[x][y]
            result[x][y] = np.array([X,Y,Z])

    return result

def project_point_to_2D(img_3D, M):


    for x in range(img_3D.shape[0]):
        for y in range(img_3D.shape[1]):
                return M@img_3D[x][y]



if __name__ == '__main__':


    depth = sint.depth_read(depth1)
    # img = im.fromarray(depth)

    # M is the intrinsic matrix, N is the extrinsic matrix, so that
    M,N = sint.cam_read(cam1)
    M_inv = np.linalg.inv(M)
    # reading image
    img1_2D = cv2.imread(path1)

    # cam_coords = M_inv @ img1_2D * depth.flatten()



    # converting img to 3D using x = PX and depth map

    pix1 = img_plane_pixel_to_3D(M, [121,253], depth[121,253])
    print(M@pix1/pix1[2])

    # X, Y, _ = M_inv @ np.array([121, 253, 1])
    # Z = depth[121][253]
    #
    # print(X,Y,Z)
    # print(M@[X,Y,Z]/Z)

    # img1_3D = reproject_to_3D(img1_2D, M, depth)
    # print(img1_3D)
    # print(project_point_to_2D(img1_3D, M))


    # print(img1)
    # cv2.imshow('BGR Image', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #img.show()






