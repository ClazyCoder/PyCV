import numpy as np

def convolution(img, kernel):
    res = img.copy()
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, 'Wrong Kernel Size!'
    pass

def rotate(img, angle):
    pass

def canny(img, t1, t2):
    pass

def draw_line(img, x1, y1, x2, y2):
    pass

def calc_homography(points1, points2):
    pass

def calibrate_camera(img_points, obj_points):
    pass