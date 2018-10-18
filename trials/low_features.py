import sys
import cv2
import os
from matplotlib import pyplot as plt

sys.path.append('../')

def trial_gradient(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = abs(cv2.Laplacian(gray_img, cv2.CV_32F))
    sobelx = abs(cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5))
    sobely = abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5))

    cv2.imshow("img", img) 
    cv2.imshow("gray", gray_img)
    cv2.imshow("laplacian", laplacian)
    cv2.imshow("sobelx", sobelx)
    cv2.imshow("sobely", sobely)
    cv2.waitKey()

if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/flood/collection/'
    img_path = os.path.join(root_folder, 'stream_test/imgs/original_resize_14.png')
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

    trial_gradient(img)

