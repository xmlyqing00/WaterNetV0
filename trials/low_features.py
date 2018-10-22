import sys
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../')

def trial_gradient(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows, cols = gray_img.shape
    crow,ccol = (int)(rows/2) , (int)(cols/2)
    w = 50
    fshift[crow-w:crow+w, ccol-w:ccol+w] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()

    # plt.subplot(121),plt.imshow(gray_img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

if __name__ == '__main__':

    # root_folder = '/Ship01/Dataset/flood/collection/'
    # img_path = os.path.join(root_folder, 'stream_test/imgs/original_resize_14.png')
    img_path = 'original_resize_50.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

    trial_gradient(img)

