# Yongqing Liang
# root # lyq.me
# Created at: 2018-09-16

import os
import numpy as np
import cv2
import argparse

from cvt_object_label import cvt_object_label

def get_img_list(imgs_folder):

    tmp_list = os.listdir(imgs_folder)

    img_list = []
    for i in range(len(tmp_list)):
        
        ori_img_folder = os.path.join(imgs_folder, tmp_list[i])
        if os.path.isdir(ori_img_folder):
            img_list.append(tmp_list[i])

    img_list.sort(key=len)

    return img_list


def cvt_labelme_prev(video_folder, dst_folder, ori_label_color, dst_label_color):

    if dst_label_color is None:
        dst_label_color = ori_label_color

    # Get img list
    img_list = get_img_list(video_folder)

    print("Convert", video_folder)
    print("to", dst_folder)
    print(len(img_list), "frames.")

    # Create dst folders
    dst_imgs_folder = os.path.join(dst_folder, 'imgs/')
    dst_labels_folder = os.path.join(dst_folder, 'labels/')
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_imgs_folder)
        os.mkdir(dst_labels_folder)
    
    # Loop each img
    for i in range(len(img_list)):

        ori_imgs_folder = os.path.join(video_folder, img_list[i])

        # Write img
        ori_img_path = os.path.join(ori_imgs_folder, 'img.png')
        dst_img_path = os.path.join(dst_imgs_folder, str(i) + '.png')

        ori_img = cv2.imread(ori_img_path)
        scale = 1080 / ori_img.shape[1]
        if (scale < 1):
            ori_img = cv2.resize(ori_img, None, fx=scale, fy=scale)
        cv2.imwrite(dst_img_path, ori_img)

        # Write label
        ori_label_path = os.path.join(ori_imgs_folder, 'label_modified_vis.png')
        dst_label_path = os.path.join(dst_labels_folder, str(i) + '.png')

        ori_label = cv2.imread(ori_label_path)
        dst_label = cvt_object_label(ori_label, ori_label_color, dst_label_color)
        if (scale < 1):
            dst_label = cv2.resize(dst_label, None, fx=scale, fy=scale)
        cv2.imwrite(dst_label_path, dst_label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--videofolder', default=None, type=str, metavar='PATH')
    args = parser.parse_args()
    video_folder = args.videofolder

    # video_folder = '/Ship01/Dataset/flood/FloodMeasurement/Creek_training_data'
<<<<<<< HEAD
    dst_folder = os.path.join(get_path.dataset_path(), 'creek0')
=======
    dst_folder = os.path.join(video_folder, 'creek0')
>>>>>>> 2213840ecf8be1dafbd1ffe8fd1f9da26b59d7ae
    ori_label_color = (255, 0, 188)
    dst_label_color = (255, 255, 255)

    cvt_labelme_prev(video_folder, dst_folder, ori_label_color, dst_label_color)
