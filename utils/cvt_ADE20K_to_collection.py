import os
import re
from scipy.io import loadmat
import numpy as np
import cv2

from cvt_object_label import cvt_object_label
import get_path


def cvt_ADE20K_to_collection(dataset_folder, dst_folder, dst_label_color):

    # order BGR
    annotations = {
        'canal': 0x00740a,
        'sea': 0x00d850,
        'lake': 0x007132,
        'river': 0x005050,
        'water': 0x00b26e,
        'waterway': 0x00cb6e,
        'stream': 0x002e64,
        'water surf': 0x00c16e
    }

    re_str = ''
    for key, val in enumerate(annotations):
        if key > 0:
            re_str += '|'
        re_str += '# ' + val + ' #'
    re_str = '(' + re_str + ')'
    print(re_str)

    dst_imgs_folder = os.path.join(dst_folder, 'imgs/')
    dst_labels_folder = os.path.join(dst_folder, 'labels/')
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_imgs_folder)
        os.mkdir(dst_labels_folder)
    

    for root, dirs, files in os.walk(dataset_folder):

        files.sort(key = lambda x: (x))

        for name in files:
            
            filename, ext = os.path.splitext(name)
            if ext != '.txt':
                continue
            
            atr_path = os.path.join(root, name)

            file = open(atr_path, 'r')
            atr_data = file.read()
            file.close()
            
            res = re.search(re_str, atr_data)
            if res is None:
                continue

            basename = filename[:-4]

            print(basename)

            ori_img_path = os.path.join(root, basename + '.jpg')
            dst_img_path = os.path.join(dst_imgs_folder, basename + '.png')
            img = cv2.imread(ori_img_path)
            cv2.imwrite(dst_img_path, img)

            ori_label_path = os.path.join(root, basename + '_seg.png')
            dst_label_path = os.path.join(dst_labels_folder, basename + '.png')
            label = cv2.imread(ori_label_path)

            height, width, channels = label.shape
            
            blue_mask = np.zeros([height, width, 1], dtype=np.uint8)
            green_mask = np.ones([height, width, 1], dtype=np.uint8) * 0xff
            red_mask = np.ones([height, width, 1], dtype=np.uint8) * 0xff
            pre_mask = cv2.merge((blue_mask, green_mask, red_mask))
            
            label = cv2.bitwise_and(label, pre_mask)
            
            mask_all = np.zeros([height, width, 3], dtype=np.uint8)
            for key, val in enumerate(annotations):
                ori_label_color = (0, 
                                   (annotations[val] >> 0x8) & 0xff,
                                   annotations[val] & 0xff)
                mask = cvt_object_label(label, ori_label_color, dst_label_color)
                mask_all = cv2.bitwise_or(mask_all, mask)

            cv2.imwrite(dst_label_path, mask_all)


            

        

if __name__ == '__main__':

    dataset_folder = '/Ship01/Dataset/ADE20K'
    dst_folder = os.path.join(get_path.dataset(), 'ADE20K')
    dst_label_color = (200, 0, 0)

    cvt_ADE20K_to_collection(dataset_folder, dst_folder, dst_label_color)