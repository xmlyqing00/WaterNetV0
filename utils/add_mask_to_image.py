import cv2
import numpy as np
import os

from cvt_object_label import cvt_object_label

def add_mask_to_image(image_folder, 
                      mask_folder,
                      output_folder, 
                      label_color=(255, 255, 255),
                      stride=1):
    
    alpha = 0.5

    image_list = os.listdir(image_folder)
    mask_list = os.listdir(mask_folder)

    if (len(image_list) == 0):
        exit(-1)
    
    # assert(len(image_list) == len(mask_list))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    image_list.sort(key = lambda x: (len(x), x))
    mask_list.sort(key = lambda x: (len(x), x))

    stride = max(0, int(stride))
    for image_idx in range(0, len(mask_list), stride):
        
        image_path = os.path.join(image_folder, image_list[image_idx])
        image = cv2.imread(image_path)

        mask_path = os.path.join(mask_folder, mask_list[image_idx])
        mask = cv2.imread(mask_path)

        mask = cvt_object_label(mask, [255, 255, 255], label_color)

        image_mask = image.copy()
        cv2.addWeighted(image, alpha, mask, 1 - alpha, 0, image_mask)

        filename, ext = os.path.splitext(image_list[image_idx])
        output_name = filename + '_mask.png'
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, image_mask)

        print("Add mask to image", output_path)


if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/flood/collection/stream_test'
    image_folder = os.path.join(root_folder, 'imgs/')
    mask_folder = os.path.join(root_folder, 'segs/')
    output_folder = os.path.join(root_folder, 'overlays')
    label_color = (200, 0, 0)
    stride = 1

    add_mask_to_image(image_folder, mask_folder,  output_folder, label_color, stride)