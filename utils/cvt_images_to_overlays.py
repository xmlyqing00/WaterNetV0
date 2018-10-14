import cv2
import numpy as np
import os

from cvt_object_label import cvt_object_label

def add_mask_to_image(image, mask, label_color):
    
    roi = cvt_object_label(mask, label_color, [255, 255, 255])
    mask = cvt_object_label(mask, label_color, [0, 255, 255])
    
    # cv2.imshow("roi", roi)
    complement = cv2.bitwise_not(roi)
    complement_img = cv2.bitwise_and(complement, image)
    # cv2.imshow("com", complement_img)
    # cv2.imshow("img", image)
    mask = mask + complement_img
    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    alpha = 0.8
    image_mask = image.copy()
    cv2.addWeighted(image, alpha, mask, 1 - alpha, 0, image_mask)

    return image_mask

def cvt_images_to_overlays(image_folder, 
                           mask_folder,
                           output_folder, 
                           label_color=(255, 255, 255),
                           stride=1):

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

        image_mask = add_mask_to_image(image, mask, [255, 255, 255])

        filename, ext = os.path.splitext(image_list[image_idx])
        output_name = filename + '_mask.png'
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, image_mask)

        print("Add mask to image", output_path)

def run_cvt_images_to_overlays():
    root_folder = '/Ship01/Dataset/flood/collection/stream_test'
    image_folder = os.path.join(root_folder, 'imgs/')
    mask_folder = os.path.join(root_folder, 'segs/')
    output_folder = os.path.join(root_folder, 'overlays')
    label_color = (200, 0, 0)
    stride = 1

    cvt_images_to_overlays(image_folder, mask_folder,  output_folder, label_color, stride)

def run_add_mask_to_image():

    root_folder = '/Ship01/Documents/MyPapers/FloodHydrograph/imgs'
    image = cv2.imread(os.path.join(root_folder, '1798_original.png'))
    mask = cv2.imread(os.path.join(root_folder, '1798_before_smoothed.png'))
    image_mask = add_mask_to_image(image, mask, [200, 0, 0])
    cv2.imwrite(os.path.join(root_folder, '1798_before_smoothed_overlay.png'), image_mask)


if __name__ == '__main__':
    
    # run_cvt_images_to_overlays()
    run_add_mask_to_image()