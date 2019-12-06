import sys
import cv2
import numpy as np
import os
import argparse

sys.path.append('../')
from utils.cvt_object_label import cvt_object_label

def TimeSeriesSmooth(result_folder,
                     output_folder,
                     label_color,
                     stride=1):
    
    result_list = os.listdir(result_folder)
    if (len(result_list) == 0):
        exit(-1)
    result_list.sort(key = lambda x: (len(x), x))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    history_mask = None
    decay = 0.2
    smooth_thres = 200

    for result_idx in range(0, len(result_list), stride):

        result_path = os.path.join(result_folder, result_list[result_idx])
        result_img = cv2.imread(result_path)

        print("Working on", result_path)

        water_mask = cvt_object_label(result_img, label_color, [255, 255, 255])
        if history_mask is None:
            history_mask = water_mask
        else:
            history_mask = (1 - decay) * history_mask + decay * water_mask

        ret, water_mask_smoothed = cv2.threshold(history_mask, smooth_thres, 255, cv2.THRESH_BINARY)
        
        output_name = str(result_idx) + '.png'
        output_path = os.path.join(output_folder, output_name)        
        cv2.imwrite(output_path, water_mask_smoothed)
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--rootfolder', default=None, type=str, metavar='PATH')
    args = parser.parse_args()
    root_folder = args.rootfolder

    # root_folder = '/Ship01/Dataset/flood/canyu_result/Houston'
    result_folder = os.path.join(root_folder, 'original_results')
    output_folder = os.path.join(root_folder, 'seg_pier_smoothed')
    label_color = [0, 200, 200]
    stride = 1

    TimeSeriesSmooth(result_folder, output_folder, label_color, stride)