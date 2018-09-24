import sys
import cv2
import numpy as np
import pandas as pd
import os

sys.path.append('../')
import utils

def estimate_waterlevel(result_folder,
                        output_file,
                        ref_obj,
                        label_color,
                        stride=1):
    
    horizontal_width = 10
    horizontal_left = -(int)(horizontal_width / 2)
    horizontal_right = (int)(horizontal_width / 2)

    result_list = os.listdir(result_folder)
    if (len(result_list) == 0):
        exit(-1)
    result_list.sort(key = lambda x: (len(x), x))

    waterlevel_df = pd.DataFrame(index=np.arange(len(result_list)), columns=['pier_height'])

    for result_idx in range(0, len(result_list), stride):

        result_path = os.path.join(result_folder, result_list[result_idx])
        result_img = cv2.imread(result_path)

        print("Working on", result_path)

        height, width, channels = result_img.shape

        for y in range(ref_obj['anchor_pt'][1], height):

            water_count = 0
            for x in range(horizontal_left, horizontal_right):
                img_pt = result_img[y][x + ref_obj['anchor_pt'][0]]
                if img_pt[0] == label_color[0] and img_pt[1] == label_color[1] and img_pt[2] == label_color[2]:
                    water_count = water_count + 1
            
            if water_count > horizontal_width / 2:
                pier_height = y - ref_obj['anchor_pt'][1]
                waterlevel_df.loc[result_idx] = [pier_height]
                break

    waterlevel_df['merged_ratio'] = 1 - waterlevel_df['pier_height'] / ref_obj['ori_height']
    waterlevel_df.to_csv(output_file)

    print('Estimate waterlevel done.', result_folder, output_file)

if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/flood/canyu_result/Houston/'
    result_folder = os.path.join(root_folder, 'water_smoothed_results')
    output_file = os.path.join(root_folder, 'waterlevel_smoothed1.csv')
    ref_obj = {
        'anchor_pt': [500, 145],
        'ori_height': 166
    }
    label_color = [255, 255, 255]
    stride = 1

    estimate_waterlevel(result_folder, output_file, ref_obj, label_color, stride)