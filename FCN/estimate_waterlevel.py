import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal

sys.path.append('../')
from utils.laplacian_smooth import laplacian_smooth

def estimate_waterlevel(result_folder,
                        output_file,
                        ref_obj,
                        label_color,
                        stride=1,
                        constraint_flag=False):

    horizontal_width = 10
    horizontal_left = -(int)(horizontal_width / 2)
    horizontal_right = (int)(horizontal_width / 2)

    result_list = os.listdir(result_folder)
    if (len(result_list) == 0):
        exit(-1)
    result_list.sort(key = lambda x: (len(x), x))

    waterlevel_df = pd.DataFrame(index=np.arange(len(result_list)), columns=['pier_height'])

    n = len(result_list)
    # n = 100
    pier_height = 0

    for result_idx in range(0, n, stride):

        result_path = os.path.join(result_folder, result_list[result_idx])
        result_img = cv2.imread(result_path)

        print("Working on", result_path)

        height, width, channels = result_img.shape
        if not constraint_flag:
            pier_height = 0
            
        for y in range(420, ref_obj['anchor_pt'][1], -1):

            pier_count = 0
            for x in range(490, 540):
                img_pt = result_img[y][x]
                if img_pt[0] == label_color[0] and img_pt[1] == label_color[1] and img_pt[2] == label_color[2]:
                    pier_count += 1

            # print(y, pier_count)
            
            if pier_count > 5:
                pier_height = y - ref_obj['anchor_pt'][1]
                break
        
        waterlevel_df.loc[result_idx, 'pier_height'] = pier_height

    waterlevel_df['merged_ratio'] = 1 - waterlevel_df['pier_height'] / ref_obj['ori_height']
    waterlevel_df.to_csv(output_file, index=False)

    if constraint_flag:
        laplacian_smooth(output_file, 'merged_ratio', output_file)


if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/flood/canyu_result/Houston/'
    result_folder = os.path.join(root_folder, 'original_results')
    output_file = os.path.join(root_folder, 'waterlevel_pier_original.csv')
    ref_obj = {
        'anchor_pt': [506, 144],
        'ori_height': 166
    }
    # label_color = [255, 255, 255]
    label_color = [0, 200, 200]
    stride = 1
    constraint_flag = False

    estimate_waterlevel(result_folder, output_file, ref_obj, label_color, stride, constraint_flag)