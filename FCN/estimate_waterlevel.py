import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal

sys.path.append('../')
import utils

waterlevel_df = None

def median_filter(idx, kernel_size=9):
    global waterlevel_df
    half_kernel_size = int(kernel_size / 2)

    median_val = waterlevel_df.loc[idx-half_kernel_size:idx+half_kernel_size+1]['pier_height'].median()
    waterlevel_df.loc[idx]['pier_height_smoothed'] = median_val
    print('idx', idx, 'before', waterlevel_df.loc[idx]['pier_height'], 'after', waterlevel_df.loc[idx]['pier_height_smoothed'])


def estimate_waterlevel(result_folder,
                        output_file,
                        ref_obj,
                        label_color,
                        stride=1):
    global waterlevel_df

    horizontal_width = 10
    horizontal_left = -(int)(horizontal_width / 2)
    horizontal_right = (int)(horizontal_width / 2)

    result_list = os.listdir(result_folder)
    if (len(result_list) == 0):
        exit(-1)
    result_list.sort(key = lambda x: (len(x), x))

    waterlevel_df = pd.DataFrame(index=np.arange(len(result_list)), columns=['pier_height', 'pier_height_smoothed', 'laplacian_val'])

    n = len(result_list)
    # n = 100
    for result_idx in range(0, n, stride):

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

    # Laplace detect, median filter
    waterlevel_df['pier_height_smoothed'] = waterlevel_df['pier_height']

    laplacian_kernel = [-1/6, 8/3, -5, 8/3, -1/6]
    width = len(laplacian_kernel)
    half_width = int(width / 2)
    for result_idx in range(half_width, n - half_width, stride):

        laplacian_val = 0
        for i in range(0, width):
            laplacian_val += laplacian_kernel[i] * waterlevel_df.loc[result_idx+i-half_width]['pier_height']
        waterlevel_df.loc[result_idx]['laplacian_val'] = laplacian_val

        if abs(laplacian_val) > 30:
            median_filter(result_idx)
            median_filter(result_idx - 1)
            median_filter(result_idx + 1)

            print(result_idx, laplacian_val)

    waterlevel_df['merged_ratio'] = 1 - waterlevel_df['pier_height_smoothed'] / ref_obj['ori_height']
    waterlevel_df.to_csv(output_file)

    print('Estimate waterlevel done.', result_folder, output_file)

    plt.figure()
    waterlevel_df['laplacian_val'][half_width:-half_width-1].plot()

    plt.figure()
    waterlevel_df['pier_height_smoothed'].plot()

    plt.show()


if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/flood/canyu_result/Houston/'
    result_folder = os.path.join(root_folder, 'water_smoothed_results')
    output_file = os.path.join(root_folder, 'waterlevel_smoothed2.csv')
    ref_obj = {
        'anchor_pt': [500, 145],
        'ori_height': 166
    }
    label_color = [255, 255, 255]
    stride = 1

    estimate_waterlevel(result_folder, output_file, ref_obj, label_color, stride)