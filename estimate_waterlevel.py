#!/usr/bin/env python3
import cv2
import os
import argparse
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def constraint_temporal(result_folder, output_folder, stride=1):
    result_list = os.listdir(result_folder)
    if len(result_list) == 0:
        exit(-1)

    result_list.sort(key=lambda x: (len(x), x))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    history_score = None
    decay = 0.2
    smooth_thres = 230

    for result_idx in trange(0, len(result_list), stride):

        result_path = os.path.join(result_folder, result_list[result_idx])
        water_mask = cv2.imread(result_path)

        binary_mask = np.all(water_mask == 255, axis=2)
        seg_score = np.zeros((2,) + water_mask.shape[:2])
        seg_score[0, np.invert(binary_mask)] = 1
        seg_score[1, binary_mask] = 1

        if history_score is None:
            history_score = seg_score
        else:
            history_score = (1 - decay) * history_score + decay * seg_score

        water_mask_smoothed = np.argmax(history_score, axis=0) * 255
        # ret, water_mask_smoothed = cv2.threshold(history_score, smooth_thres, 255, cv2.THRESH_BINARY)

        output_name = str(result_idx) + '.png'
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, water_mask_smoothed)


def estimate_waterlevel(result_dir, csv_path, ref_obj, stride=1, label_color=(255, 255, 255)):
    result_list = os.listdir(result_dir)
    if len(result_list) == 0:
        exit(-1)
    result_list.sort(key=lambda x: (len(x), x))

    waterlevel_df = pd.DataFrame(index=np.arange(len(result_list)), columns=['pier_height'])

    n = len(result_list)

    for result_idx in trange(0, n, stride):

        result_path = os.path.join(result_dir, result_list[result_idx])
        result_img = cv2.imread(result_path)

        pier_height = ref_obj['ori_height']

        for y in range(ref_obj['anchor_pt'][1], ref_obj['anchor_pt'][1] + ref_obj['ori_height'] + 10):

            pier_count = 0
            for x in range(ref_obj['anchor_pt'][0] - 2, ref_obj['anchor_pt'][0] + 3):
                img_pt = result_img[y][x]
                if img_pt[0] == label_color[0] and img_pt[1] == label_color[1] and img_pt[2] == label_color[2]:
                    pier_count += 1

            if pier_count > 3:
                pier_height = y - ref_obj['anchor_pt'][1]
                break

        # print(pier_height)
        waterlevel_df.loc[result_idx, 'pier_height'] = pier_height

    waterlevel_df['merged_ratio'] = 1 - waterlevel_df['pier_height'] / ref_obj['ori_height']

    waterlevel_df.to_csv(csv_path, index=False)


def constraint_prior(csv_path, key):
    df = pd.read_csv(csv_path)

    # Laplace detect, median filter
    df[key + '_smoothed'] = df[key]

    laplacian_kernel = [1 / 4, 1 / 3, 1 / 2, 1, -25 / 6, 1, 1 / 2, 1 / 3, 1 / 4]
    width = len(laplacian_kernel)
    half_width = int(width / 2)
    n = len(df)

    smooth_flag = True
    smooth_idx = 0
    while smooth_flag and smooth_idx < 10:

        smooth_flag = False
        df['tmp'] = df[key + '_smoothed']
        smooth_idx += 1
        cnt = 0

        pbar = tqdm(range(half_width, n - half_width), desc=f'Iter: {smooth_idx}')
        for idx in pbar:

            laplacian_val = 0
            for i in range(0, width):
                laplacian_val += laplacian_kernel[i] * df.loc[idx + i - half_width, key + '_smoothed']

            if abs(laplacian_val) > 0.1:
                for i in range(idx - half_width, idx + half_width):
                    median_val = df.loc[i - width:i + width + 1, key + '_smoothed'].median()
                    df.loc[i, 'tmp'] = median_val
                smooth_flag = True

                df[key + '_smoothed'] = df['tmp']
                cnt += 1

            pbar.set_postfix(remain=cnt)

    df = df.drop(columns=['tmp'])
    df.to_csv(csv_path, index=False)

    print('Estimate waterlevel done.', csv_path)

    # plt.figure()
    # df[key].plot()
    # df[key + '_smoothed'].plot()
    # plt.show()


def cvt_px2ft(csv_path, key):
    df = pd.read_csv(csv_path)
    df[key] = df['merged_ratio_smoothed'] * 56.69 + 2.72
    df.to_csv(csv_path, index=False)

    # plt.figure()
    # df[key].plot()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--out-dir', default='data/', type=str, metavar='PATH', help='Path to the output dir.')
    parser.add_argument(
        '--anchor-x', type=int, default=506, help='Referece point X. 535'
    )
    parser.add_argument(
        '--anchor-y', type=int, default=144, help='Referece point Y.'
    )
    parser.add_argument(
        '--ori-h', type=int, default=166, help='Original height.'
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    result_folder = os.path.join(out_dir, 'raw')
    output_folder = os.path.join(out_dir, 'time_smoothed')

    stride = 1
    ref_obj = {
        'anchor_pt': [args.anchor_x, args.anchor_y],
        'ori_height': args.ori_h
    }

    constraint_temporal(result_folder, output_folder, stride)

    csv_path = os.path.join(args.out_dir, 'water_level.csv')
    estimate_waterlevel(output_folder, csv_path, ref_obj, stride)
    constraint_prior(csv_path, 'merged_ratio')
    cvt_px2ft(csv_path, 'Elevation (ft)')
