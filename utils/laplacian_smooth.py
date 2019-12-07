import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def median_filter(df, key, idx, kernel_size=25):
    half_kernel_size = int(kernel_size / 2)
    median_val = df.loc[idx-half_kernel_size:idx+half_kernel_size+1, 'tmp'].median()
    df.loc[idx, key + '_smoothed'] = median_val
    print('idx', idx, 'before', df.loc[idx, 'tmp'], 'after', df.loc[idx, key + '_smoothed'])

def laplacian_smooth(csv_path, key, output_file):
    
    df = pd.read_csv(csv_path)

    # Laplace detect, median filter
    df[key + '_smoothed'] = df[key]
    df['laplacian_val'] = 0

    laplacian_kernel = [-1/6, 8/3, -5, 8/3, -1/6]
    width = len(laplacian_kernel)
    half_width = int(width / 2)
    n = len(df)

    smooth_flag = True
    cnt = 0
    while smooth_flag:
        
        df['tmp'] = df[key + '_smoothed']
        smooth_flag = False
        cnt += 1

        for idx in range(half_width, n - half_width):

            laplacian_val = 0
            for i in range(0, width):
                laplacian_val += laplacian_kernel[i] * df.loc[idx+i-half_width, 'tmp']
            if (cnt == 1):
                df.loc[idx, 'laplacian_val'] = laplacian_val

            if abs(laplacian_val) > 0.15:
                for i in range(-2, 3):
                    median_filter(df, key, max(min(idx + i, n-1), 0) )

                smooth_flag = True

    plt.figure()
    df[key].plot
    
    plt.figure()
    df['laplacian_val'][half_width:-half_width-1].plot()

    plt.figure()
    df[key + '_smoothed'].plot()

    df = df.drop(columns=['tmp'])
    df.to_csv(output_file, index=False)

    print('Estimate waterlevel done.', output_file)

    plt.figure()
    df['laplacian_val'][half_width:-half_width-1].plot()

    plt.figure()
    df[key + '_smoothed'].plot()

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--rootfolder', default=None, type=str, metavar='PATH')
    args = parser.parse_args()
    root_folder = args.rootfolder

    # root_folder = '/Ship01/Dataset/flood/canyu_result/Houston/'
    csv_file = os.path.join(root_folder, 'waterlevel_smoothed3.csv')
    output_file = csv_file

    laplacian_smooth(csv_file, 'merged_ratio', output_file)