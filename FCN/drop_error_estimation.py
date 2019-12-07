import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def drop_error_estimation(in_csv_path, out_csv_path):
    
    thres = 0.06
    est_df = pd.read_csv(in_csv_path)
    truth_data_idx = 0
    for i in range(1, len(est_df)):
        if abs(est_df.loc[i, 'merged_ratio_smoothed'] - est_df.loc[truth_data_idx, 'merged_ratio_smoothed']) > thres:
            continue

        k = (est_df.loc[i, 'merged_ratio_smoothed'] - est_df.loc[truth_data_idx, 'merged_ratio_smoothed']) / (i - truth_data_idx)
        if i - truth_data_idx > 1:
            print(truth_data_idx, i, k)

        for j in range(1, i - truth_data_idx):
            est_df.loc[truth_data_idx + j, 'merged_ratio_smoothed'] = \
                est_df.loc[truth_data_idx, 'merged_ratio_smoothed'] + k * j

        truth_data_idx = i

    est_df['merged_ratio_smoothed'].plot()
    plt.show()
    
    est_df.to_csv(out_csv_path)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--rootfolder', default=None, type=str, metavar='PATH')
    args = parser.parse_args()
    root_folder = args.rootfolder

    # root_folder = '/Ship01/Dataset/flood/canyu_result/Houston'
    in_csv_path = os.path.join(root_folder, 'waterlevel_pier_smoothed0.csv')
    out_csv_path = os.path.join(root_folder, 'waterlevel_pier_smoothed1.csv')

    drop_error_estimation(in_csv_path, out_csv_path)