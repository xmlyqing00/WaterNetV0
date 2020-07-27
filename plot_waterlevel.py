import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from PIL import Image
from datetime import datetime, timedelta
import argparse
import bisect
from pandas.plotting import register_matplotlib_converters

time_fmt = mdates.DateFormatter('%m/%d/%y')

tick_spacing = 24
ticker_locator = mdates.HourLocator(interval=tick_spacing)
font_size = 18


def plot_cmp(est_csv_path, gt_csv_path):
    est_csv = pd.read_csv(est_csv_path)
    gt_csv = pd.read_csv(gt_csv_path)

    time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0])
    time_arr_est = pd.to_datetime(gt_csv.iloc[:, 2])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    ax.plot(time_arr_est, est_csv.iloc[:, 3] * 0.3048, 'b-', label=f'Segmentation Estimation', lw=2)
    ax.plot(time_arr_gt, gt_csv.iloc[:, 1] * 0.3048, 'r-', label=f'Milam Street Stream Gage', lw=2)
    ax.set_ylabel('Water Elevation (m) (NAVD88)', fontsize=font_size)
    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size)

    water_level_path = os.path.join(args.root, 'cmp_gt.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    ax.plot(time_arr_est, est_csv.iloc[:, 1], 'g-', label='Raw Segmentation', lw=2)
    ax.plot(time_arr_est, est_csv.iloc[:, 2], 'b-', label='Refined Segmentation', lw=2)
    ax.set_ylabel('Ratio of Bridge Pier Height Covered by Buffalo Bayou', fontsize=font_size)
    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size)

    water_level_path = os.path.join(args.root, 'cmp_constraint.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)

    # est_csv['merged_ratio'].plot()
    # est_csv['merged_ratio_smoothed'].plot()
    # plt.ylim(-0.15, 1)
    # plt.ylabel('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Waterlevel')
    parser.add_argument(
        '--root', default='data/', type=str, metavar='PATH', help='Path to data folder (default: data/).')
    args = parser.parse_args()

    gt_csv_path = os.path.join(args.root, 'buffalo_gt.csv')
    est_csv_path = os.path.join(args.root, 'water_level.csv')

    plot_cmp(est_csv_path, gt_csv_path, )
