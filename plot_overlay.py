import cv2
import numpy as np
import os
import argparse
from glob import glob
from tqdm import trange
from scipy.ndimage.morphology import binary_dilation


def add_overlay(img, mask, colors, alpha=0.7, cscale=1):
    img_overlay = img.copy()
    ones_np = np.ones(img.shape) * (1 - alpha)

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    canvas = img * alpha + ones_np * np.array(colors[0])[::-1]

    binary_mask = np.all(mask == 255, axis=2)
    img_overlay[binary_mask] = canvas[binary_mask]

    contour = binary_dilation(binary_mask) ^ binary_mask
    img_overlay[contour, :] = 0

    return img_overlay


def run_plot_overlays(img_dir, seg_dir, out_dir):
    img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
    seg_list = sorted(glob(os.path.join(seg_dir, '*.png')))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in trange(0, len(img_list), 2):
        img = cv2.imread(img_list[i])
        seg = cv2.imread(seg_list[i])

        overlay = add_overlay(img, seg, (255, 0, 0))  # RGB order

        out_name = os.path.basename(img_list[i])[:-4] + '.png'
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, overlay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--img-dir', default='/Ship01/Dataset/water_v1/JPEGImages/houston', type=str, metavar='PATH',
        help='Path to the input image folder.'
    )
    parser.add_argument(
        '--seg-dir', default='data/raw/', type=str, metavar='PATH',
        help='Path to the segmentation folder.'
    )
    parser.add_argument(
        '--out-dir', default='data/overlay_raw/', type=str, metavar='PATH',
        help='Path to the output overlay folder.'
    )
    args = parser.parse_args()

    run_plot_overlays(args.img_dir, args.seg_dir, args.out_dir)
