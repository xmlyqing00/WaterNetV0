import cv2
import numpy as np
import os
import argparse

def cvt_images_to_video(image_folder,
                        video_path,
                        fps = 30,
                        fourcc = cv2.VideoWriter_fourcc(*'XVID'),
                        stride=1):

    image_list = os.listdir(image_folder)
    if (len(image_list) == 0):
        exit(-1)
    image_list.sort(key = lambda x: (len(x), x))

    first_image_path = os.path.join(image_folder, image_list[0])
    height, width, channels = cv2.imread(first_image_path).shape
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    stride = max(0, int(stride))
    for image_idx in range(0, len(image_list), stride):

        image_path = os.path.join(image_folder, image_list[image_idx])
        image = cv2.imread(image_path)
        video.write(image)
        
        print("Write", image_path)

    video.release()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LSU WaterLevel Estimation')
    parser.add_argument(
        '--rootfolder', default=None, type=str, metavar='PATH')
    args = parser.parse_args()
    root_folder = args.rootfolder

    # root_folder = '/Ship01/Dataset/water/collection/'
    test_name = 'boston_harbor0'
    method = 'RGMP'
    image_folder = os.path.join(root_folder, 'overlays/', method, test_name)
    video_path = os.path.join(root_folder, 'videos/', method, test_name + '.mp4')
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stride = 1

    cvt_images_to_video(image_folder, video_path, fps, fourcc, stride)