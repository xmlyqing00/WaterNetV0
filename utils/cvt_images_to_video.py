import cv2
import numpy as np
import os

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
    
    root_folder = '/Ship01/Dataset/flood/canyu_result/Houston/'
    image_folder = os.path.join(root_folder, 'water_smoothed_results/')
    video_path = os.path.join(root_folder, 'water_smoothed_result.avi')
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    stride = 10

    cvt_images_to_video(image_folder, video_path, fps, fourcc, stride)