import os
import cv2

def format_file_names(folder):
    
    file_list = os.listdir(folder)
    idx = 0
    file_list.sort(key = lambda x: (len(x), x))

    for file_name in file_list:
        file_path0 = os.path.join(folder, file_name)
        file_path1 = os.path.join(folder, str(idx) + '.png')
        img = cv2.imread(file_path0)
        cv2.imwrite(file_path1, img)
        # os.rename(file_path0, file_path1)
        idx += 1


if __name__ == '__main__':
    root = '/Ship01/Dataset/water/collection/imgs/'
    folder = os.path.join(root, 'boston_harbor1')
    format_file_names(folder)