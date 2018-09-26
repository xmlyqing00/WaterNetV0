import os
import sys
from PIL import Image
from torch.utils import data

sys.path.append('../')
from utils.add_prefix import add_prefix

class Dataset(data.Dataset):

    def __init__(self, 
                 dataset_path, 
                 input_transforms=None, 
                 target_transforms=None):
        
        self.input_list = []
        self.target_list = []

        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        
        for sub_folder in os.listdir(dataset_path):
            
            sub_folder_path = os.path.join(dataset_path, sub_folder)
            imgs_path = os.path.join(sub_folder_path, 'imgs/')
            labels_path = os.path.join(sub_folder_path, 'labels/')

            imgs_list = os.listdir(imgs_path)
            labels_list = os.listdir(labels_path)
            assert(len(imgs_list) == len(labels_list))

            imgs_list.sort(key = lambda x: (len(x), x))
            labels_list.sort(key = lambda x: (len(x), x))

            self.input_list += [add_prefix(imgs_path, name) for name in imgs_list]
            self.target_list += [add_prefix(labels_path, name) for name in labels_list]


    def __getitem__(self, index):
        
        input = Image.open(self.input_list[index])
        target = Image.open(self.target_list[index])
        target = target.convert('L')
        
        if self.input_transforms is not None:
            input = self.input_transforms(input)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return input, target

    def __len__(self):
        return len(self.input_list)