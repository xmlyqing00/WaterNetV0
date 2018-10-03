import os
import sys
from PIL import Image
from torch.utils import data

sys.path.append('../')
from utils.add_prefix import add_prefix

class Dataset(data.Dataset):

    def __init__(self, 
                 mode,
                 dataset_path, 
                 input_transforms=None, 
                 target_transforms=None):
        
        self.mode = mode
        if mode != 'test':
            self.target_list = []
            self.target_transforms = target_transforms

        self.input_list = []
        self.input_transforms = input_transforms
        
        if mode != 'test':

            for sub_folder in os.listdir(dataset_path):
                
                sub_folder_path = os.path.join(dataset_path, sub_folder)

                imgs_path = os.path.join(sub_folder_path, 'imgs/')
                imgs_list = os.listdir(imgs_path)
                imgs_list.sort(key = lambda x: (len(x), x))
                self.input_list += [add_prefix(imgs_path, name) for name in imgs_list]
            
                labels_path = os.path.join(sub_folder_path, 'labels/')
                labels_list = os.listdir(labels_path)
                labels_list.sort(key = lambda x: (len(x), x))
                self.target_list += [add_prefix(labels_path, name) for name in labels_list]

        else:

            imgs_list = os.listdir(dataset_path)
            imgs_list.sort(key = lambda x: (len(x), x))
            self.input_list += [add_prefix(dataset_path, name) for name in imgs_list]

    def __getitem__(self, index):
        
        input = Image.open(self.input_list[index])

        if self.input_transforms is not None:
            input = self.input_transforms(input)

        if self.mode != 'test':
            target = Image.open(self.target_list[index])
            target = target.convert('L')

            if self.target_transforms is not None:
                target = self.target_transforms(target)
            
            return input, target

        return input


    def __len__(self):
        return len(self.input_list)