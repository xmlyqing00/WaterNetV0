import os
from glob import glob
from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self,
                 mode,
                 root,
                 video_name=None,
                 input_transforms=None,
                 target_transforms=None):

        self.mode = mode
        if mode != 'test':
            self.mask_list = []
            self.target_transforms = target_transforms

        self.img_list = []
        self.input_transforms = input_transforms

        if mode != 'test':

            with open(os.path.join(root, 'train.txt'), 'r') as lines:
                for line in lines:
                    dataset_name = line.strip()
                    if len(dataset_name) == 0:
                        continue

                    img_dir = os.path.join(root, 'JPEGImages', dataset_name)
                    mask_dir = os.path.join(root, 'Annotations', dataset_name)

                    img_list = sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png')))
                    mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

                    self.img_list += img_list
                    self.mask_list += mask_list

        else:

            assert video_name
            self.img_list = sorted(glob(os.path.join(root, 'JPEGImages', video_name, '*.jpg')))

    def __getitem__(self, index):

        input = Image.open(self.img_list[index])

        if self.input_transforms is not None:
            input = self.input_transforms(input)

        if self.mode != 'test':
            target = Image.open(self.mask_list[index])
            target = target.convert('L')

            if self.target_transforms is not None:
                target = self.target_transforms(target)

            return input, target

        return input

    def __len__(self):
        return len(self.img_list)
