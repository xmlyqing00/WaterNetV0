#!/usr/bin/env python3
import argparse
import os

import cv2
import torch
from tqdm import tqdm
from torchvision import transforms

from FCN.model import WaterNetV0
from FCN.dataset import Dataset


def test_WaterNetV0():

    # Hyper parameters
    parser = argparse.ArgumentParser(description='WaterNetV0 Testing')
    parser.add_argument(
        '-c', '--checkpoint', default='cp/checkpoint_58.pth.tar', type=str, metavar='PATH',
        help='Path to latest checkpoint (default: cp/checkpoint_58.pth.tar).')
    parser.add_argument(
        '-i', '--imgs-path', default='/Ship01/Dataset/water_v1', type=str, metavar='PATH',
        help='Path to the test imgs (default: /Ship01/Dataset/water_v1/).')
    parser.add_argument(
        '-o', '--out-path', default='data/raw', type=str, metavar='PATH',
        help='Path to the output segmentations (default: data/raw/).')
    parser.add_argument(
        '--name', default='houston', type=str,
        help='Test video name (default: houston).')
    args = parser.parse_args()

    print('Args:', args)

    water_thres = 128

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': 4,
            'pin_memory': True
        }

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    dataset = Dataset(
        mode='test',
        root=args.imgs_path,
        video_name=args.name,
        input_transforms=transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

    # Model
    waternetv0 = WaterNetV0().to(device)

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        waternetv0.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Start testing

    waternetv0.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    for i, input in enumerate(tqdm(test_loader)):

        input = input.to(device)
        output = waternetv0(input)

        seg = output.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        ret, seg = cv2.threshold(seg, water_thres, 255, cv2.THRESH_BINARY)

        seg_path = os.path.join(args.out_path, str(i) + '.png')
        cv2.imwrite(seg_path, seg)


if __name__ == '__main__':
    test_WaterNetV0()
