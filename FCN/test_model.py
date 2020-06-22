#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import torch
from torch.utils import model_zoo
from torchvision import transforms

sys.path.append('../')
from model import FCNResNet
from utils.dataset import Dataset
from utils.AvgMeter import AverageMeter


def test_FCNResNet():

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch FCNResNet Testing')
    parser.add_argument(
        '-c', '--checkpoint', default='../data/models/checkpoint_58.pth.tar', type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-i', '--imgs-path', default='../data/houston_small', type=str, metavar='PATH',
        help='Path to the test imgs (default: none).')
    parser.add_argument(
        '-o', '--out-path', default='output/', type=str, metavar='PATH',
        help='Path to the output segmentations (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.imgs_path is None:
        raise ValueError('Must input test images path.')
    if args.out_path is None:
        raise ValueError('Must input output images path.')

    water_thres = 5

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
        dataset_path=args.imgs_path,
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
    fcn_resnet = FCNResNet().to(device)

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        fcn_resnet.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Start testing

    fcn_resnet.eval()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for i, input in enumerate(test_loader):

        print(i)

        input = input.to(device)
        output = fcn_resnet(input)

        seg = output.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        ret, seg = cv2.threshold(seg, water_thres, 255, cv2.THRESH_BINARY)

        seg_path = os.path.join(args.out_path, str(i) + '.png')
        cv2.imwrite(seg_path, seg)


if __name__ == '__main__':
    test_FCNResNet()
