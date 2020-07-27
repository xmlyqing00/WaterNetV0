#!/usr/bin/env python3
import os
import argparse
import torch
from tqdm import tqdm
from torch.utils import model_zoo
from torchvision import transforms

from FCN.model import WaterNetV0
from FCN.dataset import Dataset
from utils.AvgMeter import AverageMeter


def adjust_learning_rate(optimizer, start_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_WaterNetV0():
    # Hyper parameters
    parser = argparse.ArgumentParser(description='WaterNetV0 Training')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N',
        help='Manual epoch number (useful on restarts, default 0).')
    parser.add_argument(
        '--total-epochs', default=100, type=int, metavar='N',
        help='Number of total epochs to run (default 100).')
    parser.add_argument(
        '--lr', '--learning-rate', default=0.05, type=float,
        metavar='LR', help='Initial learning rate.')
    parser.add_argument(
        '--resume', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '--dataset', default='/Ship01/Dataset/water_v1', metavar='PATH',
        help='Path to the training dataset')
    parser.add_argument(
        '--modelpath', default='cp/', metavar='PATH',
        help='Path to the models.')

    args = parser.parse_args()

    print('Args:', args)

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
        mode='train',
        root=args.dataset,
        input_transforms=transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ]),
        target_transforms=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        **dataset_args
    )

    # Model
    waternetv0 = WaterNetV0().to(device)

    # Criterion and Optimizor
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.SGD(
        params=waternetv0.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    start_epoch = args.start_epoch

    # Load pretrained model
    if args.resume:
        if os.path.isfile(args.resume):
            print('Load checkpoint \'{}\''.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            waternetv0.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('No checkpoint found at \'{}\''.format(args.resume))
    else:
        print('Load pretrained ResNet 34.')
        resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        pretrained_model = model_zoo.load_url(resnet34_url)
        waternetv0.load_pretrained_model(pretrained_model)

    # Start training
    waternetv0.train()
    if not os.path.exists(args.modelpath):
        os.mkdir(args.modelpath)

    for epoch in range(start_epoch, args.total_epochs):

        losses = AverageMeter()
        adjust_learning_rate(optimizer, args.lr, epoch)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (input, target) in enumerate(pbar):
            input, target = input.to(device), target.to(device)

            output = waternetv0(input)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            pbar.set_postfix(loss=f'{losses.avg:.6f}')

        if (epoch + 1) % 20 == 0:
            model_path = os.path.join(args.modelpath, 'checkpoint_{0}.pth.tar'.format(epoch))
            torch.save(
                obj={
                    'epoch': epoch,
                    'model': waternetv0.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': losses.avg,
                },
                f=model_path
            )
            print(f'Model saved in {model_path}.')


if __name__ == '__main__':
    train_WaterNetV0()
