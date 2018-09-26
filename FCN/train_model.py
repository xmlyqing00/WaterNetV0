import os
import argparse
import sys
import time
import torch
from torch.utils import model_zoo
from torchvision import transforms

sys.path.append('../')
from model import FCNResNet
from utils.dataset import Dataset
from utils.get_path import dataset_path, saved_models_path
from utils.AvgMeter import AverageMeter


def adjust_learning_rate(optimizer, start_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_FCNResNet():
    
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch FCNResNet Training')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts, default 0)')
    parser.add_argument(
        '--total-epochs', default=100, type=int, metavar='N',
        help='number of total epochs to run (default 100)')
    parser.add_argument(
        '--lr', '--learning-rate', default=0.05, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument(
        '--resume', default=None, type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
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
        dataset_path=dataset_path(), 
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
    fcn_resnet = FCNResNet().to(device)

    # Criterion and Optimizor
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.SGD(
        params=fcn_resnet.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    # Load pretrained model
    if args.resume:
        if os.path.isfile(args.resume):
            print("Load checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            fcn_resnet.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        print('Load pretrained ResNet 34.')
        # resnet32_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        pretrained_model = torch.load(os.path.join(saved_models_path(), 'resnet34-333f7ec4.pth'))
        fcn_resnet.load_pretrained_model(pretrained_model)

    # Start training
    fcn_resnet.train()
    epoch_endtime = time.time()
    if not os.path.exists(saved_models_path()):
        os.mkdir(saved_models_path())

    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.total_epochs):
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        batch_endtime = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)   

        for i, (input, target) in enumerate(train_loader):
            
            input, target = input.to(device), target.to(device)

            output = fcn_resnet(input)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            batch_time.update(time.time() - batch_endtime)
            batch_endtime = time.time()

            if i % 100 == 0:
                print('Epoch: [{0}/{1} | {2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, args.total_epochs, i, len(train_loader),
                      batch_time=batch_time, loss=losses))

        epoch_time.update(time.time() - epoch_endtime)
        epoch_endtime = time.time()

        torch.save(
            obj={
                'epoch': epoch,
                'model': fcn_resnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': losses.avg,
            },
            f=os.path.join(saved_models_path(), 'checkpoint.pth.tar')
        )

        print('Epoch: [{0}/{1}]\t'
              'Time {epoch_time.val:.3f} ({epoch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
              epoch, args.total_epochs, 
              epoch_time=epoch_time, loss=losses))
        
        print('Model saved.')


if __name__ == '__main__':
    train_FCNResNet()