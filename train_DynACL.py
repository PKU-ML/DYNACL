import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import time

import numpy as np

from data.dataset import *
from optimizer.lars import LARS
from random import randint
import os
# import apex

parser = argparse.ArgumentParser(description='DynACL')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models', required=True)
parser.add_argument('--data', type=str, default='data/CIFAR10',
                    help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset to be used, (cifar10 or cifar100)')

parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run')

parser.add_argument('--print_freq', default=10,
                    type=int, help='print frequency')

parser.add_argument('--checkpoint', default='', type=str,
                    help='saving pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')

parser.add_argument('--optimizer', default='lars',
                    type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine',
                    type=str, help='lr scheduler type')

parser.add_argument('--swap_param', type=float,
                    default=2/3, help='weight swap param')

parser.add_argument('--twoLayerProj', action='store_true',
                    help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int,
                    help='how many iterations employed to attack the model')

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--val_frequency', type=int, default=50, help='test performance frequency')
parser.add_argument('--reload_frequency', type=int, default=50, help='data reload frequency')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
n_gpu = torch.cuda.device_count()
device = 'cuda'

def main():
    global args

    assert args.dataset in ['cifar100', 'cifar10', 'stl10']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))

    bn_names = ['normal', 'pgd']

    # define model
    if args.dataset != 'stl10':
        from models.resnet_multi_bn import resnet18_momentum, proj_head
    else:
        from models.resnet_multi_bn_stl import resnet18_momentum, proj_head
    model = resnet18_momentum(pretrained=False, bn_names=bn_names)

    ch = model.encoder_k.fc.in_features
    model.encoder_q.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    model.encoder_k.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    model._init_encoder_k()
    model.cuda()
    cudnn.benchmark = True

    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(
        0.4, 0.4, 0.4, 0.1)], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(96 if args.dataset == 'stl10' else 32),
        transforms.RandomHorizontalFlip(p=0.5),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    tfs_val = transforms.Compose([
        transforms.RandomCrop(96 if args.dataset == 'stl10' else 32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(
                root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR10(
            root=args.data, train=True, transform=tfs_val, download=True)
        test_datasets = datasets.CIFAR10(
            root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(
                root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR100(
            root=args.data, train=True, transform=tfs_val, download=True)
        test_datasets = datasets.CIFAR100(
            root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 100
    elif args.dataset == 'stl10':
        train_datasets = CustomSTL10(
                root=args.data, split='unlabeled', transform=tfs_train, download=True)
        val_train_datasets = datasets.STL10(
            root=args.data, split='train', transform=tfs_val, download=True)
        test_datasets = datasets.STL10(
            root=args.data, split='test', transform=tfs_test, download=True)
        num_classes = 10
    else:
        print("unknown dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)

    val_train_loader = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=4,
        batch_size=args.batch_size)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range(start_epoch - 1):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(
                args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    for epoch in range(start_epoch, args.epochs + 1):

        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        # reload the dataset
        if(epoch % args.reload_frequency == 1 or args.resume or args.reload_frequency == 1):
            args.resume = False
            strength = 1 - (epoch - 1) / args.epochs
            train_loader = reload(strength)
            log.info("<== Data reloaded ==>")
            
        log.info("current strength is {}".format(strength))
        train(train_loader, model, optimizer, scheduler,
                epoch, log, num_classes=num_classes)


        if(epoch % 25 == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.encoder_k.state_dict(),
            }, filename=os.path.join(save_dir, 'model_encoder_k.pt'))

        if epoch % args.val_frequency == 0:

            acc, tacc, rtacc = validate(val_train_loader, test_loader,
                                        model, log, num_classes=num_classes)
            log.info('train_accuracy {acc:.3f}'
                     .format(acc=acc))
            log.info('test_accuracy {tacc:.3f}'
                     .format(tacc=tacc))
            log.info('test_robust_accuracy {rtacc:.3f}'
                     .format(rtacc=rtacc))

            # evaluate acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': acc,
                'tacc': tacc,
                'rtacc': rtacc,
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, filename=os.path.join(save_dir, 'model_encoder_k_{}.pt'.format(epoch)))


def train(train_loader, model, optimizer, scheduler, epoch, log, num_classes):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()

        inputs_adv = PGD_contrastive(model, inputs, iters=args.pgd_iter)
        features_adv = model.train()(inputs_adv, 'pgd', swap=True)
        features = model.train()(inputs, 'normal', swap=True)
        model._momentum_update_encoder_k()

        weight_adv = min(1.0 + (epoch // args.reload_frequency) * (args.reload_frequency  / args.epochs) * args.swap_param, 2)

        loss = (nt_xent(features) * (2 - weight_adv) +
                    nt_xent(features_adv) * weight_adv) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                         epoch, i, len(train_loader), loss=losses,
                         data_time=data_time_meter, train_time=train_time_meter))
    scheduler.step()
    return losses.avg


def validate(val_loader, test_loader, model, log, num_classes=10):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    # Note that the backbone of model.encoder_k never needs gradient

    previous_fc = model.encoder_k.fc
    ch = model.encoder_k.fc.in_features
    model.encoder_k.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 25
    lr = 0.01

    parameters = list(filter(lambda p: p.requires_grad, model.encoder_k.parameters()))
    assert(len(parameters) == 2)

    optimizer = torch.optim.SGD(
        parameters, lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10,20], gamma=0.1)

    for epoch in range(epochs_max):
        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (sample) in enumerate(val_loader):

            x, y = sample[0].cuda(), sample[1].cuda()
            p = model.eval()(x, 'pgd')

            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(float(loss.detach().cpu()))

            train_time = time.time() - end
            end = time.time()
            train_time_meter.update(train_time)

        scheduler.step()

        log.info('Test epoch: ({0})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'train_time: {train_time.avg:.2f}\t'.format(
                     epoch, loss=losses, train_time=train_time_meter))

    acc = []
    round = 0
    for loader in [val_loader, test_loader, test_loader]:
        round += 1
        losses = AverageMeter()
        losses.reset()
        top1 = AverageMeter()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            if round == 3:
                inputs = pgd_attack(model, inputs, targets, device,
                                    eps=8.0/255, alpha=2.0/255, iters=20, advFlag='pgd').data

            for name, param in model.encoder_k.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            parameters = list(
                filter(lambda p: p.requires_grad, model.encoder_k.parameters()))
            assert len(parameters) == 2  # fc.weight, fc.bias
            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs, 'pgd')
                loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if i % args.print_freq == 0:
                log.info('Test: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                             i, len(loader), loss=losses, top1=top1))

        acc.append(top1.avg)

    # recover every thing
    model.encoder_k.fc = previous_fc
    model.cuda()

    for param in model.encoder_k.parameters():
        param.requires_grad = False

    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def PGD_contrastive(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)


    for _ in range(iters):
        features = model.eval()(inputs + delta, 'pgd', swap=True)

        model.zero_grad()
        loss = nt_xent(features)
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

    return (inputs + delta).detach()

def reload(strength):
    global args
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(
        0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(
            96 if args.dataset == 'stl10' else 32, scale=(1.0 - 0.9 * strength, 1.0)),
        # No need to decay horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        datasets = CustomCIFAR10(
                root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'cifar100':
        datasets = CustomCIFAR100(
                root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'stl10':
        datasets = CustomSTL10(
            root=args.data, split='unlabeled', transform=tfs_train, download=True)
    else:
        assert False
    loader = torch.utils.data.DataLoader(
        datasets,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=True)

    return loader

if __name__ == '__main__':
    main()
