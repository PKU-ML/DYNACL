import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from utils import pgd_attack, accuracy
import torch.nn as nn
from autoattack import AutoAttack
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models.resnet_multi_bn import resnet18
from models.resnet import resnet18 as resnet18_singleBN
import time
from utils import AverageMeter, logger, trades_loss_dual
import numpy as np
import copy

parser = argparse.ArgumentParser(
    description='TRADES full finetuning')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models,\
                    we recommend to specify it as a subdirectory of the pretraining export path',
                    required=True)

parser.add_argument('--data', type=str, default='data/CIFAR10',
                    help='location of the data')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset to be used (cifar10 or cifar100)')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing')

parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')

parser.add_argument('--epsilon', type=float, default=8. / 255.,
                    help='perturbation')
parser.add_argument('--step-size', type=float, default=2. / 255.,
                    help='perturb step size')
parser.add_argument('--num-steps-train', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--num-steps-test', type=int, default=20,
                    help='perturb number of steps')

parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--eval-only', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--resume', action='store_true',
                    help='if resume training')

parser.add_argument('--decreasing_lr', default='15,20',
                    help='decreasing strategy')

parser.add_argument('--bnNameCnt', default=1, type=int) # do not modify

parser.add_argument('--gpu_id', type=str, default='1')

args = parser.parse_args()

# settings
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
model_dir = os.path.join('checkpoints', args.experiment)
print(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
log = logger(os.path.join(model_dir))
log.info(str(args))
device = 'cuda'
cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == 'cifar10':
    train_datasets = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_train)
    vali_datasets = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_datasets = torchvision.datasets.CIFAR100(
        root=args.data, train=True, download=True, transform=transform_train)
    vali_datasets = torchvision.datasets.CIFAR100(
        root=args.data, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 100
else:
    print("dataset {} is not supported".format(args.dataset))
    assert False

train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=args.batch_size, shuffle=True)

train_noAug_loader = torch.utils.data.DataLoader(
    vali_datasets,
    batch_size=args.batch_size, shuffle=True)

vali_loader = torch.utils.data.DataLoader(
    vali_datasets,
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False)

def train(args, model, device, train_loader, optimizer, epoch, log):
    model.train()
    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        loss = trades_loss_dual(model=model,
                                   x_natural=data,
                                   y=target,
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps_train,
                                   natural_mode='normal')

        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))


def eval_test(model, device, loader, log, advFlag='pgd'):
    model.eval()
    test_loss = 0
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if advFlag is not None:
                output = model.eval()(data, 'pgd')
            else:
                output = model.eval()(data)
            test_loss += F.cross_entropy(output,
                                         target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    test_loss /= len(loader.dataset)
    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, whole,
        100. * correct / whole))
    test_accuracy = correct / whole
    return test_loss, test_accuracy * 100

def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=20, advFlag='pgd'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        if advFlag is not None:
            input_adv = pgd_attack(model, input, target, device,
                                eps=epsilon, iters=attack_iter, alpha=alpha, advFlag='pgd').data
            with torch.no_grad():
                output = model.eval()(input_adv, 'pgd')
                loss = criterion(output, target)
        else:
            input_adv = pgd_attack(model, input, target, device,
                                eps=epsilon, iters=attack_iter, alpha=alpha).data
            with torch.no_grad():
                output = model.eval()(input_adv)
                loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def main():
    # init model, ResNet18() can be also used here for training
    bn_names = ['normal', 'pgd']
    model = resnet18(pretrained=False, bn_names=bn_names, num_classes=num_classes)
    model.cuda()

    parameters = model.parameters()

    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = 0

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # zero init fc
        state_dict['fc.weight'] = torch.zeros(num_classes, 512).cuda()
        state_dict['fc.bias'] = torch.zeros(num_classes).cuda()

        model.load_state_dict(state_dict, strict=False)
        log.info('read checkpoint {}'.format(args.checkpoint))

    elif args.resume:
        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range(start_epoch):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(
                args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        # linear classification
        train(args, model, device, train_loader, optimizer, epoch, log)
        scheduler.step()

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, os.path.join(model_dir, 'model_finetune.pt'))

    model_save = resnet18_singleBN(num_classes=num_classes) # original resnet (without multi BatchNorm)
    state_dict = torch.load(os.path.join(model_dir, 'model_finetune.pt'))['state_dict']
    state_dict = cvt_state_dict(state_dict,args)
    model_save.load_state_dict(state_dict)
    model_save.eval().cuda()

    _, test_tacc = eval_test(model_save, device, test_loader, log, advFlag=None)
    test_atacc = eval_adv_test(model_save, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                               criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test, advFlag=None)
    log.info("On the final model, test tacc is {}, test atacc is {}".format(
        test_tacc, test_atacc))
    
    aa_loader = torch.utils.data.DataLoader(
        testset, batch_size=10000, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    log_path = 'checkpoints/' + args.experiment + '/robustness_result.txt'
    runAA(model_save, aa_loader, log_path)
    torch.save({
        'state_dict': model_save.state_dict(),
    }, os.path.join(model_dir, 'model_full_finetune_singleBN.pt'))

def cvt_state_dict(state_dict, args):
    # deal with adv bn
    state_dict_new = copy.deepcopy(state_dict)

    if args.bnNameCnt >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(args.bnNameCnt), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new['fc.weight'] = state_dict['fc.weight']
    state_dict_new['fc.bias'] = state_dict['fc.bias']
    return state_dict_new

def runAA(model, loader, log_path):
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_path)
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
        adversary.run_standard_evaluation(images, labels, bs=100)

if __name__ == '__main__':
    main()
