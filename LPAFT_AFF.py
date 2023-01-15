import torch.nn as nn
import torch
import argparse
from autoattack import AutoAttack
import numpy as np
import os
import time
import copy

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
# from models.resnet_multi_bn import resnet
from models.resnet_multi_bn import resnet18
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from data.dataset import CIFAR10IndexPseudoLabelEnsemble, CIFAR100IndexPseudoLabelEnsemble

from optimizer.lars import LARS

from models.resnet import resnet18 as resnet18_single

parser = argparse.ArgumentParser(description='DynACL++ (LPAFT-AFF)')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models,\
                    we recommend to specify it as a subdirectory of the pretraining export path',
                    required=True)
parser.add_argument('--data', type=str, default='data/CIFAR10',
                    help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset to be used, (cifar10 or cifar100)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--batch_size_AT', type=int, default=128, help='batch size')

parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')
parser.add_argument('--epochs_head', default=10, type=int,
                    help='number of epochs to train head')

parser.add_argument('--print_freq', default=50,
                    type=int, help='print frequency')
parser.add_argument('--checkpoint', default='', type=str,
                    help='saving pretrained model')

parser.add_argument('--optimizer', default='sgd',
                    type=str, help='optimizer type')
parser.add_argument('--lr', default=0.1, type=float, help='optimizer lr')
parser.add_argument('--lr_head', default=0.01, type=float, help='optimizer lr')

parser.add_argument('--twoLayerProj', action='store_true',
                    help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int,
                    help='how many iterations employed to attack the model')

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--val_frequency', type=int, default=5)

parser.add_argument('--label_path', type=str, default='', help='path of label')
parser.add_argument('--bnNameCnt', type=int, default=1,
                    help='0 for normal route, 1 for adv route')
parser.add_argument('--eval-only', action='store_true',)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
n_gpu = torch.cuda.device_count()
device = 'cuda'

pseudo_label = torch.load(args.label_path, map_location="cpu").numpy().tolist()


def main():
    global args

    assert args.dataset in ['cifar100', 'cifar10']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    
    num_classes = 10 if args.dataset != 'cifar100' else 100

    bn_names = ['normal', 'pgd']
    model = resnet18(pretrained=False, bn_names=bn_names)
    model.fc = nn.Linear(512, num_classes)
    model.cuda()
    cudnn.benchmark = True

    tfs_val = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CIFAR10IndexPseudoLabelEnsemble(root=args.data,
                                                         transform=tfs_val,
                                                         pseudoLabel=pseudo_label,
                                                         download=True)
        val_train_datasets = datasets.CIFAR10(
            root=args.data, train=True, transform=tfs_val, download=True)
        test_datasets = datasets.CIFAR10(
            root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = CIFAR100IndexPseudoLabelEnsemble(root=args.data,
                                                          transform=tfs_val,
                                                          pseudoLabel=pseudo_label,
                                                          download=True)
        val_train_datasets = datasets.CIFAR100(
            root=args.data, train=True, transform=tfs_val, download=True)
        test_datasets = datasets.CIFAR100(
            root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 100
    else:
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True, drop_last=True)

    val_train_loader = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)
    
    val_train_loader_AT = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=4,
        batch_size=args.batch_size_AT,
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
        optimizer_head = torch.optim.SGD(
            model.fc.parameters(), lr=args.lr_head, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10,20], gamma=0.1)
    scheduler_head = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_head, milestones=[], gamma=0.1)

    start_epoch = 1

    assert args.checkpoint != ''
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    state_dict['fc.weight'] = torch.zeros(num_classes, 512).cuda()
    state_dict['fc.bias'] = torch.zeros(num_classes).cuda()
    model.load_state_dict(state_dict, strict=False) 
    
    if args.eval_only:
        validate(val_train_loader, test_loader,
                        model, log, num_classes=num_classes)
        return
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    
    # linear probing
    log.info("Starts linear probing")
    for epoch in range(start_epoch, args.epochs_head + 1):

        log.info("current lr is {}".format(
            optimizer_head.state_dict()['param_groups'][0]['lr']))
        
        train_head(train_loader, model, optimizer_head, scheduler_head, epoch, log)

    # Pseudo finetuning
    log.info('Starts pseudo finetuning')
    for name, param in model.named_parameters():
        param.requires_grad = True

    for epoch in range(start_epoch, args.epochs + 1):

        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        train(train_loader, model, optimizer, scheduler, epoch, log)

        if(epoch % 5 == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % args.val_frequency == 0 and epoch > 0:

            acc, tacc, rtacc = validate(val_train_loader, test_loader,
                                        model, log, num_classes=num_classes)
            
            # evaluate acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': acc,
                'tacc': tacc,
                'rtacc': rtacc,
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))

    log.info('Pseudo-label finetune ends. \nTest: (SLF evaluation)')
    validate(val_train_loader, test_loader, model, log, num_classes=num_classes)

    # AFF evalution
    log.info('Starts AFF evaluation')
    # zero init FC
    model.fc.weight = torch.nn.Parameter(torch.zeros(model.fc.weight.shape))
    model.fc.bias = torch.nn.Parameter(torch.zeros(model.fc.bias.shape))
    model.fc.cuda()
    
    optimizer_AFF = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    scheduler_AFF = torch.optim.lr_scheduler.MultiStepLR(optimizer_AFF, milestones=[15,20], gamma=0.1)
    
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        log.info("current lr is {}".format(
            optimizer_AFF.state_dict()['param_groups'][0]['lr']))

        # linear classification
        train_AFF(args, model, device, val_train_loader_AT, optimizer_AFF, epoch, log)
        scheduler_AFF.step()

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, os.path.join(save_dir, 'model_finetune.pt'))  

    model_save = resnet18_single(num_classes=num_classes) # original resnet (without multi BatchNorm)
    state_dict = torch.load(os.path.join(save_dir, 'model_finetune.pt'))['state_dict']
    state_dict = cvt_state_dict(state_dict,args)
    model_save.load_state_dict(state_dict)
    model_save.eval().cuda()
    
    _, test_tacc = eval_test(model_save, device, test_loader, log, advFlag=None)
    test_atacc = eval_adv_test(model_save, device, test_loader, epsilon=8/255, alpha=2/255,
                               criterion=F.cross_entropy, log=log, attack_iter=20)
    log.info("On the final model (AFF evaluation), test tacc is {}, test atacc is {}".format(
        test_tacc, test_atacc))
    
    log_path = 'checkpoints/' + args.experiment + '/robustness_result.txt'
    runAA(model_save, log_path)
    torch.save({
        'state_dict': model_save.state_dict(),
        }, os.path.join(save_dir, 'model_full_finetune_singleBN.pt'))


def train(train_loader, model, optimizer, scheduler, epoch, log):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    end = time.time()

    for i, (inputs, _, targets, _) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)
        inputs = inputs.cuda()
        
        loss = trades_loss_dual(model=model,
                            x_natural=inputs,
                            y=targets.long().cuda(),
                            optimizer=optimizer,
                            perturb_steps=10,
                            natural_mode='pgd')
        
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

def train_head(train_loader, model, optimizer, scheduler, epoch, log):
    model.eval()
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    end = time.time()

    criterion = nn.CrossEntropyLoss()

    for i, (inputs, _, targets, _) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)
        inputs = inputs.cuda()

        outputs = model.eval()(inputs, 'pgd')
        loss = criterion(outputs, targets.long().cuda())

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
    model.eval()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.fc
    ch = model.fc.in_features
    model.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 25
    lr = 0.01

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
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
    model.fc = previous_fc
    model.cuda()
    for param in model.parameters():
        param.requires_grad = True
        
    log.info('train_accuracy {acc:.3f}'
                     .format(acc=acc[0]))
    log.info('test_accuracy {tacc:.3f}'
                     .format(tacc=acc[1]))
    log.info('test_robust_accuracy {rtacc:.3f}'
                     .format(rtacc=acc[2]))

    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

def runAA(model, log_path):
    model.eval()
    global args
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        test_datasets = datasets.CIFAR10(
            root=args.data, train=False, transform=tfs_test, download=True)
    else:
        test_datasets = datasets.CIFAR100(
            root=args.data, train=False, transform=tfs_test, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=10000, pin_memory=True, num_workers=4)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_path)
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        adversary.run_standard_evaluation(images, labels, bs=100)

def train_AFF(args, model, device, train_loader, optimizer, epoch, log):
    model.train()
    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    # criterion = nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        loss = trades_loss_dual(model=model,
                                   x_natural=data,
                                   y=target,
                                   optimizer=optimizer,
                                   natural_mode='normal')

        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % 10 == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))

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

if __name__ == '__main__':
    main()
