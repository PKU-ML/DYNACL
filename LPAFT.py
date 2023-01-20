import torch.nn as nn
import torch
import argparse
from autoattack import AutoAttack

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from data.dataset import *

import numpy as np

from optimizer.lars import LARS
from random import randint

parser = argparse.ArgumentParser(description='DynACL++ (LPAFT for SLF & ALF)')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models,\
                    we recommend to specify it as a subdirectory of the pretraining export path',
                    required=True)

parser.add_argument('--data', type=str, default='data/CIFAR10',
                    help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset to be used, (cifar10 or cifar100 or stl10)')

parser.add_argument('--batch_size', type=int, default=512, help='batch size')

parser.add_argument('--epochs_head', default=10, type=int,
                    help='number of epochs to train head')
parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')

parser.add_argument('--print_freq', default=50,
                    type=int, help='print frequency')
parser.add_argument('--checkpoint', required=True, type=str,
                    help='saving pretrained model')

parser.add_argument('--optimizer', default='sgd',
                    type=str, help='optimizer type')
parser.add_argument('--lr', default=0.1, type=float, help='optimizer lr')
parser.add_argument('--lr_head', default=0.01, type=float, help='optimizer lr')

parser.add_argument('--twoLayerProj', action='store_true',
                    help='if specified, use two layers linear head for simclr proj head')

parser.add_argument('--pgd_iter', default=5, type=int,
                    help='how many iterations employed to attack the model')
parser.add_argument('--val_frequency', type=int, default=5, help='validation frequency')
parser.add_argument('--evaluation_mode', type=str, default='SLF', 
                    help='SLF (standard linear) or ALF (adversarial linear)')

parser.add_argument('--gpu_id', type=str, default='0')

parser.add_argument('--label_path', type=str,
                    required=True, help='path to pseudo label')
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

    assert args.dataset in ['cifar100', 'cifar10', 'stl10']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    
    if args.dataset == 'stl10':
        from models.resnet_stl import resnet18
    else:
        from models.resnet import resnet18
        
    num_classes = 10 if args.dataset != 'cifar100' else 100

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
    model.cuda()
    cudnn.benchmark = True

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
    elif args.dataset == 'stl10':
        train_datasets = STL10IndexPseudoLabelEnsemble(
            root=args.data, split='unlabeled', transform=tfs_val, pseudoLabel=pseudo_label, download=True)
        val_train_datasets = datasets.STL10(
            root=args.data, split='train', transform=tfs_val, download=True)
        test_datasets = datasets.STL10(
            root=args.data, split='test', transform=tfs_test, download=True)
        num_classes = 10        

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

    state_dict = cvt_state_dict(state_dict, args, num_classes) # load the adversarial route
    model.load_state_dict(state_dict, strict=True)
    log.info("checkpoint loaded from " + args.checkpoint)
    
    if args.eval_only:
        acc, tacc, rtacc = validate(val_train_loader, test_loader,
                                    model, log, num_classes=num_classes, autoattack=True)
        log.info('train_accuracy {acc:.3f}'
                    .format(acc=acc))
        log.info('test_accuracy {tacc:.3f}'
                    .format(tacc=tacc))
        log.info('test_robust_accuracy {rtacc:.3f}'
                    .format(rtacc=rtacc))
        return
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    
    for epoch in range(start_epoch, args.epochs_head + 1):

        log.info("current lr is {}".format(
            optimizer_head.state_dict()['param_groups'][0]['lr']))
        
        train_head(train_loader, model, optimizer_head, scheduler_head, epoch, log)

    validate(val_train_loader, test_loader, model, log, num_classes=num_classes)
         
    for name, param in model.named_parameters():
        param.requires_grad = True

    for epoch in range(start_epoch, args.epochs + 1):

        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        train(train_loader, model, optimizer, scheduler, epoch, log)

        if(epoch % 5 == 0): # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % args.val_frequency == 0 and epoch > 0:

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

    log.info(f'Pseudo-label finetune ends. \nStart {args.evaluation_mode} test:')
    acc, tacc, rtacc = validate(val_train_loader, test_loader,
                                model, log, num_classes=num_classes, autoattack=True, evaluation_mode=args.evaluation_mode)
    log.info('train_accuracy {acc:.3f}'
             .format(acc=acc))
    log.info('test_accuracy {tacc:.3f}'
             .format(tacc=tacc))
    log.info('test_robust_accuracy {rtacc:.3f}'
             .format(rtacc=rtacc))


def train(train_loader, model, optimizer, scheduler, epoch, log):
    model.train()
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    end = time.time()

    for i, (inputs, _, targets, _) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)
        inputs = inputs.cuda()

        loss = trades_loss(model=model,
                            x_natural=inputs,
                            y=targets.long().cuda(),
                            optimizer=optimizer,
                            perturb_steps=10)

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

        outputs = model.eval()(inputs)
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


def validate(val_loader, test_loader, model, log, num_classes=10, autoattack=False, evaluation_mode='SLF'):
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
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.fc
    ch = model.fc.in_features
    model.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 25
    
    if args.dataset == 'cifar10':
        lr = 0.01
    elif args.dataset == 'cifar100':
        lr = 0.05
    else:
        lr = 0.1

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
            
            if evaluation_mode == 'ALF':
                x = pgd_attack(model, x, y, device).data
                
            p = model.eval()(x)

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
            if round == 3: # run adversarial test for the last trail
                inputs = pgd_attack(model, inputs, targets, device).data

            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs)
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

    if autoattack:
        log_path = 'checkpoints/' + args.experiment + '/robustness_result.txt'
        runAA(model, log_path)

    # recover every thing
    model.fc = previous_fc
    model.cuda()
    for param in model.parameters():
        param.requires_grad = True

    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def cvt_state_dict(state_dict, args, num_classes):
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
        # print(name)
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

    # zero init fc
    state_dict_new['fc.weight'] = torch.zeros(num_classes, 512).cuda()
    state_dict_new['fc.bias'] = torch.zeros(num_classes).cuda()

    return state_dict_new


def runAA(model, log_path): # run AutoAttack
    model.eval()
    global args
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        test_datasets = datasets.CIFAR10(
            root=args.data, train=False, transform=tfs_test, download=True)
    elif args.dataset == 'cifar100':
        test_datasets = datasets.CIFAR100(
            root=args.data, train=False, transform=tfs_test, download=True)
    else:
        test_datasets = datasets.STL10(
            root=args.data, split='test' ,transform=tfs_test, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=8000 if args.dataset == 'stl10' else 10000, pin_memory=True, num_workers=4) # load whole dataset
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-t'], log_path=log_path)
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        adversary.run_standard_evaluation(images, labels, bs=100)


if __name__ == '__main__':
    main()
