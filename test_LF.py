import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from utils import pgd_attack
from autoattack import AutoAttack
import torch.backends.cudnn as cudnn

import time
from utils import AverageMeter, eval_adv_test, logger, eval_test
import numpy as np
import copy

parser = argparse.ArgumentParser(
    description='Linear Finetuning (SLF and ALF)')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models,\
                    we recommend to specify it as a subdirectory of the pretraining export path',
                    required=True)

parser.add_argument('--data', type=str, default='data/CIFAR10',
                    help='location of the data')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset to be used (cifar10 or cifar100)')

parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 512)')

parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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

parser.add_argument('--eval-only', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')

parser.add_argument('--resume', action='store_true',
                    help='if resume training')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the start epoch number')

parser.add_argument('--decreasing_lr', default='10,20',
                    help='decreasing strategy')
parser.add_argument('--cvt_state_dict', action='store_true',
                    help='Need to be specified if pseudo-label finetune is not implemented')

parser.add_argument('--bnNameCnt', default=1, type=int)
parser.add_argument('--evaluation_mode', type=str, default='SLF',
                    help='SLF or ALF')

parser.add_argument('--test_frequency', type=int, default=0,
                    help='validation frequency during finetuning, 0 for no evaluation')   

parser.add_argument('--gpu_id', type=str, default='0')

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
    transforms.RandomCrop(96 if args.dataset == 'stl10' else 32, padding=4),
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
elif args.dataset == 'stl10':
    train_datasets = torchvision.datasets.STL10(
        root=args.data, split='train', transform=transform_train, download=True)
    vali_datasets = datasets.STL10(
        root=args.data, split='train', transform=transform_test, download=True)
    testset = datasets.STL10(
        root=args.data, split='test', transform=transform_test, download=True)
    num_classes = 10     
else:
    print("dataset {} is not supported".format(args.dataset))
    assert False

train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=args.batch_size, shuffle=True)

vali_loader = torch.utils.data.DataLoader(
    vali_datasets,
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=True)

def train(args, model, device, train_loader, optimizer, epoch, log):
    # model.train()
    model.eval()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()
        
        if args.evaluation_mode == 'ALF':
            data = pgd_attack(model, data, target, device, eps=args.epsilon,
                                      alpha=args.step_size, iters=args.num_steps_train, forceEval=True).data

        output = model.eval()(data)
        loss = criterion(output, target)

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
    
def main():
    if args.dataset == 'stl10':
        from models.resnet_stl import resnet18
    else:
        from models.resnet import resnet18
    model = resnet18(num_classes=num_classes).to(device)
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = args.start_epoch

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if args.cvt_state_dict:
            state_dict = cvt_state_dict(
                state_dict, args, num_classes=num_classes)
        elif not args.eval_only and not args.resume:
            state_dict['fc.weight'] = torch.zeros(num_classes, 512).cuda()
            state_dict['fc.bias'] = torch.zeros(num_classes).cuda()

        # model.normalize = torch.nn.Identity()
        model.load_state_dict(state_dict, strict=False)
        log.info('read checkpoint {}'.format(args.checkpoint))

    if args.resume:
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

    if args.eval_only:
        model.eval()
        _, test_tacc = eval_test(model, device, test_loader, log)
        test_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                                   criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
        log_path = 'checkpoints/' + args.experiment + '/robustness_result.txt'
        t_loader = torch.utils.data.DataLoader(
            testset, batch_size=10000 if args.dataset != 'stl10' else 8000, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        runAA(model, t_loader, log_path)
        log.info("On the {}, test tacc is {}, test atacc is {}".format(
            args.checkpoint, test_tacc, test_atacc))
        
        return

    best_atacc = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        # linear classification
        train(args, model, device, train_loader, optimizer, epoch, log)
        scheduler.step()

        # evaluation
        if (not args.test_frequency == 0) and (epoch % args.test_frequency == 1 or args.test_frequency == 1):
            print('================================================================')
            eval_test(model, device, test_loader, log)
            vali_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
            if vali_atacc > best_atacc:
                best_atacc = vali_atacc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, os.path.join(model_dir, 'model_bestAT.pt'))
            print('================================================================')

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, os.path.join(model_dir, 'model_finetune.pt'))

    # testing
    _, test_tacc = eval_test(model, device, test_loader, log)
    test_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                               criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
    log.info("On the final model, test tacc is {}, test atacc is {}".format(
        test_tacc, test_atacc))
        
    log_path = 'checkpoints/' + args.experiment + '/robustness_result.txt'
    aa_loader = torch.utils.data.DataLoader(
        testset, batch_size=8000 if args.dataset == 'stl10' else 10000, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    runAA(model, aa_loader, log_path)

def runAA(model, loader, log_path):
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_path)
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
        adversary.run_standard_evaluation(images, labels, bs=100)

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

if __name__ == '__main__':
    main()
