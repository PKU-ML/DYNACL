

import os
import numpy as np
from kmeans_pytorch import kmeans
import torch 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import torch.nn as nn
import argparse

'''
Assign and store the pseudo labels according to the
features extracted by the normal route of the backbone.
Labels will be stored in the same folder of the checkpoint.
'''

parser = argparse.ArgumentParser(description='Assign pseudo label')
parser.add_argument('--gpu_id', type=str, default='9')
parser.add_argument('--checkpoint', type=str, default='', 
                    help='EXAMPLE: ./checkpoints/PRETRAIN_EXPORT_PATH/model_encoder_k.pt')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data', type=str, default='data/CIFAR10')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if not args.dataset == 'stl10':
    from models.resnet_multi_bn import resnet18, proj_head
else:
    from models.resnet_multi_bn_stl import resnet18, proj_head
bn_names = ['normal', 'pgd']
model = resnet18(num_classes = 10, bn_names = bn_names)
ch = model.fc.in_features
model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=False)
model.cuda()

def assign(load_path, name):
    checkpoint = torch.load(load_path, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    
    model.load_state_dict(checkpoint)
    # model.cuda()
    tfs_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    if args.dataset == 'cifar10':
        test_datasets = datasets.CIFAR10(
                root=args.data, train=True, transform=tfs_test, download=True)
    elif args.dataset == 'cifar100':
        test_datasets = datasets.CIFAR100(
                root=args.data, train=True, transform=tfs_test, download=True)
    elif args.dataset == 'stl10':
        test_datasets = datasets.STL10(
            root=args.data, split='unlabeled', transform=tfs_test, download=True)
    else:
        raise NotImplementedError
    test_loader = torch.utils.data.DataLoader(
            test_datasets,
            num_workers=4,
            batch_size=5000,
            shuffle=False)
    rep = None

    for x, _ in test_loader:
        with torch.no_grad():
            x = x.cuda()
            x = model.eval()(x, 'normal', return_features=True)
            print(x.size())
            if(rep == None):
                rep = x.cpu()
            else:
                rep = torch.concat([rep,x.cpu()])
                
    rep = rep.cuda()
    
    for num_clusters in [10]:
    
        cluster_ids_x, _ = kmeans(
            X=rep, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            )
    
        print(cluster_ids_x)
        save_dir = load_path.replace(load_path.split('/')[-1], name+f'_{num_clusters}.pth')
        torch.save(cluster_ids_x, save_dir)

def main():
    load_path = args.checkpoint
    assign(load_path,'cluster_ids')
    
if __name__ == '__main__':
    main()