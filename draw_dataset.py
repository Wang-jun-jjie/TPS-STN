import argparse
import logging
import time
from pathlib import PurePath
import itertools
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser( description='PyTorch draw dataset comparison')
parser.add_argument('--seed',             default=70082353,    type=int,   help='random seed')
parser.add_argument('--batch_size', '-b', default=64,       type=int,   help='mini-batch size (default: 8)')
parser.add_argument('--image-size',       default=28,          type=int,   help='input image size (default: 28 for MNIST)')
parser.add_argument('--data-directory-1', default='./mnist_png',type=str, help='dataset1 root directory')
parser.add_argument('--data-directory-2', default='./mnist_dis',type=str, help='dataset2 root directory')
parser.add_argument('--dataset1-name',    default='MNIST',type=str, help='dataset1 name')
parser.add_argument('--dataset2-name',    default='Distorted MNIST',type=str, help='dataset2 name')
args = parser.parse_args()

# warrning: filename actually include the last dirname with it
class ImageFolderWithFilename(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithFilename, self).__getitem__(index)
        path, _ = self.imgs[index]
        path_split = PurePath(path).parts
        filename = path_split[-2] + '/' +path_split[-1]
        new_tuple = (original_tuple + (filename,))
        return new_tuple

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('==> Preparing dataset..')
    # Training dataset
    mean, std = (0.1307,), (0.3081,)
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test1_dataset = ImageFolderWithFilename(root=args.data_directory_1+'/training', \
        transform=test_transform)
    test1_loader = torch.utils.data.DataLoader(test1_dataset, batch_size=args.batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test2_dataset = ImageFolderWithFilename(root=args.data_directory_2+'/testing', \
        transform=test_transform)
    test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    def draw():
        print('==> Draw from dataset..')
        mean, std = (0.1307,), (0.3081,)
        inv_normalize = transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
        )
        with torch.no_grad():
            data1 = next(iter(test1_loader))[0]
            data1 = inv_normalize(data1)
            data1 = torch.clamp(data1, 0, 1)
            data2 = next(iter(test2_loader))[0]
            data2 = inv_normalize(data2)
            data2 = torch.clamp(data2, 0, 1)
            data1_grid = torchvision.utils.make_grid(data1.detach().cpu()).permute(1,2,0).numpy()
            data2_grid = torchvision.utils.make_grid(data2.detach().cpu()).permute(1,2,0).numpy()
            fig, axs = plt.subplots(1, 2, figsize=(24, 12))
            axs[0].imshow(data1_grid)
            axs[0].set_title(args.dataset1_name)
            axs[1].imshow(data2_grid)
            axs[1].set_title(args.dataset2_name)
            plt.show()
            plt.savefig(result_folder+"/__"+args.dataset1_name+"__"+args.dataset2_name)
            plt.close()

    draw()    
    print('==> Done.')


main()