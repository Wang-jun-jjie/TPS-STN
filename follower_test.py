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

parser = argparse.ArgumentParser( description='PyTorch model test on MNIST')
parser.add_argument('--resume', '-r',     action='store_true', help='resume from checkpoint')
parser.add_argument('--sess',             default='default',   type=str,   help='session id')
parser.add_argument('--seed',             default=70082353,    type=int,   help='random seed')
parser.add_argument('--batch_size', '-b', default=8,       type=int,   help='mini-batch size (default: 8)')
parser.add_argument('--epochs',           default=20,          type=int,   help='number of total epochs to run')
parser.add_argument('--image-size',       default=28,          type=int,   help='input image size (default: 28 for MNIST)')
parser.add_argument('--data-directory',   default='./mnist_png',type=str, help='dataset inputs root directory')
# parser.add_argument('--output_directory', default='./mnist_dis',type=str, help='dataset outputs root directory')
args = parser.parse_args()

class follower(nn.Module):
    def __init__(self, args):
        super(follower, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

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
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.RandomAffine(
            # degrees=15, # degree
            # shear=(-0.15, 0.15, -0.15, 0.15),
        # ),
        transforms.Normalize(mean, std), 
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = ImageFolderWithFilename(root=args.data_directory+'/training', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_dataset = ImageFolderWithFilename(root=args.data_directory+'/testing', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    # Load model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model = follower(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model = follower(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Logger
    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logger = logging.getLogger(__name__)
    logname = model.__class__.__name__ + '_' + args.sess + \
        '_' + str(args.seed) + '.log'
    logfile = os.path.join(result_folder, logname)
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
    )
    logger.info(args)
    # Training
    def train(epoch):
        print('\nEpoch: {:04}'.format(epoch))
        train_loss, correct, total = 0, 0, 0
        model.train()
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output_logit = model(data)
            loss = F.cross_entropy(output_logit, target)
            loss.backward()
            optimizer.step()
            preds = F.softmax(output_logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)

            train_loss += loss.item() * target.size(0)
            total += target.size(0)
            correct += (preds_top_class.view(target.shape) == target).sum().item()

        return (train_loss / batch_idx, 100. * correct / total)
    # Test
    def test(epoch):
        model.eval()
        # load the best state_dict
        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output_logit = model(data)
                loss = F.cross_entropy(output_logit, target)
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                test_loss += loss.item() * target.size(0)
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()
        
        return (test_loss / batch_idx, 100. * correct / total)
    # Save checkpoint
    def checkpoint(acc, epoch):
        print('==> Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/' + args.sess + '_' + str(args.seed) + '.pth'
        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            }, save_path)
    # Run
    logger.info('Epoch \t Seconds \t \t Train Loss \t Train Acc')
    start_train_time = time.time()
    for epoch in range(epoch_start, args.epochs):
        start_epoch_time = time.time()
        
        train_loss, train_acc = train(epoch)
        epoch_time = time.time()
        logger.info('%5d \t %7.1f \t \t %10.4f \t %9.4f',
            epoch, epoch_time - start_epoch_time, train_loss, train_acc)
        # Save checkpoint.
        if train_acc - prev_acc  > 0.1:
            prev_acc = train_acc
            checkpoint(train_acc, epoch)
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    logger.info('Test Loss \t Test Acc')
    test_loss, test_acc = test(epoch)
    logger.info('%9.4f \t %8.4f', test_loss, test_acc)
    print('==> Done.')

main()
