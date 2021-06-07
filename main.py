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
parser.add_argument('--batch_size', '-b', default=64,       type=int,   help='mini-batch size (default: 8)')
parser.add_argument('--epochs',           default=20,          type=int,   help='number of total epochs to run')
parser.add_argument('--n-grid-density', '-g',default=5,          type=int,   help='number control point by side')
parser.add_argument('--image-size',       default=28,          type=int,   help='input image size (default: 28 for MNIST)')
parser.add_argument('--data-directory',   default='./mnist_dis',type=str, help='dataset inputs root directory')
parser.add_argument('--output_directory', default='./mnist_stn',type=str, help='dataset outputs root directory')
parser.add_argument('--dataset1-name',    default='Distorted MNIST',type=str, help='dataset1 name')
parser.add_argument('--dataset2-name',    default='Distorted MNIST enhanced by STN',type=str, help='dataset2 name')
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

class localizer(nn.Module):
    def __init__(self, args, source_points):
        super(localizer, self).__init__()
        self.args = args
        n_output = args.n_grid_density**2 *2
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_output)
        # insert control point as bias
        bias = torch.flatten(source_points)
        self.fc2.bias.data.copy_(bias)
        self.fc2.weight.data.zero_()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = torch.tanh(x) # bounded
        return x.view(self.args.batch_size, -1, 2)

class affine_localizer(nn.Module):
    def __init__(self, args):
        super(affine_localizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 6)
        # Initialize the weights/bias with identity transformation
        self.fc2.bias.data.copy_(torch.tensor([0.5, 0, 0, 0, 0.5, 0], dtype=torch.float).to(device))
        self.fc2.weight.data.zero_()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class tps_warper(nn.Module):
    def __init__(self, args, target_control_points):
        super(tps_warper, self).__init__()
        self.args = args
        self.target_control_points = target_control_points
        N = target_control_points.size(0)
        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        # create target cordinate matrix
        HW = self.args.image_size * self.args.image_size
        target_coordinate = list(itertools.product(range(self.args.image_size), range(self.args.image_size)))
        target_coordinate = torch.Tensor(target_coordinate).to(device) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (self.args.image_size - 1) - 1
        X = X * 2 / (self.args.image_size - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1).to(device), target_coordinate
        ], dim = 1)
        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2).expand(self.args.batch_size, 3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix
    
    def forward(self, image, source_control_points):
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, self.padding_matrix], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        grid = source_coordinate.view(batch_size, self.args.image_size, self.args.image_size, 2)
        warped_image = F.grid_sample(image, grid, align_corners=True)
        return warped_image

class affine_warper(nn.Module):
    def __init__(self, args):
        super(affine_warper, self).__init__()
        self.args = args
    
    def forward(self, image, theta):
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        warped_image = F.grid_sample(image, grid, align_corners=False)
        return warped_image
    
class tps_stn(nn.Module):
    def __init__(self, args):
        super(tps_stn, self).__init__()
        self.args = args
        src_points = self.get_src_points(args.batch_size, args.n_grid_density).to(device)
        self.src_points = src_points.unsqueeze(0).repeat(args.batch_size, 1, 1)

        self.follower = follower(args)
        self.localizer = localizer(args, src_points)
        self.affine_localizer = affine_localizer(args)
        self.tps_warper = tps_warper(args, src_points)
        self.affine_warper = affine_warper(args)
        # self.tps_warper = tps_warper(args, self.src_points)

    # src_points create upon the n_grid_density is given
    def get_src_points(self, batch_size, n_grid_density=4, grid_span=0.9):
        src_points_1d = torch.linspace(-grid_span, grid_span, steps=n_grid_density)
        src_points_2d = torch.cartesian_prod(src_points_1d, src_points_1d)
        return src_points_2d
        
    def stn(self, x):
        kernel_points = self.localizer(x)
        theta = self.affine_localizer(x).view(-1, 2, 3)
        warped_x = self.tps_warper(x, kernel_points)
        warped_x = self.affine_warper(warped_x, theta)
        return warped_x
        
    def forward(self, x):
        x = self.stn(x)
        x = self.follower(x)
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
        model = tps_stn(args).to(device)
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
        model = tps_stn(args).to(device)
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
    # Adjust testset using STN
    def convert():
        model.eval()
        # load the best state_dict
        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        # create folders
        testing_directory = args.output_directory + '/testing/'
        if not os.path.exists(testing_directory):
            os.makedirs(testing_directory)
        for classes in range(10):
            if not os.path.exists(testing_directory + str(classes)):
                os.makedirs(testing_directory + str(classes))
        print('==> Converting testing dataset using STN..')
        mean, std = (0.1307,), (0.3081,)
        inv_normalize = transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
        )
        with torch.no_grad():
            for batch_idx, (data, target, filename) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output_image = model.stn(data)
                output_image = inv_normalize(output_image)
                output_image = torch.clamp(output_image, 0, 1)
                
                for index, f in enumerate(filename):
                    image = output_image[index].detach().cpu()
                    f = testing_directory + f # it includes last dirname (same as target)
                    torchvision.utils.save_image(image, f)
    # visualize
    def visualize():
        result_folder = './results/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        model.eval()
        # load the best state_dict
        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        mean, std = (0.1307,), (0.3081,)
        inv_normalize = transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
        )
        with torch.no_grad():
            data = next(iter(test_loader))[0].to(device)
            input_image = inv_normalize(data)
            input_image = torch.clamp(input_image, 0, 1)
            output_image = model.stn(data)
            output_image = inv_normalize(output_image)
            output_image = torch.clamp(output_image, 0, 1)
            in_grid = torchvision.utils.make_grid(input_image.detach().cpu()).permute(1,2,0).numpy()
            out_grid = torchvision.utils.make_grid(output_image.detach().cpu()).permute(1,2,0).numpy()
            fig, axs = plt.subplots(1, 2, figsize=(24, 12))
            axs[0].imshow(in_grid)
            axs[0].set_title(args.dataset1_name)
            axs[1].imshow(out_grid)
            axs[1].set_title(args.dataset2_name)
            plt.show()
            plt.savefig(result_folder+"/__"+args.dataset1_name+"__"+args.dataset2_name)
            plt.close()

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

    # Converting testing dataset using STN..
    # convert()
    print('==> Done.')
    
    visualize()

if __name__ == "__main__":
    main()