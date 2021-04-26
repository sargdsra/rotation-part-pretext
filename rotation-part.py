import torch
import torch.optim as optim
import torch.nn as nn

from dataloader import RotationPartLoader
from model import Network
from train import train_model


data_file = 'images.txt'
num_parts = 4
num_angles = 4
lr = 0.001
batch_size = 64
epochs = 200
model_path = None
filename = 'checkpoint_rotation_part_resnet50.pth.tar'
log_filename = 'log_rotation_part_resnet50.log'
use_gpu = torch.cuda.is_available()
num_workers = 3
shuffle = True
device = torch.device("cuda" if use_gpu else "cpu")

dataset = RotationPartLoader(data_file, num_parts, num_angles)
train_loader = torch.utils.data.DataLoader(dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers)

if model_path is not None:
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    net = checkpoint['model']
    optimizer = checkpoint['optimizer']
else:
    net = Network(num_outputs = num_parts * num_angles)
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    start_epoch = 0

net = net.to(device)
criterion = nn.CrossEntropyLoss()
train_model(net, criterion, optimizer, start_epoch, epochs, dataset, train_loader, device, filename, log_filename)