import argparse
import sys
import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from my_cnn import Model

import mltracker

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')   


### Training
def train(net, device, train_loader, optimizer):
    net.train() # set the network in training mode

    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # convert data type
        optimizer.zero_grad() # clears gradients
        output = net(data) # forwards layers in the network
        loss = torch.nn.functional.cross_entropy(output, target) # compute loss
        train_loss += loss.item()
        loss.backward() # compute gradient
        optimizer.step() # gradient back-propagation

    train_loss /= len(train_loader)
    return train_loss


### Testing
def test(net, device, test_loader):
    net.eval() # sets the network in test mode
    test_loss, true_positives = 0, 0

    with torch.no_grad(): # do not compute backward (gradients)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            true_positives += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * true_positives / len(test_loader.dataset)
    return test_loss, accuracy


### Main
def main(args):
    # Set arguments
    batch = args.batch      # batch size
    lr = args.lr            # learning rate
    epochs = args.epochs    # number of training epochs
    
    mltracker.log_param('batch', batch)
    mltracker.log_param('lr', lr)
    mltracker.log_param('epochs', epochs)
    mltracker.set_version('pytorch-cnn')
    
    print('# Hyper parameter: epochs: %d, batch size: %.4f, learning rate: %.4f' % (epochs, batch, lr))

    # Set experiment environments
    torch.manual_seed(3) # set random seed for reproducability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load database
    print("# Load Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True
    )

    testset = datasets.CIFAR10(
        root=args.data_dir, train=False, download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False
    )

    # Initialize the network architecture and training optimizer
    print('# Init network')
    net = Model().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # Start training and validation
    print('# Start training')
    for epoch in range(1, epochs + 1):
        train_loss = train(net, device, train_loader, optimizer) # train
        test_loss, accuracy = test(net, device, test_loader) # validate
        print(' == Epoch %02d ' % (epoch))
        print('  Train/Test Loss: %.6f / %.6f' % (train_loss, test_loss))
        print('  Test Accuracy: %.2f' % (accuracy))
        mltracker.log_metric('accracy', accuracy)
        mltracker.log_metric('train_loss', train_loss)
        mltracker.log_metric('test_loss', test_loss)

    # Save trained weights and the model
    print('# Save model')
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(net.state_dict(), path)
    
    mltracker.log_file(path)
    mltracker.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparam
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)

    # Container environment
    parser.add_argument("--model-dir", type=str, default='.')
    parser.add_argument("--data-dir", type=str, default='./data')

    main(parser.parse_args())
