import argparse
import sys
import os
import os.path as osp

import boto3
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

### Load image database
# Torch provides CIFAR10 DB
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
def loadDataBase(path=osp.join('.','cifar10_data')):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( # normalize image intensities
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_db = datasets.CIFAR10(path,  # path to DB
                            train=True,  # data type flag
                            download=True, # can download if not exist
                            transform=trans # image transform
                            )
    test_db = datasets.CIFAR10(path, train=False, download=True, transform=trans)
    return train_db, test_db

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

### Save model to S3
def saveS3(local_file, args):
    s3 = boto3.client('s3', aws_access_key_id=args.access_key,
                      aws_secret_access_key=args.secret_key)
    version = 'cifar10-' + datetime.now().strftime("%y%m%d-%H%M%S") + '.pt'
    s3.upload_file(local_file, args.bucket, version)

### Main
def main(args):
    """ Main function
    """
    # Set arguments
    batch = args.batch      # batch size
    lr = args.lr            # learning rate
    epochs = args.epochs    # number of training epochs

    # Set experiment environments
    torch.manual_seed(3) # set random seed for reproducability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load database
    print('# Load database')
    train_db, test_db = loadDataBase()
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_db,
                                            batch_size=batch,
                                            shuffle=True)

    # Initialize the network architecture and training optimizer
    print('# Init network')
    net = models.resnet50().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # Start training and validation
    print('# Start training')
    for epoch in range(1, epochs + 1):
        train_loss = train(net, device, train_loader, optimizer) # train
        test_loss, accuracy = test(net, device, test_loader) # validate
        print(' == Epoch %02d ' % (epoch))
        print('  Train/Test Loss: %.6f / %.6f' % (train_loss, test_loss))
        print('  Test Accuracy: %.2f' % (accuracy))

    # Save trained weights and the model
    print('# Save model')
    path = os.path.join(args.model_dir, "cifar10_cnn.pt")
    torch.save(net.state_dict(), path)
    saveS3(path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparam
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.getenv('SM_MODEL_DIR', '.'))
    parser.add_argument("--access-key", type=str, default=os.getenv('ACCESS_KEY', '.'))
    parser.add_argument("--secret-key", type=str, default=os.getenv('SECRET_KEY', '.'))
    parser.add_argument("--bucket", type=str, default=os.getenv('BUCKET', '.'))

    main(parser.parse_args())
